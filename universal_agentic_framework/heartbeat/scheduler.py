"""Heartbeat scheduler — one global beat, embedded in the LangGraph service.

Mirrors ``caching/scheduler.py`` (``CacheScheduler``): a plain in-memory
``AsyncIOScheduler`` running inside the uvicorn event loop. Two jobs:

* ``heartbeat_beat`` — fires every *N* minutes and runs every registered task.
* ``heartbeat_control`` — fires every 30s, reads the admin-configured rate from
  Postgres (written via FastAPI, a separate process) and reschedules the beat
  **only when the rate actually changed**.

The schedule is rebuilt deterministically from config + the admin rate on every
startup, so no durable jobstore is needed.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta
from typing import List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.db import HEARTBEAT_RATE_SETTING_KEY
from universal_agentic_framework.config.schemas import HeartbeatSettings
from universal_agentic_framework.heartbeat.task import HeartbeatTask, RunStore
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)

BEAT_JOB_ID = "heartbeat_beat"
CONTROL_JOB_ID = "heartbeat_control"
CONTROL_INTERVAL_SECONDS = 30
MIN_RATE_MINUTES = 1
MAX_RATE_MINUTES = 1440  # 24h


def _import_entry_point(entry_point: str):
    """Resolve a ``module.path:ClassName`` entry point (same shape tools use)."""
    module_path, _, attr = entry_point.partition(":")
    if not module_path or not attr:
        raise ValueError(f"Invalid heartbeat task type: {entry_point!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr, None)
    if cls is None:
        raise ImportError(f"Cannot find {attr} in {module_path}")
    return cls


def _build_default_stores():
    """Build run + settings stores from a single DB pool (best-effort)."""
    try:
        from backend.db import GlobalSettingsStore, HeartbeatRunStore, init_db_pool

        pool = init_db_pool()
        return HeartbeatRunStore(pool), GlobalSettingsStore(pool)
    except Exception as exc:  # noqa: BLE001 — heartbeat must not block boot
        logger.warning("heartbeat_stores_unavailable", error=str(exc))
        return None, None


def _coerce_rate(value, default: int) -> int:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_RATE_MINUTES, min(MAX_RATE_MINUTES, rate))


def _trigger_minutes(trigger) -> Optional[int]:
    interval = getattr(trigger, "interval", None)
    if isinstance(interval, timedelta):
        return int(round(interval.total_seconds() / 60))
    return None


class HeartbeatScheduler:
    """Owns the AsyncIOScheduler and the heartbeat task registry."""

    def __init__(
        self,
        config: HeartbeatSettings,
        *,
        run_store: Optional[RunStore] = None,
        settings_store=None,
        build_default_stores: bool = True,
    ) -> None:
        self._config = config
        self._run_store = run_store
        self._settings_store = settings_store
        if build_default_stores and run_store is None and settings_store is None:
            self._run_store, self._settings_store = _build_default_stores()
        self.scheduler = AsyncIOScheduler()
        self._tasks: List[HeartbeatTask] = []
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    # --- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            logger.warning("heartbeat_already_running")
            return

        self._tasks = self._build_tasks()
        rate = self._effective_rate()

        self.scheduler.add_job(
            self._beat,
            IntervalTrigger(minutes=rate),
            id=BEAT_JOB_ID,
            name="Heartbeat beat",
            max_instances=1,   # a slow tick never overlaps the next beat
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._control,
            IntervalTrigger(seconds=CONTROL_INTERVAL_SECONDS),
            id=CONTROL_JOB_ID,
            name="Heartbeat control",
            max_instances=1,
            replace_existing=True,
        )
        self.scheduler.start()
        self._running = True
        logger.info(
            "heartbeat_started",
            rate_minutes=rate,
            tasks=[t.name for t in self._tasks],
        )

    def stop(self) -> None:
        if not self._running:
            return
        try:
            self.scheduler.shutdown(wait=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("heartbeat_shutdown_failed", error=str(exc))
        finally:
            self._running = False
            logger.info("heartbeat_stopped")

    def get_jobs(self) -> list:
        return [
            {"id": job.id, "name": job.name, "next_run_time": job.next_run_time, "trigger": str(job.trigger)}
            for job in self.scheduler.get_jobs()
        ]

    # --- Jobs ---------------------------------------------------------------

    async def _beat(self) -> None:
        for task in self._tasks:
            try:
                await task.tick()
            except Exception as exc:  # noqa: BLE001 — one task can't break the beat
                logger.warning("heartbeat_task_failed", task=task.name, error=str(exc))

    async def _control(self) -> None:
        """Reschedule the beat when the admin rate changed (else do nothing)."""
        desired = await asyncio.to_thread(self._effective_rate)
        job = self.scheduler.get_job(BEAT_JOB_ID)
        if job is None:
            return
        current = _trigger_minutes(job.trigger)
        if current == desired:
            return
        # IMPORTANT: only reschedule on a real change — an interval reschedule
        # resets next_run_time, so doing it every 30s would push the beat out
        # indefinitely and it would never fire.
        self.scheduler.reschedule_job(BEAT_JOB_ID, trigger=IntervalTrigger(minutes=desired))
        logger.info("heartbeat_rescheduled", from_minutes=current, to_minutes=desired)

    # --- Helpers ------------------------------------------------------------

    def _build_tasks(self) -> List[HeartbeatTask]:
        tasks: List[HeartbeatTask] = []
        for tcfg in self._config.tasks:
            if not tcfg.enabled:
                continue
            try:
                cls = _import_entry_point(tcfg.type)
                tasks.append(
                    cls(
                        name=tcfg.name,
                        cooldown_seconds=tcfg.cooldown_seconds,
                        run_store=self._run_store,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — skip a bad task, keep the rest
                logger.warning("heartbeat_task_load_failed", task=tcfg.name, type=tcfg.type, error=str(exc))
        return tasks

    def _effective_rate(self) -> int:
        """Admin override (global_settings) if set, else the config default."""
        default = int(self._config.default_rate_minutes)
        if not self._settings_store:
            return default
        try:
            value = self._settings_store.get_setting(HEARTBEAT_RATE_SETTING_KEY)
        except Exception as exc:  # noqa: BLE001 — fall back to config, never crash
            logger.warning("heartbeat_rate_read_failed", error=str(exc))
            return default
        if value is None:
            return default
        return _coerce_rate(value, default)
