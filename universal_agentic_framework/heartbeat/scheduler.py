"""Heartbeat scheduler — one global beat, embedded in the LangGraph service.

Mirrors ``caching/scheduler.py`` (``CacheScheduler``): a plain in-memory
``AsyncIOScheduler`` running inside the uvicorn event loop. Two jobs:

* ``heartbeat_beat`` — fires every *N* minutes and *enqueues* this beat's work
  (one item per ``global`` task, one per active user for ``per_user`` tasks); a
  bounded worker pool drains the queue and runs each task's ``tick``.
* ``heartbeat_control`` — fires every 30s, reads the admin-configured rate from
  Postgres (written via FastAPI, a separate process) and reschedules the beat
  **only when the rate actually changed**, and periodically prunes run history.

The schedule is rebuilt deterministically from config + the admin rate on every
startup, so no durable jobstore is needed. Task ``observe``/``reason``/``act``
phases must be stateless (per-tick state flows through ``TickContext``), since a
single task instance is run concurrently for many users by the worker pool.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.db import HEARTBEAT_COOLDOWNS_SETTING_KEY, HEARTBEAT_RATE_SETTING_KEY
from universal_agentic_framework.config.schemas import HeartbeatSettings
from universal_agentic_framework.heartbeat.task import HeartbeatTask, RunStore, TickContext
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)

BEAT_JOB_ID = "heartbeat_beat"
CONTROL_JOB_ID = "heartbeat_control"
CONTROL_INTERVAL_SECONDS = 30
MIN_RATE_MINUTES = 1
MAX_RATE_MINUTES = 1440  # 24h
MIN_COOLDOWN_SECONDS = 0
MAX_COOLDOWN_SECONDS = 604800  # 7 days

# Per-user fan-out is drained by a bounded worker pool — the worker count is the
# throttle (so a large user set never hammers the LLM all at once). The queue is
# unbounded (items are cheap (task, user_id) tuples); QUEUE_HIGH_WATER is a
# safety valve: if a beat finds this much still queued, the previous beat hasn't
# drained, so it skips its enqueue and warns instead of piling on.
HEARTBEAT_WORKERS = 4
QUEUE_HIGH_WATER = 5000

# Run-history retention: pruned periodically from the control job so per-user
# fan-out can't grow heartbeat_runs without bound.
RUN_RETENTION_DAYS = 7
PRUNE_EVERY_CONTROL_TICKS = 120  # ~ every hour at a 30s control cadence

# A queued unit of work: the task to run and the user it runs for (None = global).
WorkItem = Tuple[HeartbeatTask, Optional[str]]


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
    """Build run + settings + user + cognitive-memory stores from one DB pool.

    Best-effort: returns all-None on failure so the heartbeat never blocks boot.
    The three cognitive stores (procedural / conflicts / audit) are the Dreaming
    Engine's data contracts (plan-memory.md Phase 1); they ride the same pool.
    """
    try:
        from backend.db import (
            GlobalSettingsStore,
            HeartbeatRunStore,
            MemoryAuditLogStore,
            MemoryConflictStore,
            ProceduralOverrideStore,
            UserStore,
            init_db_pool,
        )

        pool = init_db_pool()
        return (
            HeartbeatRunStore(pool),
            GlobalSettingsStore(pool),
            UserStore(pool),
            ProceduralOverrideStore(pool),
            MemoryConflictStore(pool),
            MemoryAuditLogStore(pool),
        )
    except Exception as exc:  # noqa: BLE001 — heartbeat must not block boot
        logger.warning("heartbeat_stores_unavailable", error=str(exc))
        return None, None, None, None, None, None


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
        user_store=None,
        procedural_store=None,
        conflict_store=None,
        audit_store=None,
        build_default_stores: bool = True,
    ) -> None:
        self._config = config
        self._run_store = run_store
        self._settings_store = settings_store
        self._user_store = user_store
        # Dreaming Engine data contracts (plan-memory.md Phase 1); consumed by the
        # per-user DreamingEngineTask added in Phase 3.
        self._procedural_store = procedural_store
        self._conflict_store = conflict_store
        self._audit_store = audit_store
        if (
            build_default_stores
            and run_store is None
            and settings_store is None
            and user_store is None
            and procedural_store is None
            and conflict_store is None
            and audit_store is None
        ):
            (
                self._run_store,
                self._settings_store,
                self._user_store,
                self._procedural_store,
                self._conflict_store,
                self._audit_store,
            ) = _build_default_stores()
        self.scheduler = AsyncIOScheduler()
        self._tasks: List[HeartbeatTask] = []
        self._running = False
        # Per-user fan-out queue + worker pool (created on start(), on the loop).
        self._queue: "asyncio.Queue[WorkItem]" = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._control_ticks = 0

    @property
    def running(self) -> bool:
        return self._running

    # --- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            logger.warning("heartbeat_already_running")
            return

        self._tasks = self._build_tasks()
        # Remember the config cooldown per task so an admin override can be applied
        # live AND cleared back to the configured default.
        self._config_cooldowns = {t.name: int(t.cooldown_seconds) for t in self._config.tasks}
        self._sync_cooldowns()
        rate = self._effective_rate()

        # Drain the fan-out queue with a bounded worker pool (the throttle). These
        # are plain asyncio tasks on the server loop, independent of APScheduler.
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"heartbeat-worker-{i}")
            for i in range(HEARTBEAT_WORKERS)
        ]

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
            workers=len(self._workers),
            tasks=[{"name": t.name, "scope": t.scope} for t in self._tasks],
        )

    def stop(self) -> None:
        if not self._running:
            return
        for worker in self._workers:
            worker.cancel()
        self._workers = []
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
        """Enqueue this beat's work (non-blocking); workers run it.

        Global tasks enqueue one item; per-user tasks fan out one item per active
        user. If a previous beat is still draining (queue above the high-water
        mark) we skip rather than pile on — the next beat catches up, and per-user
        cooldown keeps the skipped users from being shortchanged.
        """
        backlog = self._queue.qsize()
        if backlog >= QUEUE_HIGH_WATER:
            logger.warning("heartbeat_beat_skipped_backlog", queued=backlog)
            return

        enqueued = 0
        for task in self._tasks:
            try:
                if task.scope == "per_user":
                    enqueued += await self._enqueue_per_user(task)
                else:
                    self._queue.put_nowait((task, None))
                    enqueued += 1
            except Exception as exc:  # noqa: BLE001 — one task can't break the beat
                logger.warning("heartbeat_task_enqueue_failed", task=task.name, error=str(exc))
        logger.info("heartbeat_beat_enqueued", items=enqueued)

    async def _enqueue_per_user(self, task: HeartbeatTask) -> int:
        if self._user_store is None:
            return 0
        recency = getattr(task, "active_user_recency_days", None)
        try:
            if recency is not None and hasattr(self._user_store, "get_recently_active_user_ids"):
                user_ids = await asyncio.to_thread(
                    self._user_store.get_recently_active_user_ids, recency
                )
            else:
                user_ids = await asyncio.to_thread(self._user_store.get_active_user_ids)
        except Exception as exc:  # noqa: BLE001 — best-effort; skip this task this beat
            logger.warning("heartbeat_user_list_failed", task=task.name, error=str(exc))
            return 0
        for user_id in user_ids:
            self._queue.put_nowait((task, user_id))
        return len(user_ids)

    async def _worker(self, index: int) -> None:
        """Drain (task, user_id) items forever. A worker never dies on a tick error."""
        while True:
            task, user_id = await self._queue.get()
            try:
                await task.tick(TickContext(user_id=user_id))
            except Exception as exc:  # noqa: BLE001 — tick records its own errors
                logger.warning(
                    "heartbeat_task_failed", task=task.name, user_id=user_id, error=str(exc)
                )
            finally:
                self._queue.task_done()

    async def _control(self) -> None:
        """Reschedule the beat when the admin rate changed; apply per-task cooldown
        overrides; prune old runs periodically."""
        self._control_ticks += 1
        if self._control_ticks % PRUNE_EVERY_CONTROL_TICKS == 0:
            await self._prune_runs()

        # Per-task cooldown overrides are picked up live (no restart), like the rate.
        await asyncio.to_thread(self._sync_cooldowns)

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

    async def _prune_runs(self) -> None:
        if self._run_store is None or not hasattr(self._run_store, "prune_runs"):
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=RUN_RETENTION_DAYS)
        try:
            pruned = await asyncio.to_thread(self._run_store.prune_runs, cutoff=cutoff)
        except Exception as exc:  # noqa: BLE001 — retention is best-effort
            logger.warning("heartbeat_prune_failed", error=str(exc))
            return
        if pruned:
            logger.info("heartbeat_runs_pruned", pruned=pruned)

    # --- Helpers ------------------------------------------------------------

    def _build_tasks(self) -> List[HeartbeatTask]:
        tasks: List[HeartbeatTask] = []
        for tcfg in self._config.tasks:
            if not tcfg.enabled:
                continue
            try:
                cls = _import_entry_point(tcfg.type)
            except Exception as exc:  # noqa: BLE001 — skip a bad task, keep the rest
                logger.warning("heartbeat_task_load_failed", task=tcfg.name, type=tcfg.type, error=str(exc))
                continue

            if self._is_dreaming_class(cls):
                task = self._build_dreaming_task(tcfg)
                if task is not None:
                    tasks.append(task)
                continue

            try:
                tasks.append(
                    cls(
                        name=tcfg.name,
                        cooldown_seconds=tcfg.cooldown_seconds,
                        run_store=self._run_store,
                        scope=tcfg.scope,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — skip a bad task, keep the rest
                logger.warning("heartbeat_task_load_failed", task=tcfg.name, type=tcfg.type, error=str(exc))
        return tasks

    @staticmethod
    def _is_dreaming_class(cls: Any) -> bool:
        try:
            from universal_agentic_framework.heartbeat.tasks.dreaming import DreamingEngineTask

            return isinstance(cls, type) and issubclass(cls, DreamingEngineTask)
        except Exception:  # noqa: BLE001
            return False

    def _build_dreaming_task(self, tcfg) -> Optional[HeartbeatTask]:
        """Construct the Dreaming Engine task — gated by the feature flag and its
        data contracts (audit/conflict stores). Returns None (skips) when off or
        when its stores/backend are unavailable, never blocking the heartbeat."""
        try:
            from universal_agentic_framework.config import load_features_config

            features = load_features_config()
            if not getattr(features, "dreaming_engine_enabled", False):
                logger.info("dreaming_task_disabled_by_flag", task=tcfg.name)
                return None
        except Exception as exc:  # noqa: BLE001 — no flag access → stay off
            logger.warning("dreaming_flag_read_failed", task=tcfg.name, error=str(exc))
            return None

        if self._audit_store is None or self._conflict_store is None:
            logger.warning("dreaming_stores_unavailable", task=tcfg.name)
            return None

        try:
            from universal_agentic_framework.heartbeat.tasks.dreaming import build_dreaming_task

            return build_dreaming_task(
                name=tcfg.name,
                cooldown_seconds=tcfg.cooldown_seconds,
                scope=tcfg.scope,
                run_store=self._run_store,
                audit_store=self._audit_store,
                conflict_store=self._conflict_store,
                procedural_store=self._procedural_store,
            )
        except Exception as exc:  # noqa: BLE001 — backend/config failure → skip task
            logger.warning("dreaming_task_build_failed", task=tcfg.name, error=str(exc))
            return None

    def _effective_cooldowns(self) -> dict:
        """Admin per-task cooldown overrides from global_settings (clamped). Missing
        tasks fall back to their config cooldown in ``_sync_cooldowns``."""
        if not self._settings_store:
            return {}
        try:
            raw = self._settings_store.get_setting(HEARTBEAT_COOLDOWNS_SETTING_KEY)
        except Exception as exc:  # noqa: BLE001 — never crash the control loop
            logger.warning("heartbeat_cooldown_read_failed", error=str(exc))
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict = {}
        for name, value in raw.items():
            try:
                out[str(name)] = max(MIN_COOLDOWN_SECONDS, min(MAX_COOLDOWN_SECONDS, int(value)))
            except (TypeError, ValueError):
                continue
        return out

    def _sync_cooldowns(self) -> None:
        """Apply admin cooldown overrides to the live task instances (override wins,
        else the config default). ``_within_cooldown`` reads ``cooldown_seconds`` per
        tick, so updating the attribute takes effect on the next beat — no restart."""
        overrides = self._effective_cooldowns()
        defaults = getattr(self, "_config_cooldowns", {})
        for task in self._tasks:
            desired = overrides.get(task.name, defaults.get(task.name, task.cooldown_seconds))
            if int(task.cooldown_seconds) != int(desired):
                logger.info(
                    "heartbeat_cooldown_applied",
                    task=task.name,
                    from_seconds=task.cooldown_seconds,
                    to_seconds=int(desired),
                )
                task.cooldown_seconds = int(desired)

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
