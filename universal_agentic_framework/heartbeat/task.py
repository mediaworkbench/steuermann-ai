"""Heartbeat task base class — the four-phase tick lifecycle.

Each beat runs every registered task's :meth:`tick`, which orchestrates the
``observe → reason → act`` phases and records the outcome. The phases are no-op
hooks today; concrete tasks override them once real observation/decision/action
is wired in. ``tick`` never raises — every outcome (ok / skipped / error) is
recorded and logged so the heartbeat keeps beating.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from backend.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class RunStore(Protocol):
    """Minimal interface a run-history store must satisfy (see ``HeartbeatRunStore``)."""

    def record_run(
        self, *, task_name: str, status: str, duration_ms: int = 0, detail: Optional[Dict[str, Any]] = None
    ) -> None: ...

    def recent_runs(self, task_name: str, limit: int = 5) -> list[Dict[str, Any]]: ...


class HeartbeatTask:
    """Base class for heartbeat tasks.

    Subclasses override :meth:`observe`, :meth:`reason`, and :meth:`act`. The
    default implementations are no-ops, so a bare ``HeartbeatTask`` simply
    records an ``ok`` tick — useful as a liveness probe.
    """

    def __init__(
        self,
        *,
        name: str,
        cooldown_seconds: int = 0,
        run_store: Optional[RunStore] = None,
    ) -> None:
        self.name = name
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self._run_store = run_store
        # Scaffold for future external calls: when observe() starts hitting flaky
        # APIs, repeated failures trip the breaker instead of acting on bad data.
        self._breaker = AsyncCircuitBreaker(
            f"heartbeat:{name}", CircuitBreakerConfig()
        )

    # --- Overridable phases (no-ops for now) -------------------------------

    async def observe(self) -> Any:
        """Gather fresh observations. Override to query APIs/DB/etc."""
        return None

    async def reason(self, observations: Any) -> Any:
        """Decide what (if anything) to do. Override to call the model."""
        return None

    async def act(self, decision: Any) -> None:
        """Execute the decision. Override to perform real actions."""
        return None

    # --- Lifecycle ----------------------------------------------------------

    async def tick(self) -> Dict[str, Any]:
        """Run one beat for this task. Never raises."""
        if self.cooldown_seconds > 0 and await self._within_cooldown():
            await self._record("skipped", 0, {"reason": "cooldown"})
            logger.info("heartbeat_tick", task=self.name, status="skipped", reason="cooldown")
            return {"status": "skipped"}

        start = time.monotonic()

        # Observe — guarded by the circuit breaker. A failure here means we do
        # NOT reason or act on incomplete/corrupt data (article: "fail gracefully").
        try:
            observations = await self._breaker.call(self.observe)
        except CircuitBreakerOpenError as exc:
            return await self._fail(start, "observe", exc, breaker_open=True)
        except Exception as exc:  # noqa: BLE001 — outcome is recorded, never propagated
            return await self._fail(start, "observe", exc)

        # Reason + act.
        try:
            decision = await self.reason(observations)
            await self.act(decision)
        except Exception as exc:  # noqa: BLE001
            return await self._fail(start, "reason_act", exc)

        duration_ms = int((time.monotonic() - start) * 1000)
        await self._record("ok", duration_ms, {})
        logger.info("heartbeat_tick", task=self.name, status="ok", duration_ms=duration_ms)
        return {"status": "ok", "duration_ms": duration_ms}

    # --- Internals ----------------------------------------------------------

    async def _fail(
        self, start: float, phase: str, exc: Exception, *, breaker_open: bool = False
    ) -> Dict[str, Any]:
        duration_ms = int((time.monotonic() - start) * 1000)
        detail = {"phase": phase, "error": str(exc)}
        if breaker_open:
            detail["circuit_breaker"] = "open"
        await self._record("error", duration_ms, detail)
        logger.warning(
            "heartbeat_tick",
            task=self.name,
            status="error",
            phase=phase,
            error=str(exc),
        )
        return {"status": "error", "phase": phase}

    async def _within_cooldown(self) -> bool:
        if not self._run_store:
            return False
        try:
            rows = await asyncio.to_thread(self._run_store.recent_runs, self.name, 1)
        except Exception as exc:  # noqa: BLE001 — cooldown is best-effort
            logger.warning("heartbeat_cooldown_check_failed", task=self.name, error=str(exc))
            return False
        if not rows:
            return False
        last = rows[0]
        if last.get("status") != "ok":
            return False
        fired_at = last.get("fired_at")
        if not isinstance(fired_at, datetime):
            return False
        if fired_at.tzinfo is None:
            fired_at = fired_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - fired_at).total_seconds()
        return age_seconds < self.cooldown_seconds

    async def _record(self, status: str, duration_ms: int, detail: Dict[str, Any]) -> None:
        if not self._run_store:
            return
        try:
            await asyncio.to_thread(
                self._run_store.record_run,
                task_name=self.name,
                status=status,
                duration_ms=duration_ms,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 — observability write must never break a beat
            logger.warning("heartbeat_run_record_failed", task=self.name, error=str(exc))
