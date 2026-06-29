"""Heartbeat task base class — the four-phase tick lifecycle.

Each beat runs every registered task's :meth:`tick`, which orchestrates the
``observe → reason → act`` phases and records the outcome. The phases are no-op
hooks today; concrete tasks override them once real observation/decision/action
is wired in. ``tick`` never raises — every outcome (ok / skipped / error) is
recorded and logged so the heartbeat keeps beating.

A task may be ``global`` (run once per beat) or ``per_user`` (fanned out once per
active user each beat). The per-user ``user_id`` flows through the lifecycle via
:class:`TickContext`: it scopes the run record, the cooldown lookup, and the
circuit breaker so one user can never poison the rest of the fleet.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from backend.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TickContext:
    """Per-tick context. ``user_id`` is ``None`` for global-scope tasks."""

    user_id: Optional[str] = None


class RunStore(Protocol):
    """Minimal interface a run-history store must satisfy (see ``HeartbeatRunStore``)."""

    def record_run(
        self,
        *,
        task_name: str,
        status: str,
        duration_ms: int = 0,
        detail: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None: ...

    def recent_runs(
        self, task_name: str, limit: int = 5, *, user_id: Optional[str] = None
    ) -> list[Dict[str, Any]]: ...


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
        scope: str = "global",
    ) -> None:
        self.name = name
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self._run_store = run_store
        self.scope = scope
        # Scaffold for future external calls: when observe() starts hitting flaky
        # APIs, repeated failures trip the breaker instead of acting on bad data.
        # Keyed per user_id (None for global) so one user's failures don't trip
        # the breaker for everyone else under a per-user task.
        self._breakers: Dict[Optional[str], AsyncCircuitBreaker] = {}

    def _breaker_for(self, user_id: Optional[str]) -> AsyncCircuitBreaker:
        breaker = self._breakers.get(user_id)
        if breaker is None:
            suffix = f":{user_id}" if user_id is not None else ""
            breaker = AsyncCircuitBreaker(
                f"heartbeat:{self.name}{suffix}", CircuitBreakerConfig()
            )
            self._breakers[user_id] = breaker
        return breaker

    # --- Overridable phases (no-ops for now) -------------------------------

    async def observe(self, ctx: TickContext) -> Any:
        """Gather fresh observations. Override to query APIs/DB/etc."""
        return None

    async def reason(self, ctx: TickContext, observations: Any) -> Any:
        """Decide what (if anything) to do. Override to call the model."""
        return None

    async def act(self, ctx: TickContext, decision: Any) -> None:
        """Execute the decision. Override to perform real actions."""
        return None

    # --- Lifecycle ----------------------------------------------------------

    async def tick(self, ctx: Optional[TickContext] = None) -> Dict[str, Any]:
        """Run one beat for this task. Never raises."""
        if ctx is None:
            ctx = TickContext()

        if self.cooldown_seconds > 0 and await self._within_cooldown(ctx):
            await self._record(ctx, "skipped", 0, {"reason": "cooldown"})
            logger.info(
                "heartbeat_tick", task=self.name, user_id=ctx.user_id,
                status="skipped", reason="cooldown",
            )
            return {"status": "skipped"}

        start = time.monotonic()

        # Observe — guarded by the circuit breaker. A failure here means we do
        # NOT reason or act on incomplete/corrupt data (article: "fail gracefully").
        try:
            observations = await self._breaker_for(ctx.user_id).call(self.observe, ctx)
        except CircuitBreakerOpenError as exc:
            return await self._fail(ctx, start, "observe", exc, breaker_open=True)
        except Exception as exc:  # noqa: BLE001 — outcome is recorded, never propagated
            return await self._fail(ctx, start, "observe", exc)

        # Reason + act.
        try:
            decision = await self.reason(ctx, observations)
            await self.act(ctx, decision)
        except Exception as exc:  # noqa: BLE001
            return await self._fail(ctx, start, "reason_act", exc)

        duration_ms = int((time.monotonic() - start) * 1000)
        await self._record(ctx, "ok", duration_ms, {})
        logger.info(
            "heartbeat_tick", task=self.name, user_id=ctx.user_id,
            status="ok", duration_ms=duration_ms,
        )
        return {"status": "ok", "duration_ms": duration_ms}

    # --- Internals ----------------------------------------------------------

    async def _fail(
        self, ctx: TickContext, start: float, phase: str, exc: Exception, *, breaker_open: bool = False
    ) -> Dict[str, Any]:
        duration_ms = int((time.monotonic() - start) * 1000)
        detail = {"phase": phase, "error": str(exc)}
        if breaker_open:
            detail["circuit_breaker"] = "open"
        await self._record(ctx, "error", duration_ms, detail)
        logger.warning(
            "heartbeat_tick",
            task=self.name,
            user_id=ctx.user_id,
            status="error",
            phase=phase,
            error=str(exc),
        )
        return {"status": "error", "phase": phase}

    async def _within_cooldown(self, ctx: TickContext) -> bool:
        if not self._run_store:
            return False
        try:
            rows = await asyncio.to_thread(
                self._run_store.recent_runs, self.name, 1, user_id=ctx.user_id
            )
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

    async def _record(
        self, ctx: TickContext, status: str, duration_ms: int, detail: Dict[str, Any]
    ) -> None:
        if not self._run_store:
            return
        try:
            await asyncio.to_thread(
                self._run_store.record_run,
                task_name=self.name,
                status=status,
                duration_ms=duration_ms,
                detail=detail,
                user_id=ctx.user_id,
            )
        except Exception as exc:  # noqa: BLE001 — observability write must never break a beat
            logger.warning("heartbeat_run_record_failed", task=self.name, error=str(exc))
