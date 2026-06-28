"""A trivial heartbeat task that proves the beat loop end-to-end.

It performs no external calls — :meth:`observe` just reports liveness — so every
beat records an ``ok`` run with an ``"alive"`` marker. Use it as the reference
implementation when adding real tasks: override ``observe``/``reason``/``act``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from universal_agentic_framework.heartbeat.task import HeartbeatTask, TickContext


class HealthHeartbeatTask(HeartbeatTask):
    """Records an ``"alive"`` tick on every beat (no external integration).

    Works as both a ``global`` liveness probe and a ``per_user`` fan-out demo —
    the run row's ``user_id`` (not the observation) is what makes it per-user.
    """

    async def observe(self, ctx: TickContext) -> Any:
        return {
            "alive": True,
            "user_id": ctx.user_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
