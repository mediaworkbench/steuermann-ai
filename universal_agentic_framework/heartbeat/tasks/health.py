"""A trivial heartbeat task that proves the beat loop end-to-end.

It performs no external calls — :meth:`observe` just reports liveness — so every
beat records an ``ok`` run with an ``"alive"`` marker. Use it as the reference
implementation when adding real tasks: override ``observe``/``reason``/``act``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from universal_agentic_framework.heartbeat.task import HeartbeatTask


class HealthHeartbeatTask(HeartbeatTask):
    """Records an ``"alive"`` tick on every beat (no external integration)."""

    async def observe(self) -> Any:
        return {"alive": True, "observed_at": datetime.now(timezone.utc).isoformat()}
