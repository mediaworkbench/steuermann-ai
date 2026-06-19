"""Heartbeat subsystem — a virtual cron for proactive agents.

One global "beat" fires on a schedule; each beat runs the registered tasks
through a four-phase tick (observe → reason → act → log). For now the phases
are no-op hooks so future API/agent integration is drop-in. See
``docs/technical_architecture.md`` and the heartbeat section of a profile's
``core.yaml`` for configuration.
"""

from universal_agentic_framework.heartbeat.scheduler import HeartbeatScheduler
from universal_agentic_framework.heartbeat.task import HeartbeatTask

__all__ = ["HeartbeatScheduler", "HeartbeatTask"]
