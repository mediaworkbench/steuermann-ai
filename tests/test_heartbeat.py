"""Tests for the heartbeat (virtual cron) subsystem.

Covers the task tick lifecycle (ok / skipped / error), the scheduler's
effective-rate resolution + reschedule-only-on-change control logic, and the
admin heartbeat-rate endpoint. All pure unit tests — no Postgres, no live
scheduler timing (the beat interval is minutes, so it never fires mid-test).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal_agentic_framework.config.schemas import (
    HeartbeatSettings,
    HeartbeatTaskSettings,
)
from universal_agentic_framework.heartbeat.task import HeartbeatTask, TickContext
from universal_agentic_framework.heartbeat.scheduler import (
    BEAT_JOB_ID,
    CONTROL_JOB_ID,
    PRUNE_EVERY_CONTROL_TICKS,
    QUEUE_HIGH_WATER,
    HeartbeatScheduler,
    _coerce_rate,
    _trigger_minutes,
)

_HEALTH = "universal_agentic_framework.heartbeat.tasks.health:HealthHeartbeatTask"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeRunStore:
    def __init__(self, recent: Optional[List[Dict[str, Any]]] = None) -> None:
        self.records: List[Dict[str, Any]] = []
        self.pruned_cutoffs: List[Any] = []
        self._recent = list(recent or [])

    def record_run(self, *, task_name, status, duration_ms=0, detail=None, user_id=None):
        self.records.append(
            {
                "task_name": task_name,
                "user_id": user_id,
                "status": status,
                "duration_ms": duration_ms,
                "detail": detail or {},
            }
        )

    def recent_runs(self, task_name, limit=5, *, user_id=None):
        return list(self._recent)[:limit]

    def prune_runs(self, *, cutoff):
        self.pruned_cutoffs.append(cutoff)
        return 0


class _FakeUserStore:
    def __init__(self, user_ids: List[str]) -> None:
        self._user_ids = list(user_ids)

    def get_active_user_ids(self) -> List[str]:
        return list(self._user_ids)


class _FakeSettingsStore:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    def get_setting(self, key: str):
        return self.value


def _config(rate: int = 5, *, enabled: bool = True, tasks=None) -> HeartbeatSettings:
    return HeartbeatSettings(
        enabled=enabled,
        default_rate_minutes=rate,
        tasks=tasks
        if tasks is not None
        else [
            HeartbeatTaskSettings(
                name="health",
                type="universal_agentic_framework.heartbeat.tasks.health:HealthHeartbeatTask",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Task lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tick_ok_records_run():
    store = _FakeRunStore()
    result = await HeartbeatTask(name="t", run_store=store).tick()
    assert result["status"] == "ok"
    assert store.records[-1]["status"] == "ok"


@pytest.mark.asyncio
async def test_tick_runs_phases_in_order():
    class _Recording(HeartbeatTask):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.calls: List[str] = []

        async def observe(self, ctx):
            self.calls.append("observe")
            return {"x": 1}

        async def reason(self, ctx, observations):
            self.calls.append("reason")
            return observations

        async def act(self, ctx, decision):
            self.calls.append("act")

    task = _Recording(name="t", run_store=_FakeRunStore())
    await task.tick()
    assert task.calls == ["observe", "reason", "act"]


@pytest.mark.asyncio
async def test_observe_failure_records_error_and_skips_act():
    class _FailObserve(HeartbeatTask):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.acted = False

        async def observe(self, ctx):
            raise RuntimeError("boom")

        async def act(self, ctx, decision):
            self.acted = True

    store = _FakeRunStore()
    task = _FailObserve(name="t", run_store=store)
    result = await task.tick()
    assert result["status"] == "error"
    assert result["phase"] == "observe"
    assert task.acted is False
    assert store.records[-1]["status"] == "error"
    assert store.records[-1]["detail"]["phase"] == "observe"


@pytest.mark.asyncio
async def test_cooldown_skips_recent_ok_run():
    store = _FakeRunStore(recent=[{"status": "ok", "fired_at": datetime.now(timezone.utc)}])
    result = await HeartbeatTask(name="t", cooldown_seconds=300, run_store=store).tick()
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_cooldown_elapsed_runs_again():
    store = _FakeRunStore(
        recent=[{"status": "ok", "fired_at": datetime.now(timezone.utc) - timedelta(seconds=600)}]
    )
    result = await HeartbeatTask(name="t", cooldown_seconds=300, run_store=store).tick()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_tick_without_store_still_ok():
    result = await HeartbeatTask(name="t").tick()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_tick_records_user_id_for_per_user_run():
    store = _FakeRunStore()
    result = await HeartbeatTask(name="t", run_store=store).tick(TickContext(user_id="u42"))
    assert result["status"] == "ok"
    assert store.records[-1]["user_id"] == "u42"


@pytest.mark.asyncio
async def test_per_user_cooldown_isolation():
    """A recent ok run for one user must not put a different user on cooldown."""

    class _PerUserStore(_FakeRunStore):
        def recent_runs(self, task_name, limit=5, *, user_id=None):
            if user_id == "busy":
                return [{"status": "ok", "fired_at": datetime.now(timezone.utc)}]
            return []

    store = _PerUserStore()
    task = HeartbeatTask(name="t", cooldown_seconds=300, run_store=store, scope="per_user")
    busy = await task.tick(TickContext(user_id="busy"))
    fresh = await task.tick(TickContext(user_id="fresh"))
    assert busy["status"] == "skipped"
    assert fresh["status"] == "ok"


# --------------------------------------------------------------------------- #
# Scheduler helpers
# --------------------------------------------------------------------------- #
def test_effective_rate_uses_config_default():
    sched = HeartbeatScheduler(_config(rate=7), build_default_stores=False)
    assert sched._effective_rate() == 7


def test_effective_rate_admin_override_wins():
    sched = HeartbeatScheduler(
        _config(rate=7), settings_store=_FakeSettingsStore(value=15), build_default_stores=False
    )
    assert sched._effective_rate() == 15


def test_effective_rate_override_clamped():
    sched = HeartbeatScheduler(
        _config(rate=7), settings_store=_FakeSettingsStore(value=99999), build_default_stores=False
    )
    assert sched._effective_rate() == 1440


def test_effective_rate_bad_override_falls_back():
    sched = HeartbeatScheduler(
        _config(rate=7), settings_store=_FakeSettingsStore(value="garbage"), build_default_stores=False
    )
    assert sched._effective_rate() == 7


def test_build_tasks_resolves_entry_point():
    sched = HeartbeatScheduler(_config(), build_default_stores=False)
    tasks = sched._build_tasks()
    assert len(tasks) == 1 and tasks[0].name == "health"


def test_build_tasks_skips_disabled():
    cfg = _config(
        tasks=[
            HeartbeatTaskSettings(
                name="health",
                type="universal_agentic_framework.heartbeat.tasks.health:HealthHeartbeatTask",
                enabled=False,
            )
        ]
    )
    sched = HeartbeatScheduler(cfg, build_default_stores=False)
    assert sched._build_tasks() == []


def test_coerce_rate():
    assert _coerce_rate(10, 5) == 10
    assert _coerce_rate(0, 5) == 1       # clamped to min
    assert _coerce_rate(99999, 5) == 1440  # clamped to max
    assert _coerce_rate(None, 5) == 5    # fallback
    assert _coerce_rate("x", 5) == 5


# --------------------------------------------------------------------------- #
# Scheduler control loop (reschedule only on change)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_control_reschedules_only_on_change():
    settings = _FakeSettingsStore(value=None)  # no override → default 5
    sched = HeartbeatScheduler(
        _config(rate=5),
        settings_store=settings,
        run_store=_FakeRunStore(),
        build_default_stores=False,
    )
    sched.start()
    try:
        assert len(sched.get_jobs()) == 2
        beat = sched.scheduler.get_job(BEAT_JOB_ID)
        assert _trigger_minutes(beat.trigger) == 5
        control = sched.scheduler.get_job(CONTROL_JOB_ID)
        assert control is not None

        # No change → trigger untouched (and next_run not reset).
        next_before = sched.scheduler.get_job(BEAT_JOB_ID).next_run_time
        await sched._control()
        assert _trigger_minutes(sched.scheduler.get_job(BEAT_JOB_ID).trigger) == 5
        assert sched.scheduler.get_job(BEAT_JOB_ID).next_run_time == next_before

        # Admin changes the rate → beat reschedules to the new interval.
        settings.value = 12
        await sched._control()
        assert _trigger_minutes(sched.scheduler.get_job(BEAT_JOB_ID).trigger) == 12
    finally:
        sched.stop()
    assert sched.running is False


# --------------------------------------------------------------------------- #
# Per-user fan-out + queue
# --------------------------------------------------------------------------- #
def _fanout_config() -> HeartbeatSettings:
    return _config(
        tasks=[
            HeartbeatTaskSettings(name="health", type=_HEALTH, scope="global"),
            HeartbeatTaskSettings(name="pulse", type=_HEALTH, scope="per_user"),
        ]
    )


@pytest.mark.asyncio
async def test_beat_fans_out_per_user_and_drains():
    run_store = _FakeRunStore()
    sched = HeartbeatScheduler(
        _fanout_config(),
        run_store=run_store,
        settings_store=_FakeSettingsStore(None),
        user_store=_FakeUserStore(["u1", "u2", "u3"]),
        build_default_stores=False,
    )
    sched.start()
    try:
        await sched._beat()
        await sched._queue.join()  # wait for the worker pool to drain
    finally:
        sched.stop()

    # 1 global + 3 per-user = 4 recorded runs.
    assert len(run_store.records) == 4
    per_user = sorted(r["user_id"] for r in run_store.records if r["user_id"] is not None)
    assert per_user == ["u1", "u2", "u3"]
    globals_ = [r for r in run_store.records if r["user_id"] is None]
    assert len(globals_) == 1 and globals_[0]["task_name"] == "health"


@pytest.mark.asyncio
async def test_beat_skips_when_backlogged():
    sched = HeartbeatScheduler(
        _fanout_config(),
        run_store=_FakeRunStore(),
        settings_store=_FakeSettingsStore(None),
        user_store=_FakeUserStore(["u1"]),
        build_default_stores=False,
    )
    sched._tasks = sched._build_tasks()
    # Saturate the queue past the high-water mark (no workers running to drain it).
    for _ in range(QUEUE_HIGH_WATER):
        sched._queue.put_nowait((sched._tasks[0], None))
    await sched._beat()
    assert sched._queue.qsize() == QUEUE_HIGH_WATER  # nothing new enqueued


@pytest.mark.asyncio
async def test_per_user_task_without_user_store_enqueues_nothing():
    sched = HeartbeatScheduler(
        _config(tasks=[HeartbeatTaskSettings(name="pulse", type=_HEALTH, scope="per_user")]),
        run_store=_FakeRunStore(),
        settings_store=_FakeSettingsStore(None),
        user_store=None,
        build_default_stores=False,
    )
    sched._tasks = sched._build_tasks()
    await sched._beat()
    assert sched._queue.qsize() == 0


@pytest.mark.asyncio
async def test_control_prunes_periodically():
    run_store = _FakeRunStore()
    sched = HeartbeatScheduler(
        _config(rate=5),
        run_store=run_store,
        settings_store=_FakeSettingsStore(None),
        build_default_stores=False,
    )
    sched.start()
    try:
        # Two ticks short of the cadence: first control tick doesn't prune, the
        # next (cadence) tick prunes exactly once.
        sched._control_ticks = PRUNE_EVERY_CONTROL_TICKS - 2
        await sched._control()
        assert run_store.pruned_cutoffs == []
        await sched._control()
        assert len(run_store.pruned_cutoffs) == 1
    finally:
        sched.stop()


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #
def test_heartbeat_settings_parse():
    cfg = HeartbeatSettings(
        **{"enabled": True, "default_rate_minutes": 10, "tasks": [{"name": "health", "type": "a.b:C"}]}
    )
    assert cfg.enabled and cfg.default_rate_minutes == 10
    assert cfg.tasks[0].name == "health" and cfg.tasks[0].cooldown_seconds == 0


def test_heartbeat_settings_defaults_disabled():
    cfg = HeartbeatSettings()
    assert cfg.enabled is False and cfg.default_rate_minutes == 5 and cfg.tasks == []


# --------------------------------------------------------------------------- #
# Admin endpoint
# --------------------------------------------------------------------------- #
class _FakeGlobalSettingsStore:
    def __init__(self) -> None:
        self._d: Dict[str, Any] = {}

    def get_setting(self, key: str):
        return self._d.get(key)

    def set_setting(self, key: str, value: Any):
        self._d[key] = value
        return value


def _make_client(monkeypatch, *, role: str = "administrator", store=None) -> TestClient:
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)  # dev bypass → role from env
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_AUTH_USER_ROLE", role)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    from backend.routers.settings import router

    app = FastAPI()
    app.include_router(router)
    app.state.global_settings_store = store if store is not None else _FakeGlobalSettingsStore()
    app.state.db_pool = None  # _heartbeat_last_run returns None without a pool
    return TestClient(app)


def test_get_heartbeat_rate_defaults_to_profile(monkeypatch):
    body = _make_client(monkeypatch).get("/api/admin/settings/heartbeat-rate").json()
    assert body["heartbeat_rate_minutes"] == 5  # starter default_rate_minutes
    assert body["source"] == "default"
    assert body["enabled"] is True
    assert body["last_run"] is None


def test_put_then_get_reflects_override(monkeypatch):
    client = _make_client(monkeypatch)
    put = client.put("/api/admin/settings/heartbeat-rate", json={"heartbeat_rate_minutes": 30})
    assert put.status_code == 200
    assert put.json()["heartbeat_rate_minutes"] == 30
    assert put.json()["source"] == "override"

    got = client.get("/api/admin/settings/heartbeat-rate").json()
    assert got["heartbeat_rate_minutes"] == 30
    assert got["source"] == "override"


def test_put_rejects_out_of_bounds(monkeypatch):
    client = _make_client(monkeypatch)
    assert client.put("/api/admin/settings/heartbeat-rate", json={"heartbeat_rate_minutes": 0}).status_code == 422
    assert client.put("/api/admin/settings/heartbeat-rate", json={"heartbeat_rate_minutes": 1441}).status_code == 422


def test_heartbeat_rate_endpoints_require_admin(monkeypatch):
    client = _make_client(monkeypatch, role="user")
    assert client.get("/api/admin/settings/heartbeat-rate").status_code == 403
    assert client.put("/api/admin/settings/heartbeat-rate", json={"heartbeat_rate_minutes": 5}).status_code == 403


# --------------------------------------------------------------------------- #
# Inspector endpoints
# --------------------------------------------------------------------------- #
def test_list_heartbeat_tasks_returns_configured(monkeypatch):
    body = _make_client(monkeypatch).get("/api/admin/heartbeat/tasks").json()
    names = {t["name"]: t for t in body["tasks"]}
    # starter profile configures a global health task + a per-user demo task.
    assert names["health"]["scope"] == "global"
    assert names["user_pulse"]["scope"] == "per_user"
    assert names["health"]["last_run"] is None  # no db_pool in the test client


def test_list_heartbeat_runs_empty_without_pool(monkeypatch):
    resp = _make_client(monkeypatch).get("/api/admin/heartbeat/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_heartbeat_runs_limit_validated(monkeypatch):
    client = _make_client(monkeypatch)
    assert client.get("/api/admin/heartbeat/runs?limit=0").status_code == 422
    assert client.get("/api/admin/heartbeat/runs?limit=999").status_code == 422


def test_heartbeat_inspector_endpoints_require_admin(monkeypatch):
    client = _make_client(monkeypatch, role="user")
    assert client.get("/api/admin/heartbeat/tasks").status_code == 403
    assert client.get("/api/admin/heartbeat/runs").status_code == 403
