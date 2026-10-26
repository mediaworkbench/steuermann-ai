"""Regression test for the admin gate on POST /api/admin/reset-all-databases.

The endpoint previously inherited only ``require_api_access`` from the settings router, so
any authenticated (non-admin) user could wipe every user's data through the proxy. It must
now require an administrator. We assert the gate, not the destructive body — a non-admin is
rejected before the body runs, and an admin passes the gate (then fails later only because
the test app has no DB/Qdrant state wired, which is fine: the point is it is NOT 401/403).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.settings import router as settings_router
from backend.single_user import require_api_access


class FakeUserStore:
    def __init__(self, users: Dict[str, Dict[str, Any]]) -> None:
        self.users = users

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.users.get(user_id)


def _client(monkeypatch, *, role: str) -> TestClient:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    store = FakeUserStore(
        {"u-1": {"user_id": "u-1", "username": "u", "role_name": role, "status": "active", "token_version": 0}}
    )
    app = FastAPI()
    app.state.user_store = store
    app.include_router(settings_router)
    app.dependency_overrides[require_api_access] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def _headers(role: str) -> Dict[str, str]:
    return {
        "x-authenticated-user-id": "u-1",
        "x-authenticated-username": "u",
        "x-authenticated-role": role,
    }


def test_reset_all_databases_blocks_non_admin(monkeypatch):
    client = _client(monkeypatch, role="user")
    resp = client.post("/api/admin/reset-all-databases", headers=_headers("user"))
    assert resp.status_code == 403


def test_reset_all_databases_blocks_researcher(monkeypatch):
    client = _client(monkeypatch, role="researcher")
    resp = client.post("/api/admin/reset-all-databases", headers=_headers("researcher"))
    assert resp.status_code == 403


def test_reset_all_databases_admin_passes_gate(monkeypatch, tmp_path):
    # The endpoint body really runs here, and it wipes every Qdrant collection and both file
    # roots. Point all of them somewhere harmless: on a dev machine QDRANT_HOST=localhost
    # (conftest) reaches the live Qdrant through the published port and deleted it.
    monkeypatch.setenv("QDRANT_HOST", "127.0.0.1")
    monkeypatch.setenv("QDRANT_PORT", "1")  # closed port: connection refused immediately
    monkeypatch.setenv("CHAT_ATTACHMENTS_ROOT", str(tmp_path / "att"))
    monkeypatch.setenv("CHAT_WORKSPACE_ROOT", str(tmp_path / "ws"))
    client = _client(monkeypatch, role="administrator")
    resp = client.post("/api/admin/reset-all-databases", headers=_headers("administrator"))
    # Admin clears the require_admin gate (it would 401/403 otherwise). The body reports
    # errors because no DB/Qdrant is reachable — we only assert authorization passed.
    assert resp.status_code not in (401, 403)


class _RecordingCursor:
    def __init__(self, log, missing=()):
        self.log = log
        self.missing = set(missing)
        self.rowcount = 0
        self._row = None

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        self.rowcount = 1
        if sql.startswith("SELECT to_regclass"):
            self._row = (None if params[0] in self.missing else params[0],)
        elif sql.startswith("SELECT id FROM conversations"):
            self._rows = [("conv-1",), ("conv-2",)]

    def fetchone(self):
        return self._row

    def fetchall(self):
        return getattr(self, "_rows", [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingPool:
    def __init__(self, missing=()):
        self.log = []
        self.missing = missing

    from contextlib import contextmanager

    @contextmanager
    def connection(self):
        pool = self

        class _Conn:
            def cursor(self_inner):
                return _RecordingCursor(pool.log, pool.missing)

            def commit(self_inner):
                pass

        yield _Conn()


def test_delete_checkpoint_threads_skips_missing_tables():
    from backend.db import delete_checkpoint_threads

    log = []
    cur = _RecordingCursor(log, missing={"checkpoint_writes"})
    delete_checkpoint_threads(cur, ["c1", "c2"])
    deletes = [sql for sql, _ in log if sql.startswith("DELETE")]
    assert deletes == [
        "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s);",
        "DELETE FROM checkpoints WHERE thread_id = ANY(%s);",
    ]
    assert delete_checkpoint_threads(cur, []) == 0


def test_reset_my_data_purges_dreaming_state_and_checkpoints(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("AUTH_USERNAME", "u-1")
    monkeypatch.setenv("CHAT_ATTACHMENTS_ROOT", str(tmp_path / "att"))
    monkeypatch.setenv("CHAT_WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("QDRANT_HOST", "127.0.0.1")  # never the live Qdrant (see above)
    monkeypatch.setenv("QDRANT_PORT", "1")
    purged_memories = []

    class _FakeMemoryBackend:
        def _delete_all_memories(self, *, user_id):
            purged_memories.append(user_id)

    # The memory backend's Qdrant host comes from the profile config, not QDRANT_HOST.
    monkeypatch.setattr(
        "universal_agentic_framework.memory.factory.build_memory_backend",
        lambda cfg: _FakeMemoryBackend(),
    )
    invalidated = []
    monkeypatch.setattr(
        "backend.routers.settings.invalidate_procedural_cache", lambda uid: invalidated.append(uid)
    )
    pool = _RecordingPool()
    app = FastAPI()
    app.state.db_pool = pool
    app.include_router(settings_router)
    client = TestClient(app, raise_server_exceptions=False)

    client.post("/api/user/reset-my-data")

    statements = [(sql, params) for sql, params in pool.log if sql.startswith("DELETE")]
    tables = {sql.split()[2] for sql, _ in statements}
    for table in (
        "memory_audit_log", "memory_conflicts", "procedural_overrides", "heartbeat_runs",
        "co_occurrence_edges", "checkpoints", "checkpoint_blobs", "checkpoint_writes",
    ):
        assert table in tables, table
    checkpoint_params = [p for sql, p in statements if "FROM checkpoints " in sql]
    assert checkpoint_params == [(["conv-1", "conv-2"],)]
    assert invalidated == ["u-1"]
    assert purged_memories == ["u-1"]
