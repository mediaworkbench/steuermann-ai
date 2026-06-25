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


def test_reset_all_databases_admin_passes_gate(monkeypatch):
    client = _client(monkeypatch, role="administrator")
    resp = client.post("/api/admin/reset-all-databases", headers=_headers("administrator"))
    # Admin clears the require_admin gate (it would 401/403 otherwise). The body may error
    # because no DB/Qdrant is wired in this unit app — we only assert authorization passed.
    assert resp.status_code not in (401, 403)
