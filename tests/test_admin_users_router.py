"""Tests for the admin-only user management API (/api/admin/users, /api/admin/roles)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.admin_users import router as admin_users_router
from backend.single_user import require_api_access


def _public(u: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if u is None:
        return None
    return {k: v for k, v in u.items() if k != "password_hash"}


class FakeUserStore:
    def __init__(self) -> None:
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles = [
            {"role_id": 1, "role_name": "user", "description": "", "created_at": None},
            {"role_id": 2, "role_name": "researcher", "description": "", "created_at": None},
            {"role_id": 3, "role_name": "administrator", "description": "", "created_at": None},
        ]

    def add(self, user_id, username, role_name="user", status="active"):
        self.users[user_id] = {
            "user_id": user_id, "username": username, "email": f"{username}@x.com",
            "password_hash": "h", "role_name": role_name, "status": status,
            "must_change_password": False,
        }

    def get_all_roles(self):
        return self.roles

    def get_all_users(self, limit=100, offset=0):
        rows = list(self.users.values())
        return [_public(u) for u in rows[offset:offset + limit]], len(rows)

    def get_user_by_username_with_hash(self, username):
        for u in self.users.values():
            if u["username"] == username:
                return dict(u)
        return None

    def get_user_by_id(self, user_id):
        return _public(self.users.get(user_id))

    def create_user_with_password(self, user_id, username, email, password_hash, role_name, must_change_password=False):
        if role_name not in ("user", "researcher", "administrator"):
            raise ValueError("bad role")
        self.users[user_id] = {
            "user_id": user_id, "username": username, "email": email,
            "password_hash": password_hash, "role_name": role_name,
            "status": "active", "must_change_password": must_change_password,
        }
        return _public(self.users[user_id])

    def update_user_role(self, user_id, role_name):
        self.users[user_id]["role_name"] = role_name
        return _public(self.users[user_id])

    def update_user_status(self, user_id, status):
        self.users[user_id]["status"] = status
        return _public(self.users[user_id])

    def set_password_hash(self, user_id, password_hash, must_change_password=False):
        self.users[user_id]["password_hash"] = password_hash
        self.users[user_id]["must_change_password"] = must_change_password
        return True

    def bump_token_version(self, user_id):
        rec = self.users.get(user_id)
        if rec is None:
            return None
        rec["token_version"] = int(rec.get("token_version") or 0) + 1
        return rec["token_version"]

    def delete_user(self, user_id):
        return self.users.pop(user_id, None) is not None

    def count_admins(self, active_only=True):
        return sum(
            1 for u in self.users.values()
            if u["role_name"] == "administrator" and (not active_only or u["status"] == "active")
        )


def _client(monkeypatch, store, *, auth_enabled=False, admin_user_id="boss"):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    if auth_enabled:
        monkeypatch.setenv("AUTH_ENABLED", "true")
    else:
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        monkeypatch.setenv("AUTH_USERNAME", admin_user_id)
    app = FastAPI()
    app.state.user_store = store
    app.include_router(admin_users_router)
    # Perimeter is covered in test_security; neutralize it here (it now fails closed
    # without a CHAT_ACCESS_TOKEN, which these auth-enabled fixtures intentionally omit).
    app.dependency_overrides[require_api_access] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


class TestAdminGate:
    def test_non_admin_forbidden(self, monkeypatch):
        store = FakeUserStore()
        store.add("u-1", "alice", role_name="user")  # acting user re-validated against the DB
        client = _client(monkeypatch, store, auth_enabled=True)
        resp = client.get(
            "/api/admin/users",
            headers={
                "x-authenticated-user-id": "u-1",
                "x-authenticated-username": "alice",
                "x-authenticated-role": "user",
            },
        )
        assert resp.status_code == 403

    def test_admin_allowed_via_headers(self, monkeypatch):
        store = FakeUserStore()
        store.add("u-1", "boss", role_name="administrator")
        client = _client(monkeypatch, store, auth_enabled=True)
        resp = client.get(
            "/api/admin/users",
            headers={
                "x-authenticated-user-id": "u-1",
                "x-authenticated-username": "boss",
                "x-authenticated-role": "administrator",
            },
        )
        assert resp.status_code == 200


class TestSessionRevalidation:
    """resolve_current_user re-validates the JWT-derived identity against the DB."""

    def _hdr(self, role="administrator", uid="u-1", username="boss"):
        return {
            "x-authenticated-user-id": uid,
            "x-authenticated-username": username,
            "x-authenticated-role": role,
        }

    def test_deleted_user_rejected_401(self, monkeypatch):
        store = FakeUserStore()  # u-1 absent (deleted)
        client = _client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/admin/users", headers=self._hdr()).status_code == 401

    def test_suspended_user_rejected_403(self, monkeypatch):
        store = FakeUserStore()
        store.add("u-1", "boss", role_name="administrator", status="suspended")
        client = _client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/admin/users", headers=self._hdr()).status_code == 403

    def test_db_role_overrides_header_role(self, monkeypatch):
        # Header still claims administrator, but the DB demoted them to user.
        store = FakeUserStore()
        store.add("u-1", "boss", role_name="user")
        client = _client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/admin/users", headers=self._hdr(role="administrator")).status_code == 403

    def test_must_change_password_blocks_non_auth_routes(self, monkeypatch):
        store = FakeUserStore()
        store.add("u-1", "boss", role_name="administrator")
        store.users["u-1"]["must_change_password"] = True
        client = _client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/admin/users", headers=self._hdr()).status_code == 403


class TestRolesAndList:
    def test_list_roles(self, monkeypatch):
        store = FakeUserStore()
        client = _client(monkeypatch, store)
        resp = client.get("/api/admin/roles")
        assert resp.status_code == 200
        names = {r["role_name"] for r in resp.json()["roles"]}
        assert names == {"user", "researcher", "administrator"}

    def test_list_users(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice")
        store.add("b", "bob", role_name="researcher")
        client = _client(monkeypatch, store)
        resp = client.get("/api/admin/users")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert {u["username"] for u in body["users"]} == {"alice", "bob"}
        assert all("password_hash" not in u for u in body["users"])


class TestCreateUser:
    def test_create_success(self, monkeypatch):
        store = FakeUserStore()
        client = _client(monkeypatch, store)
        resp = client.post("/api/admin/users", json={"username": "newbie", "email": "n@x.com", "role": "researcher"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["user"]["role_name"] == "researcher"
        assert body["user"]["must_change_password"] is True
        assert len(body["temporary_password"]) >= 8

    def test_create_invalid_role_400(self, monkeypatch):
        store = FakeUserStore()
        client = _client(monkeypatch, store)
        resp = client.post("/api/admin/users", json={"username": "x", "email": "x@x.com", "role": "superuser"})
        assert resp.status_code == 400

    def test_create_bad_email_400(self, monkeypatch):
        store = FakeUserStore()
        client = _client(monkeypatch, store)
        resp = client.post("/api/admin/users", json={"username": "x", "email": "not-an-email", "role": "user"})
        assert resp.status_code == 400

    def test_create_duplicate_username_409(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice")
        client = _client(monkeypatch, store)
        resp = client.post("/api/admin/users", json={"username": "alice", "email": "a2@x.com", "role": "user"})
        assert resp.status_code == 409


class TestUpdateUser:
    def test_update_role(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice", role_name="user")
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        resp = client.patch("/api/admin/users/a", json={"role": "researcher"})
        assert resp.status_code == 200
        assert resp.json()["user"]["role_name"] == "researcher"

    def test_reset_password_returns_temp(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice")
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        resp = client.patch("/api/admin/users/a", json={"reset_password": True})
        assert resp.status_code == 200
        assert len(resp.json()["temporary_password"]) >= 8
        assert store.users["a"]["must_change_password"] is True

    def test_role_change_bumps_target_token_version(self, monkeypatch):
        # Demoting/promoting a user must invalidate their existing sessions immediately.
        store = FakeUserStore()
        store.add("a", "alice", role_name="user")
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        client.patch("/api/admin/users/a", json={"role": "researcher"})
        assert store.users["a"]["token_version"] == 1

    def test_password_reset_bumps_target_token_version(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice")
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        client.patch("/api/admin/users/a", json={"reset_password": True})
        assert store.users["a"]["token_version"] == 1

    def test_unknown_user_404(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        resp = client.patch("/api/admin/users/ghost", json={"status": "suspended"})
        assert resp.status_code == 404

    def test_cannot_demote_last_admin(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store, admin_user_id="other")  # acting as a different (dev) admin
        store.add("other", "other", role_name="administrator")
        # remove 'other' so 'boss' is the only active admin
        del store.users["other"]
        store.add("other", "other", role_name="administrator", status="suspended")
        resp = client.patch("/api/admin/users/boss", json={"role": "user"})
        assert resp.status_code == 400

    def test_cannot_deactivate_self(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        store.add("second", "second", role_name="administrator")  # not the last admin
        client = _client(monkeypatch, store, admin_user_id="boss")
        resp = client.patch("/api/admin/users/boss", json={"status": "suspended"})
        assert resp.status_code == 400

    def test_cannot_demote_self(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        store.add("second", "second", role_name="administrator")
        client = _client(monkeypatch, store, admin_user_id="boss")
        resp = client.patch("/api/admin/users/boss", json={"role": "user"})
        assert resp.status_code == 400


class TestDeleteUser:
    def test_delete_success(self, monkeypatch):
        store = FakeUserStore()
        store.add("a", "alice")
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store, admin_user_id="boss")
        resp = client.delete("/api/admin/users/a")
        assert resp.status_code == 204
        assert "a" not in store.users

    def test_delete_self_400(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        store.add("second", "second", role_name="administrator")
        client = _client(monkeypatch, store, admin_user_id="boss")
        resp = client.delete("/api/admin/users/boss")
        assert resp.status_code == 400

    def test_delete_last_admin_400(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")  # only admin
        store.add("a", "alice")
        client = _client(monkeypatch, store, admin_user_id="a-dev-admin")
        resp = client.delete("/api/admin/users/boss")
        assert resp.status_code == 400

    def test_delete_unknown_404(self, monkeypatch):
        store = FakeUserStore()
        store.add("boss", "boss", role_name="administrator")
        client = _client(monkeypatch, store)
        resp = client.delete("/api/admin/users/ghost")
        assert resp.status_code == 404
