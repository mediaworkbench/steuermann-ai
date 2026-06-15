"""Tests for the DB-backed auth router (login + forced password change).

These exercise the router logic against a fake UserStore (no live database) using a
real argon2id hash so verification is end-to-end.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import hash_password, verify_password
from backend.routers.auth import router as auth_router


class FakeUserStore:
    def __init__(
        self,
        by_username: Optional[Dict[str, Dict[str, Any]]] = None,
        by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.by_username = by_username or {}
        self.by_id = by_id or {}
        self.set_calls: list[tuple[str, str, bool]] = []

    def get_user_by_username_with_hash(self, username: str):
        return self.by_username.get(username)

    def get_user_by_id_with_hash(self, user_id: str):
        return self.by_id.get(user_id)

    def get_user_by_id(self, user_id: str):
        rec = self.by_id.get(user_id)
        return {k: v for k, v in rec.items() if k != "password_hash"} if rec else None

    def set_password_hash(self, user_id: str, password_hash: str, must_change_password: bool = False):
        self.set_calls.append((user_id, password_hash, must_change_password))
        rec = self.by_id.get(user_id)
        if rec is not None:
            rec["password_hash"] = password_hash
            rec["must_change_password"] = must_change_password
        return rec is not None


def _make_client(monkeypatch, store, *, auth_enabled=False, username="patrick") -> TestClient:
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    if auth_enabled:
        monkeypatch.setenv("AUTH_ENABLED", "true")
    else:
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        monkeypatch.setenv("AUTH_USERNAME", username)
    app = FastAPI()
    app.state.user_store = store
    app.include_router(auth_router)
    return TestClient(app, raise_server_exceptions=False)


def _user_record(**overrides) -> Dict[str, Any]:
    rec = {
        "user_id": "patrick",
        "username": "patrick",
        "email": "patrick@example.com",
        "password_hash": hash_password("correct-horse"),
        "role_id": 3,
        "role_name": "administrator",
        "status": "active",
        "must_change_password": False,
    }
    rec.update(overrides)
    return rec


class TestLogin:
    def test_login_success(self, monkeypatch):
        store = FakeUserStore(by_username={"patrick": _user_record()})
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "patrick", "password": "correct-horse"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "patrick"
        assert body["role"] == "administrator"
        assert body["must_change_password"] is False

    def test_login_reports_must_change_password(self, monkeypatch):
        store = FakeUserStore(
            by_username={"newbie": _user_record(user_id="newbie", username="newbie", role_name="user", must_change_password=True)}
        )
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "newbie", "password": "correct-horse"})
        assert resp.status_code == 200
        assert resp.json()["must_change_password"] is True

    def test_login_wrong_password_401(self, monkeypatch):
        store = FakeUserStore(by_username={"patrick": _user_record()})
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "patrick", "password": "nope"})
        assert resp.status_code == 401

    def test_login_unknown_user_401(self, monkeypatch):
        store = FakeUserStore()
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "ghost", "password": "whatever"})
        assert resp.status_code == 401

    def test_login_suspended_403(self, monkeypatch):
        store = FakeUserStore(by_username={"patrick": _user_record(status="suspended")})
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "patrick", "password": "correct-horse"})
        assert resp.status_code == 403

    def test_login_no_role_403(self, monkeypatch):
        store = FakeUserStore(by_username={"patrick": _user_record(role_name=None)})
        client = _make_client(monkeypatch, store)
        resp = client.post("/api/auth/login", json={"username": "patrick", "password": "correct-horse"})
        assert resp.status_code == 403


class TestMe:
    """GET /api/auth/me returns the DB-fresh identity (re-validated, role-authoritative)."""

    def _hdr(self, uid="u-1", role="administrator", username="alice"):
        return {
            "x-authenticated-user-id": uid,
            "x-authenticated-username": username,
            "x-authenticated-role": role,
        }

    def test_me_returns_db_fresh_role(self, monkeypatch):
        rec = _user_record(user_id="u-1", username="alice", email="a@x.com", role_name="user")
        store = FakeUserStore(by_id={"u-1": rec})
        client = _make_client(monkeypatch, store, auth_enabled=True)
        # Header still claims administrator, but the DB demoted them to user.
        resp = client.get("/api/auth/me", headers=self._hdr(role="administrator"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "user"
        assert body["email"] == "a@x.com"
        assert body["user_id"] == "u-1"

    def test_me_rejects_suspended_403(self, monkeypatch):
        rec = _user_record(user_id="u-1", role_name="user", status="suspended")
        store = FakeUserStore(by_id={"u-1": rec})
        client = _make_client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/auth/me", headers=self._hdr()).status_code == 403

    def test_me_rejects_deleted_401(self, monkeypatch):
        store = FakeUserStore()  # u-1 absent
        client = _make_client(monkeypatch, store, auth_enabled=True)
        assert client.get("/api/auth/me", headers=self._hdr()).status_code == 401


class TestChangePassword:
    def test_change_password_success_dev_bypass(self, monkeypatch):
        rec = _user_record(password_hash=hash_password("old-password"))
        store = FakeUserStore(by_id={"patrick": rec})
        client = _make_client(monkeypatch, store, username="patrick")
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "old-password", "new_password": "brand-new-pass"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert store.set_calls and store.set_calls[0][0] == "patrick"
        assert store.set_calls[0][2] is False  # must_change_password cleared
        assert verify_password("brand-new-pass", rec["password_hash"])

    def test_change_password_via_trusted_headers(self, monkeypatch):
        rec = _user_record(user_id="u-1", username="alice", role_name="user", password_hash=hash_password("temp-password"))
        store = FakeUserStore(by_id={"u-1": rec})
        client = _make_client(monkeypatch, store, auth_enabled=True)
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "temp-password", "new_password": "chosen-password"},
            headers={
                "x-authenticated-user-id": "u-1",
                "x-authenticated-username": "alice",
                "x-authenticated-role": "user",
            },
        )
        assert resp.status_code == 200
        assert store.set_calls[0][0] == "u-1"

    def test_change_password_reachable_despite_must_change_flag(self, monkeypatch):
        # A flagged user is blocked everywhere EXCEPT /api/auth/* — otherwise they could
        # never clear the flag (lockout).
        rec = _user_record(
            user_id="u-1",
            username="alice",
            role_name="user",
            password_hash=hash_password("temp-password"),
            must_change_password=True,
        )
        store = FakeUserStore(by_id={"u-1": rec})
        client = _make_client(monkeypatch, store, auth_enabled=True)
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "temp-password", "new_password": "chosen-password"},
            headers={
                "x-authenticated-user-id": "u-1",
                "x-authenticated-username": "alice",
                "x-authenticated-role": "user",
            },
        )
        assert resp.status_code == 200

    def test_change_password_wrong_current_401(self, monkeypatch):
        rec = _user_record(password_hash=hash_password("old-password"))
        store = FakeUserStore(by_id={"patrick": rec})
        client = _make_client(monkeypatch, store, username="patrick")
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "wrong", "new_password": "brand-new-pass"},
        )
        assert resp.status_code == 401

    def test_change_password_too_short_400(self, monkeypatch):
        rec = _user_record(password_hash=hash_password("old-password"))
        store = FakeUserStore(by_id={"patrick": rec})
        client = _make_client(monkeypatch, store, username="patrick")
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "old-password", "new_password": "short"},
        )
        assert resp.status_code == 400

    def test_change_password_same_as_current_400(self, monkeypatch):
        rec = _user_record(password_hash=hash_password("old-password"))
        store = FakeUserStore(by_id={"patrick": rec})
        client = _make_client(monkeypatch, store, username="patrick")
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "old-password", "new_password": "old-password"},
        )
        assert resp.status_code == 400
