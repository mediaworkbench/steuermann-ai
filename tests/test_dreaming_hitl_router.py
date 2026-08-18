"""Phase 6 tests: the Dreaming Engine HITL router.

Covers the plan's acceptance criteria with fakes (no DB/Qdrant):
- user A can't see or resolve user B's data (user-scoped → 404/empty);
- approve/reject only act on *proposed* rules;
- undo respects ``reversible_until`` (past → 410, null → 409, future → reverses);
- admin metrics are aggregates only (no text / ids / user names).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.dreaming as dreaming
from backend.routers.dreaming import router
from backend.auth import current_user_id, require_admin
from backend.single_user import require_api_access

NOW = datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Fakes (enforce user-scoping like the real SQL WHERE user_id)
# --------------------------------------------------------------------------- #
class _FakeConflict:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = rows or []

    def list_open(self, user_id):
        return [r for r in self.rows if r["user_id"] == user_id and r.get("status", "open") == "open"]

    def resolve(self, *, conflict_id, user_id, resolution, status="resolved"):
        for r in self.rows:
            if r["conflict_id"] == conflict_id and r["user_id"] == user_id:
                r["status"] = status
                r["resolution"] = resolution
                return dict(r)
        return None

    def aggregate_metrics(self):
        opens = sum(1 for r in self.rows if r.get("status", "open") == "open")
        return {"total": len(self.rows), "by_status": {"open": opens}, "open": opens}


class _FakeProc:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = {(r["user_id"], r["rule_key"]): r for r in (rows or [])}

    def get(self, user_id, rule_key):
        r = self.rows.get((user_id, rule_key))
        return dict(r) if r else None

    def set_status(self, *, user_id, rule_key, status):
        r = self.rows.get((user_id, rule_key))
        if not r:
            return None
        r["status"] = status
        return dict(r)

    def list_for_user(self, user_id):
        return [dict(r) for (u, _k), r in self.rows.items() if u == user_id]

    def aggregate_metrics(self):
        proposed = sum(1 for r in self.rows.values() if r["status"] == "proposed")
        return {"total": len(self.rows), "by_status": {}, "by_tier": {}, "proposed_pending": proposed}


class _FakeAudit:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = {r["audit_id"]: r for r in (rows or [])}
        self.recorded: List[Dict[str, Any]] = []

    def list_reversible(self, *, user_id, now=None):
        now = now or NOW
        return [
            dict(r) for r in self.rows.values()
            if r["user_id"] == user_id and r.get("reversible_until") and r["reversible_until"] > now
        ]

    def get(self, *, audit_id, user_id):
        r = self.rows.get(audit_id)
        return dict(r) if r and r["user_id"] == user_id else None

    def mark_undone(self, *, audit_id, user_id):
        r = self.rows.get(audit_id)
        if r and r["user_id"] == user_id and r.get("reversible_until"):
            r["reversible_until"] = None
            return True
        return False

    def record(self, **kw):
        self.recorded.append(kw)
        return "audit-x"

    def aggregate_metrics(self):
        return {"total": len(self.rows), "by_cycle": {}, "by_action": {"delete": 3, "promote": 1}}


class _FakeRun:
    def aggregate_task_stats(self, task_name):
        return {"count": 10, "by_status": {"ok": 9, "partial": 1}, "total_tokens": 500}


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def update_metadata(self, mid, patch):
        self.calls.append(("update", mid, patch))
        return True

    def delete_memory(self, *, memory_id, user_id):
        self.calls.append(("delete", memory_id, user_id))

    def restore_memory(self, user_id, *, text, metadata):
        self.calls.append(("restore", user_id, text))
        return None

    def count_points(self):
        return 1234


def _client(monkeypatch, *, user_id="userA", conflict=None, proc=None, audit=None,
            run=None, backend=None, admin=False):
    app = FastAPI()
    app.include_router(router)
    app.state.memory_conflict_store = conflict or _FakeConflict()
    app.state.procedural_store = proc or _FakeProc()
    app.state.memory_audit_store = audit or _FakeAudit()
    app.state.heartbeat_run_store = run or _FakeRun()
    app.dependency_overrides[require_api_access] = lambda: None
    app.dependency_overrides[current_user_id] = lambda: user_id
    if admin:
        app.dependency_overrides[require_admin] = lambda: SimpleNamespace(
            user_id=user_id, username=user_id, role="administrator"
        )
    fake_backend = backend or _FakeBackend()
    monkeypatch.setattr(dreaming, "_get_backend", lambda: fake_backend)
    monkeypatch.setattr(dreaming, "invalidate_procedural_cache", lambda _u: None)
    client = TestClient(app)
    client._fake_backend = fake_backend  # type: ignore[attr-defined]
    return client


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #
def test_conflicts_are_user_scoped(monkeypatch):
    conflict = _FakeConflict([
        {"conflict_id": "c-a", "user_id": "userA", "semantic_text": "A's", "choices": []},
        {"conflict_id": "c-b", "user_id": "userB", "semantic_text": "B's", "choices": []},
    ])
    client = _client(monkeypatch, user_id="userA", conflict=conflict)
    body = client.get("/api/memory/dreaming/conflicts").json()
    ids = [c["conflict_id"] for c in body["conflicts"]]
    assert ids == ["c-a"]  # never sees user B's conflict


def test_resolve_other_users_conflict_is_404(monkeypatch):
    conflict = _FakeConflict([{"conflict_id": "c-b", "user_id": "userB", "choices": []}])
    client = _client(monkeypatch, user_id="userA", conflict=conflict)
    resp = client.post("/api/memory/dreaming/conflicts/c-b/resolve", json={"choice": "keep_old"})
    assert resp.status_code == 404


def test_resolve_keep_old_restores_confidence(monkeypatch):
    conflict = _FakeConflict([
        {"conflict_id": "c1", "user_id": "userA", "semantic_memory_id": "sem1",
         "prior_confidence": 0.7, "choices": []},
    ])
    client = _client(monkeypatch, user_id="userA", conflict=conflict)
    resp = client.post("/api/memory/dreaming/conflicts/c1/resolve", json={"choice": "keep_old"})
    assert resp.status_code == 200 and resp.json()["choice"] == "keep_old"
    assert ("update", "sem1", {"confidence": 0.7}) in client._fake_backend.calls


def test_resolve_accept_new_deletes_semantic(monkeypatch):
    conflict = _FakeConflict([
        {"conflict_id": "c1", "user_id": "userA", "semantic_memory_id": "sem1", "choices": []},
    ])
    client = _client(monkeypatch, user_id="userA", conflict=conflict)
    client.post("/api/memory/dreaming/conflicts/c1/resolve", json={"choice": "accept_new"})
    assert ("delete", "sem1", "userA") in client._fake_backend.calls


# --------------------------------------------------------------------------- #
# Procedural approvals
# --------------------------------------------------------------------------- #
def test_approve_proposed_rule_activates(monkeypatch):
    proc = _FakeProc([{"user_id": "userA", "rule_key": "format.bullets", "rule_text": "x",
                       "tier": 1, "status": "proposed"}])
    client = _client(monkeypatch, user_id="userA", proc=proc)
    resp = client.post("/api/memory/dreaming/procedural/format.bullets/approve")
    assert resp.status_code == 200
    assert proc.rows[("userA", "format.bullets")]["status"] == "active"


def test_approve_non_proposed_rule_is_409(monkeypatch):
    proc = _FakeProc([{"user_id": "userA", "rule_key": "format.bullets", "rule_text": "x",
                       "tier": 1, "status": "observing"}])  # not yet surfaced
    client = _client(monkeypatch, user_id="userA", proc=proc)
    assert client.post("/api/memory/dreaming/procedural/format.bullets/approve").status_code == 409


def test_approve_other_users_rule_is_404(monkeypatch):
    proc = _FakeProc([{"user_id": "userB", "rule_key": "format.bullets", "rule_text": "x",
                       "tier": 1, "status": "proposed"}])
    client = _client(monkeypatch, user_id="userA", proc=proc)
    assert client.post("/api/memory/dreaming/procedural/format.bullets/approve").status_code == 404


def test_reject_proposed_rule(monkeypatch):
    proc = _FakeProc([{"user_id": "userA", "rule_key": "style.terse", "rule_text": "x",
                       "tier": 2, "status": "proposed"}])
    client = _client(monkeypatch, user_id="userA", proc=proc)
    client.post("/api/memory/dreaming/procedural/style.terse/reject")
    assert proc.rows[("userA", "style.terse")]["status"] == "rejected"


# --------------------------------------------------------------------------- #
# Undo feed
# --------------------------------------------------------------------------- #
def _audit_rows():
    return _FakeAudit([
        {"audit_id": "future", "user_id": "userA", "cycle": "forgetting", "action": "delete",
         "target_id": "m1", "before_state": {"text": "old memory", "metadata": {}},
         "reversible_until": NOW + timedelta(days=5)},
        {"audit_id": "expired", "user_id": "userA", "cycle": "forgetting", "action": "delete",
         "target_id": "m2", "before_state": {"text": "x"}, "reversible_until": NOW - timedelta(days=1)},
        {"audit_id": "permanent", "user_id": "userA", "cycle": "drift", "action": "lower_confidence",
         "target_id": "s1", "before_state": {"confidence": 0.8}, "reversible_until": None},
        {"audit_id": "b-future", "user_id": "userB", "action": "delete",
         "before_state": {"text": "b"}, "reversible_until": NOW + timedelta(days=5)},
    ])


def test_audit_feed_lists_only_own_reversible(monkeypatch):
    client = _client(monkeypatch, user_id="userA", audit=_audit_rows())
    ids = [a["audit_id"] for a in client.get("/api/memory/dreaming/audit").json()["reversible"]]
    assert ids == ["future"]  # expired + permanent + user B excluded


def test_undo_within_window_restores_and_marks_undone(monkeypatch):
    audit = _audit_rows()
    client = _client(monkeypatch, user_id="userA", audit=audit)
    resp = client.post("/api/memory/dreaming/audit/future/undo")
    assert resp.status_code == 200 and resp.json()["status"] == "undone"
    assert ("restore", "userA", "old memory") in client._fake_backend.calls
    assert audit.rows["future"]["reversible_until"] is None  # can't be undone twice


def test_undo_expired_window_is_410(monkeypatch):
    client = _client(monkeypatch, user_id="userA", audit=_audit_rows())
    assert client.post("/api/memory/dreaming/audit/expired/undo").status_code == 410


def test_undo_non_reversible_is_409(monkeypatch):
    client = _client(monkeypatch, user_id="userA", audit=_audit_rows())
    assert client.post("/api/memory/dreaming/audit/permanent/undo").status_code == 409


def test_undo_other_users_action_is_404(monkeypatch):
    client = _client(monkeypatch, user_id="userA", audit=_audit_rows())
    assert client.post("/api/memory/dreaming/audit/b-future/undo").status_code == 404


# --------------------------------------------------------------------------- #
# Admin metrics — aggregate only, no PII
# --------------------------------------------------------------------------- #
def test_admin_metrics_are_aggregate_only(monkeypatch):
    conflict = _FakeConflict([
        {"conflict_id": "c1", "user_id": "userA", "semantic_text": "SENSITIVE TEXT", "choices": []},
    ])
    proc = _FakeProc([{"user_id": "userA", "rule_key": "format.x", "rule_text": "SECRET RULE",
                       "tier": 1, "status": "proposed"}])
    client = _client(monkeypatch, user_id="admin", conflict=conflict, proc=proc, admin=True)

    resp = client.get("/api/admin/dreaming-metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["cycles_run"] == 10
    assert body["vector_count"] == 1234
    assert body["total_tokens"] == 500
    assert body["pending_resolutions"] == {"open_conflicts": 1, "proposed_procedural": 1, "total": 2}
    assert body["deletion_count"] == 3 and body["promotion_count"] == 1

    # Structurally PII-free: no user text/ids/names anywhere in the payload.
    blob = resp.text
    assert "SENSITIVE TEXT" not in blob
    assert "SECRET RULE" not in blob
    assert "userA" not in blob and "format.x" not in blob
