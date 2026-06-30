"""Phase 1 tests: cognitive-memory Postgres contracts + Stores.

Two layers:
- **No-DB** (always run): a fake cursor captures the SQL each Store emits, so we
  assert `aggregate_metrics()` selects no PII columns (admin §5A) and that the
  upsert/open paths use `ON CONFLICT`, plus the Python rollup math.
- **DB round-trips** (skipped if Postgres is unreachable): real CRUD / ON CONFLICT
  idempotency / undo-window / prune behaviour against the live schema.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from backend.db import (
    DatabaseConfig,
    DatabasePool,
    MemoryAuditLogStore,
    MemoryConflictStore,
    ProceduralOverrideStore,
)


# --------------------------------------------------------------------------- #
# No-DB fakes — capture executed SQL + drive the Python rollup logic
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.fetchall_result: list[Any] = []
        self.fetchone_result: Any = None

    def execute(self, statement, params=None) -> None:
        self.executed.append((statement, params))

    def fetchall(self):
        return self.fetchall_result

    def fetchone(self):
        return self.fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, *a) -> None:
        return None


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self) -> None:
        pass


class _FakePool:
    def __init__(self) -> None:
        self.cursor = _FakeCursor()

    def connection(self):
        pool = self

        class _Ctx:
            def __enter__(self_inner):
                return _FakeConn(pool.cursor)

            def __exit__(self_inner, *a):
                return None

        return _Ctx()


# Column names that must never appear in an aggregate_metrics() query.
_PII_COLUMNS = (
    "user_id",
    "rule_text",
    "rule_key",
    "semantic_text",
    "contradiction_text",
    "semantic_memory_id",
    "episodic_memory_id",
    "target_id",
    "before_state",
    "after_state",
    "resolution",
    "evidence",
)


def _last_sql(cursor: _FakeCursor) -> str:
    return cursor.executed[-1][0]


def test_procedural_aggregate_sql_is_pii_free_and_rolls_up():
    pool = _FakePool()
    store = ProceduralOverrideStore(pool)  # __init__ runs DDL via the fake cursor
    pool.cursor.fetchall_result = [
        {"status": "proposed", "tier": 1, "n": 2},
        {"status": "active", "tier": 1, "n": 1},
        {"status": "active", "tier": 2, "n": 3},
    ]
    metrics = store.aggregate_metrics()

    sql = _last_sql(pool.cursor).lower()
    for col in _PII_COLUMNS:
        assert col not in sql, f"aggregate SQL leaked PII column {col}: {sql}"

    assert metrics["total"] == 6
    assert metrics["by_status"] == {"proposed": 2, "active": 4}
    assert metrics["by_tier"] == {"1": 3, "2": 3}
    assert metrics["proposed_pending"] == 2


def test_procedural_upsert_uses_on_conflict_and_keeps_status_untouched():
    pool = _FakePool()
    store = ProceduralOverrideStore(pool)
    pool.cursor.fetchone_result = {"user_id": "u", "rule_key": "format.bullets", "status": "proposed"}
    store.upsert_proposal(
        user_id="u", rule_key="format.bullets", rule_text="use bullets", tier=1
    )
    sql = _last_sql(pool.cursor).upper()
    assert "ON CONFLICT" in sql
    # The conflict branch must not overwrite status (only set_status mutates it).
    assert "STATUS" not in sql.split("DO UPDATE SET")[1]


def test_conflict_aggregate_sql_is_pii_free():
    pool = _FakePool()
    store = MemoryConflictStore(pool)
    pool.cursor.fetchall_result = [
        {"status": "open", "n": 3},
        {"status": "resolved", "n": 5},
    ]
    metrics = store.aggregate_metrics()
    sql = _last_sql(pool.cursor).lower()
    for col in _PII_COLUMNS:
        assert col not in sql
    assert metrics == {"total": 8, "by_status": {"open": 3, "resolved": 5}, "open": 3}


def test_conflict_open_uses_on_conflict_do_nothing():
    pool = _FakePool()
    store = MemoryConflictStore(pool)
    pool.cursor.fetchone_result = {"conflict_id": "abc"}
    store.open_conflict(
        user_id="u", semantic_memory_id="s1", episodic_memory_id="e1"
    )
    # The INSERT (second-to-last; last may be the existing-id SELECT if no row) —
    # find the INSERT statement among executed.
    inserts = [s for s, _ in pool.cursor.executed if "INSERT INTO memory_conflicts" in s]
    assert inserts and "ON CONFLICT" in inserts[0].upper() and "DO NOTHING" in inserts[0].upper()


def test_audit_aggregate_sql_is_pii_free_and_rolls_up():
    pool = _FakePool()
    store = MemoryAuditLogStore(pool)
    pool.cursor.fetchall_result = [
        {"cycle": "forgetting", "action": "delete", "n": 4},
        {"cycle": "promotion", "action": "promote", "n": 2},
        {"cycle": "forgetting", "action": "delete_skip", "n": 1},
    ]
    metrics = store.aggregate_metrics()
    sql = _last_sql(pool.cursor).lower()
    for col in _PII_COLUMNS:
        assert col not in sql
    assert metrics["total"] == 7
    assert metrics["by_cycle"] == {"forgetting": 5, "promotion": 2}
    assert metrics["by_action"] == {"delete": 4, "promote": 2, "delete_skip": 1}


def test_procedural_set_status_rejects_invalid():
    pool = _FakePool()
    store = ProceduralOverrideStore(pool)
    with pytest.raises(ValueError):
        store.set_status(user_id="u", rule_key="k", status="bogus")


# --------------------------------------------------------------------------- #
# DB round-trips — skipped when Postgres is unreachable
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def db_pool():
    dsn = (
        f'postgresql://framework:framework@'
        f'{os.environ.get("TEST_DB_HOST", "localhost")}:5432/framework'
    )
    try:
        pool = DatabasePool(DatabaseConfig(dsn=dsn, minconn=1, maxconn=3))
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable for Phase 1 store round-trips: {exc}")
    yield pool
    pool.close()


@pytest.fixture
def user_id(db_pool):
    """A unique user per test; teardown wipes its rows from all three tables."""
    uid = f"dream-test-{uuid.uuid4().hex[:10]}"
    yield uid
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM procedural_overrides WHERE user_id = %s;", (uid,))
            cur.execute("DELETE FROM memory_conflicts WHERE user_id = %s;", (uid,))
            cur.execute("DELETE FROM memory_audit_log WHERE user_id = %s;", (uid,))
        conn.commit()


def test_procedural_crud_and_conflict_preserves_status(db_pool, user_id):
    store = ProceduralOverrideStore(db_pool)
    store.upsert_proposal(
        user_id=user_id,
        rule_key="format.bullets",
        rule_text="prefer bullet lists",
        tier=1,
        confidence=0.5,
        evidence={"sample_count": 3, "examples": ["a", "b"]},
    )
    rows = store.list_for_user(user_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "proposed"
    assert rows[0]["tier"] == 1
    assert rows[0]["evidence"]["sample_count"] == 3

    # Approve → active, then re-propose with fresh evidence: status must stay active.
    assert store.set_status(user_id=user_id, rule_key="format.bullets", status="active")
    store.upsert_proposal(
        user_id=user_id,
        rule_key="format.bullets",
        rule_text="prefer bullet lists (refined)",
        tier=1,
        confidence=0.9,
        evidence={"sample_count": 9},
    )
    active = store.list_active(user_id)
    assert len(active) == 1
    assert active[0]["status"] == "active"
    assert active[0]["confidence"] == 0.9
    assert active[0]["evidence"]["sample_count"] == 9


def test_procedural_set_status_is_user_scoped(db_pool, user_id):
    store = ProceduralOverrideStore(db_pool)
    store.upsert_proposal(user_id=user_id, rule_key="style.terse", rule_text="be terse", tier=2)
    # A different user cannot mutate this rule.
    assert store.set_status(user_id="someone-else", rule_key="style.terse", status="active") is None
    assert store.set_status(user_id=user_id, rule_key="style.terse", status="rejected")["status"] == "rejected"


def test_procedural_observation_matures_then_promotes(db_pool, user_id):
    store = ProceduralOverrideStore(db_pool)
    # First observation → observing, not surfaced and not active.
    row = store.upsert_observation(
        user_id=user_id, rule_key="format.bullets", rule_text="use bullets", tier=1,
        evidence={"observation_days": ["2026-06-29"], "sample_count": 1},
    )
    assert row["status"] == "observing"
    assert store.list_active(user_id) == []  # observing never reaches the prompt

    # Evidence refresh keeps status observing (no premature flip).
    store.upsert_observation(
        user_id=user_id, rule_key="format.bullets", rule_text="use bullets", tier=1,
        evidence={"observation_days": ["2026-06-29", "2026-06-30"], "sample_count": 2},
    )
    assert store.get(user_id, "format.bullets")["status"] == "observing"

    # Mature → proposed (surfaced for sign-off, still not active).
    promoted = store.promote_to_proposed(user_id=user_id, rule_key="format.bullets")
    assert promoted["status"] == "proposed"
    assert store.list_active(user_id) == []

    # Approve → active (now loadable into the prompt).
    store.set_status(user_id=user_id, rule_key="format.bullets", status="active")
    active = store.list_active(user_id)
    assert len(active) == 1 and active[0]["rule_key"] == "format.bullets"


def test_procedural_promote_is_guarded_to_observing(db_pool, user_id):
    store = ProceduralOverrideStore(db_pool)
    store.upsert_observation(
        user_id=user_id, rule_key="format.bullets", rule_text="x", tier=1, evidence={},
    )
    store.set_status(user_id=user_id, rule_key="format.bullets", status="rejected")
    # A rejected rule must not be reopened by promotion.
    assert store.promote_to_proposed(user_id=user_id, rule_key="format.bullets") is None
    assert store.get(user_id, "format.bullets")["status"] == "rejected"


def test_conflict_open_is_idempotent_and_resolve_is_scoped(db_pool, user_id):
    store = MemoryConflictStore(db_pool)
    cid1 = store.open_conflict(
        user_id=user_id,
        semantic_memory_id="sem-1",
        episodic_memory_id="epi-1",
        semantic_text="user likes tea",
        contradiction_text="user said they prefer coffee",
        prior_confidence=0.6,
        new_confidence=0.3,
        choices=[{"id": "keep_old"}, {"id": "accept_new"}, {"id": "depends"}],
    )
    # Re-detecting the same triple must NOT create a duplicate.
    cid2 = store.open_conflict(
        user_id=user_id, semantic_memory_id="sem-1", episodic_memory_id="epi-1"
    )
    assert cid1 == cid2
    assert len(store.list_open(user_id)) == 1

    # Cross-user resolve is impossible.
    assert store.resolve(conflict_id=cid1, user_id="intruder", resolution={"action": "keep_old"}) is None
    resolved = store.resolve(conflict_id=cid1, user_id=user_id, resolution={"action": "accept_new"})
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["action"] == "accept_new"
    assert store.list_open(user_id) == []


def test_audit_record_last_cycle_run_reversible_and_prune(db_pool, user_id):
    store = MemoryAuditLogStore(db_pool)
    now = datetime.now(timezone.utc)

    # A reversible delete (inside undo window) + a non-reversible marker.
    store.record(
        user_id=user_id,
        cycle="forgetting",
        action="delete",
        target_kind="episodic",
        target_id="mem-1",
        before_state={"text": "old memory", "user_id": user_id},
        reversible_until=now + timedelta(days=7),
    )
    store.record(
        user_id=user_id,
        cycle="promotion",
        action="promote",
        target_kind="semantic",
        target_id="mem-2",
        reversible_until=None,
    )

    # last_cycle_run resolves per-cycle.
    assert store.last_cycle_run(user_id=user_id, cycle="forgetting") is not None
    assert store.last_cycle_run(user_id=user_id, cycle="drift") is None

    # Only the row still inside its undo window is reversible.
    reversible = store.list_reversible(user_id=user_id, now=now)
    assert len(reversible) == 1
    assert reversible[0]["target_id"] == "mem-1"
    assert reversible[0]["before_state"]["text"] == "old memory"

    # Prune respects the cutoff predicate: a past cutoff deletes nothing,
    # a future cutoff removes this user's rows.
    assert store.prune(cutoff=now - timedelta(days=1)) == 0
    pruned = store.prune(cutoff=now + timedelta(days=1))
    assert pruned >= 2
    assert store.list_reversible(user_id=user_id, now=now) == []
