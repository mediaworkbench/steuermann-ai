"""Phase 3 tests: Dreaming Engine cheap cycles (Forgetting + Drift) + isolation.

All dependencies are faked — no Qdrant, LLM, or Postgres required. Covers the
plan's acceptance list: a 31-day untouched non-contributor is deleted+audited;
a contradiction lowers a semantic's confidence and opens a conflict below the
floor; provider-offline defers drift while forgetting runs (status partial,
last_cycle_run not advanced); double-runs are idempotent; per-user breakers are
isolated; and the semaphore serialises ticks to one user at a time.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from universal_agentic_framework.config.schemas import CognitiveMemorySettings
from universal_agentic_framework.heartbeat.task import TickContext
from universal_agentic_framework.heartbeat.tasks.dreaming import DreamingEngineTask


NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeAudit:
    def __init__(self, last: Optional[Dict[str, datetime]] = None) -> None:
        self.records: List[Dict[str, Any]] = []
        self._last = last or {}

    def last_cycle_run(self, *, user_id: str, cycle: str) -> Optional[datetime]:
        return self._last.get(cycle)

    def record(self, *, user_id, cycle, action, target_kind=None, target_id=None,
               before_state=None, after_state=None, reversible_until=None) -> str:
        self.records.append({
            "user_id": user_id, "cycle": cycle, "action": action,
            "target_kind": target_kind, "target_id": target_id,
            "before_state": before_state, "after_state": after_state,
            "reversible_until": reversible_until,
        })
        return "audit-%d" % len(self.records)

    def of(self, action: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["action"] == action]


class _FakeConflicts:
    def __init__(self) -> None:
        self.opened: List[Dict[str, Any]] = []
        self._keys: set = set()

    def open_conflict(self, *, user_id, semantic_memory_id, episodic_memory_id, **kw) -> str:
        key = (user_id, semantic_memory_id, episodic_memory_id)
        if key in self._keys:  # idempotent (mirrors ON CONFLICT DO NOTHING)
            return "existing"
        self._keys.add(key)
        self.opened.append({"semantic": semantic_memory_id, "episodic": episodic_memory_id, **kw})
        return "cid-%d" % len(self.opened)


class _FakeRunStore:
    def __init__(self) -> None:
        self.runs: List[Dict[str, Any]] = []

    def record_run(self, *, task_name, status, duration_ms=0, detail=None, user_id=None) -> None:
        self.runs.append({"task": task_name, "status": status, "detail": detail or {}, "user_id": user_id})

    def recent_runs(self, task_name, limit=5, *, user_id=None):
        return []  # never in cooldown


class _FakeReader:
    def __init__(self, items: Optional[List[Dict[str, Any]]] = None,
                 nearest: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._items = {i["memory_id"]: i for i in (items or [])}
        self._nearest = nearest or {}
        self.deleted: List[str] = []
        self.confidence_set: List[tuple] = []

    def fetch_all(self, user_id):
        return list(self._items.values())

    def nearest_semantic(self, user_id, text):
        return self._nearest.get(text)

    def delete_memory(self, user_id, memory_id):
        self.deleted.append(memory_id)
        self._items.pop(memory_id, None)  # gone on the next fetch (idempotency)

    def set_confidence(self, user_id, memory_id, confidence):
        self.confidence_set.append((memory_id, confidence))


def _episodic(mid, *, age_days, access=0, contributor=False, text="ep") -> Dict[str, Any]:
    return {
        "memory_id": mid,
        "text": text,
        "metadata": {
            "memory_id": mid,
            "cognitive_tier": "episodic",
            "created_at": (NOW - timedelta(days=age_days)).isoformat(),
            "access_count": access,
            "epiphany_contributor": contributor,
        },
    }


async def _always_reachable() -> bool:
    return True


async def _never_reachable() -> bool:
    return False


def _make_task(reader, *, audit=None, conflicts=None, run_store=None, adjudicator=None,
               health=_always_reachable, opt_out=lambda _u: False, semaphore=None,
               settings=None) -> DreamingEngineTask:
    async def _default_adj(_s, _e):
        return {"contradicts": False, "tokens": 5}

    return DreamingEngineTask(
        name="dreaming",
        cooldown_seconds=0,
        scope="per_user",
        run_store=run_store or _FakeRunStore(),
        audit_store=audit or _FakeAudit(),
        conflict_store=conflicts or _FakeConflicts(),
        reader=reader,
        adjudicator=adjudicator or _default_adj,
        health_gate=health,
        opt_out_checker=opt_out,
        settings=settings or CognitiveMemorySettings(),
        semaphore=semaphore,
        clock=lambda: NOW,
    )


# --------------------------------------------------------------------------- #
# Cycle C — Forgetting
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_forgetting_deletes_old_untouched_noncontributor_and_audits():
    reader = _FakeReader(items=[
        _episodic("old", age_days=31),                       # → deleted
        _episodic("recent", age_days=5),                     # too new
        _episodic("old-touched", age_days=40, access=3),     # retrieved before
        _episodic("old-contrib", age_days=40, contributor=True),  # epiphany source
    ])
    audit = _FakeAudit()
    task = _make_task(reader, audit=audit)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "ok"
    assert reader.deleted == ["old"]
    deletes = audit.of("delete")
    assert len(deletes) == 1
    assert deletes[0]["target_id"] == "old"
    assert deletes[0]["before_state"]["memory_id"] == "old"   # full record for undo
    assert deletes[0]["reversible_until"] == NOW + timedelta(days=7)
    # Forgetting completed → a cycle_run marker advances last_cycle_run.
    assert any(r["cycle"] == "forgetting" and r["action"] == "cycle_run" for r in audit.records)


@pytest.mark.asyncio
async def test_forgetting_skips_when_semantic_tier():
    # A semantic record that would otherwise match the age/access predicate must
    # never be forgotten (forgetting is episodic-only).
    sem = {
        "memory_id": "sem", "text": "fact",
        "metadata": {"memory_id": "sem", "cognitive_tier": "semantic",
                     "created_at": (NOW - timedelta(days=99)).isoformat(), "access_count": 0},
    }
    reader = _FakeReader(items=[sem])
    task = _make_task(reader)
    await task.tick(TickContext(user_id="u1"))
    assert reader.deleted == []


# --------------------------------------------------------------------------- #
# Cycle B — Drift
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_drift_contradiction_lowers_confidence_and_opens_conflict():
    reader = _FakeReader(
        items=[_episodic("ep1", age_days=0, text="I moved to Berlin")],
        nearest={"I moved to Berlin": {"memory_id": "sem1", "text": "lives in Munich", "confidence": 0.35}},
    )
    audit = _FakeAudit()
    conflicts = _FakeConflicts()

    async def _contradicts(_s, _e):
        return {"contradicts": True, "tokens": 12}

    task = _make_task(reader, audit=audit, conflicts=conflicts, adjudicator=_contradicts)
    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "ok"
    # 0.35 − 0.15 = 0.20, below the 0.30 floor → conflict opened.
    assert reader.confidence_set == [("sem1", pytest.approx(0.20))]
    assert len(conflicts.opened) == 1
    assert conflicts.opened[0]["semantic"] == "sem1"
    assert audit.of("lower_confidence")[0]["after_state"]["confidence"] == pytest.approx(0.20)
    assert task is not None


@pytest.mark.asyncio
async def test_drift_above_floor_lowers_without_conflict():
    reader = _FakeReader(
        items=[_episodic("ep1", age_days=0, text="q")],
        nearest={"q": {"memory_id": "sem1", "text": "s", "confidence": 0.9}},
    )
    conflicts = _FakeConflicts()

    async def _contradicts(_s, _e):
        return {"contradicts": True, "tokens": 1}

    task = _make_task(reader, conflicts=conflicts, adjudicator=_contradicts)
    await task.tick(TickContext(user_id="u1"))
    assert reader.confidence_set == [("sem1", pytest.approx(0.75))]  # 0.9 − 0.15
    assert conflicts.opened == []  # 0.75 ≥ floor


@pytest.mark.asyncio
async def test_drift_ambiguous_verdict_takes_no_action():
    reader = _FakeReader(
        items=[_episodic("ep1", age_days=0, text="q")],
        nearest={"q": {"memory_id": "sem1", "text": "s", "confidence": 0.5}},
    )

    async def _ambiguous(_s, _e):
        return {"contradicts": None, "tokens": 3}  # unparseable/ambiguous

    task = _make_task(reader, adjudicator=_ambiguous)
    await task.tick(TickContext(user_id="u1"))
    assert reader.confidence_set == []


# --------------------------------------------------------------------------- #
# Resilience: provider offline
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_provider_offline_defers_drift_but_runs_forgetting():
    reader = _FakeReader(
        items=[
            _episodic("old", age_days=31),
            _episodic("ep1", age_days=0, text="q"),
        ],
        nearest={"q": {"memory_id": "sem1", "text": "s", "confidence": 0.4}},
    )
    audit = _FakeAudit()
    run_store = _FakeRunStore()
    task = _make_task(reader, audit=audit, run_store=run_store, health=_never_reachable)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "partial"
    assert reader.deleted == ["old"]            # forgetting still ran (no LLM)
    assert reader.confidence_set == []          # drift deferred
    last = run_store.runs[-1]
    assert last["status"] == "partial"
    assert last["detail"]["reason"] == "provider_offline"
    assert "drift" in last["detail"]["deferred"]
    # Partial run must NOT advance drift's last_cycle_run (no drift marker).
    assert not any(r["cycle"] == "drift" and r["action"] == "cycle_run" for r in audit.records)
    # Forgetting did complete.
    assert any(r["cycle"] == "forgetting" and r["action"] == "cycle_run" for r in audit.records)


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_double_run_is_idempotent():
    reader = _FakeReader(
        items=[_episodic("old", age_days=31), _episodic("ep1", age_days=0, text="q")],
        nearest={"q": {"memory_id": "sem1", "text": "s", "confidence": 0.4}},
    )
    conflicts = _FakeConflicts()

    async def _contradicts(_s, _e):
        return {"contradicts": True, "tokens": 1}

    task = _make_task(reader, conflicts=conflicts, adjudicator=_contradicts)

    await task.tick(TickContext(user_id="u1"))
    await task.tick(TickContext(user_id="u1"))

    assert reader.deleted == ["old"]       # deleted once; gone on the 2nd fetch
    assert len(conflicts.opened) == 1      # same triple → opened once


# --------------------------------------------------------------------------- #
# Opt-out
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_opted_out_user_is_skipped():
    reader = _FakeReader(items=[_episodic("old", age_days=31)])
    run_store = _FakeRunStore()
    task = _make_task(reader, run_store=run_store, opt_out=lambda _u: True)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "skipped"
    assert reader.deleted == []
    assert run_store.runs[-1]["status"] == "skipped"
    assert run_store.runs[-1]["detail"]["reason"] == "opt_out"


# --------------------------------------------------------------------------- #
# Per-user circuit-breaker isolation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_per_user_breaker_isolation():
    class _SelectiveReader(_FakeReader):
        def fetch_all(self, user_id):
            if user_id == "A":
                raise RuntimeError("A observe boom")
            return super().fetch_all(user_id)

    reader = _SelectiveReader(items=[_episodic("old", age_days=31)])
    task = _make_task(reader)

    a = await task.tick(TickContext(user_id="A"))
    b = await task.tick(TickContext(user_id="B"))

    assert a["status"] == "error"          # A's observe failed
    assert b["status"] == "ok"             # B unaffected
    assert reader.deleted == ["old"]       # B's forgetting still ran
    # Breakers are per-user instances.
    assert task._breaker_for("A") is not task._breaker_for("B")


# --------------------------------------------------------------------------- #
# One-by-one semaphore
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_semaphore_serialises_ticks_to_one_at_a_time():
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    class _SlowReader(_FakeReader):
        def fetch_all(self, user_id):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return []

    task = _make_task(_SlowReader(), semaphore=asyncio.Semaphore(1))
    await asyncio.gather(
        task.tick(TickContext(user_id="A")),
        task.tick(TickContext(user_id="B")),
    )
    assert state["max"] == 1  # never two users dreaming at once


@pytest.mark.asyncio
async def test_concurrency_above_one_allows_overlap():
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    class _SlowReader(_FakeReader):
        def fetch_all(self, user_id):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return []

    task = _make_task(_SlowReader(), semaphore=asyncio.Semaphore(2))
    await asyncio.gather(
        task.tick(TickContext(user_id="A")),
        task.tick(TickContext(user_id="B")),
    )
    assert state["max"] == 2  # the semaphore is what serialises, not luck
