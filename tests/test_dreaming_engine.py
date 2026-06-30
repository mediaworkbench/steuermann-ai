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


class _FakeProcedural:
    """In-memory ProceduralOverrideStore with the observing→proposed semantics."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}

    def get(self, user_id, rule_key):
        return self.rows.get(rule_key)

    def upsert_observation(self, *, user_id, rule_key, rule_text, tier, confidence=0.0, evidence=None):
        existing = self.rows.get(rule_key)
        status = existing["status"] if existing else "observing"  # status untouched on conflict
        self.rows[rule_key] = {
            "user_id": user_id, "rule_key": rule_key, "rule_text": rule_text, "tier": tier,
            "status": status, "confidence": confidence, "evidence": dict(evidence or {}),
        }
        return dict(self.rows[rule_key])

    def promote_to_proposed(self, *, user_id, rule_key):
        row = self.rows.get(rule_key)
        if row and row["status"] == "observing":
            row["status"] = "proposed"
            return dict(row)
        return None  # guarded: no-op unless currently observing

    def list_active(self, user_id):
        return [dict(r) for r in self.rows.values() if r["status"] == "active"]


class _FakeReader:
    def __init__(self, items: Optional[List[Dict[str, Any]]] = None,
                 nearest: Optional[Dict[str, Dict[str, Any]]] = None,
                 vectors: Optional[List[Dict[str, Any]]] = None,
                 sources: Optional[List[List[str]]] = None) -> None:
        self._items = {i["memory_id"]: i for i in (items or [])}
        self._nearest = nearest or {}
        self._vectors = vectors or []
        self._sources = sources or []
        self.deleted: List[str] = []
        self.confidence_set: List[tuple] = []
        self.written: List[Dict[str, Any]] = []
        self.flagged: List[str] = []
        self.vectors_fetched = 0

    def fetch_all(self, user_id):
        return list(self._items.values())

    def nearest_semantic(self, user_id, text):
        return self._nearest.get(text)

    def delete_memory(self, user_id, memory_id):
        self.deleted.append(memory_id)
        self._items.pop(memory_id, None)  # gone on the next fetch (idempotency)

    def set_confidence(self, user_id, memory_id, confidence):
        self.confidence_set.append((memory_id, confidence))

    # --- promotion (Cycle A) ---
    def fetch_episodic_vectors(self, user_id):
        self.vectors_fetched += 1
        return list(self._vectors)

    def existing_semantic_sources(self, user_id):
        return [list(s) for s in self._sources]

    def write_semantic(self, user_id, text, confidence, source_episodic_ids):
        sid = "sem-%d" % (len(self.written) + 1)
        self.written.append(
            {"id": sid, "text": text, "confidence": confidence, "sources": list(source_episodic_ids)}
        )
        return sid

    def flag_contributor(self, user_id, memory_id):
        self.flagged.append(memory_id)


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
               synthesizer=None, proposer=None, procedural=None, health=_always_reachable,
               opt_out=lambda _u: False, semaphore=None, settings=None, clock=None) -> DreamingEngineTask:
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
        synthesizer=synthesizer,
        proposer=proposer,
        procedural_store=procedural,
        health_gate=health,
        opt_out_checker=opt_out,
        settings=settings or CognitiveMemorySettings(),
        semaphore=semaphore,
        clock=clock or (lambda: NOW),
    )


# Three near-collinear vectors (one cluster) + one orthogonal (excluded).
def _cluster_vectors() -> List[Dict[str, Any]]:
    return [
        {"memory_id": "e1", "text": "drinks oat milk latte", "vector": [1.0, 0.0, 0.0]},
        {"memory_id": "e2", "text": "ordered an oat latte", "vector": [0.99, 0.02, 0.0]},
        {"memory_id": "e3", "text": "oat milk in coffee", "vector": [0.98, 0.03, 0.0]},
        {"memory_id": "e4", "text": "unrelated", "vector": [0.0, 1.0, 0.0]},
    ]


async def _synth_ok(_texts):
    return {"text": "User prefers oat milk in coffee.", "tokens": 20}


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


# --------------------------------------------------------------------------- #
# Cycle A — Promotion / Epiphany (Phase 4)
# --------------------------------------------------------------------------- #
def test_strip_reasoning_removes_think_blocks():
    # Reasoning models inline <think>…</think>; only the final answer must survive
    # (regression for the live finding that synthesis returned empty content).
    from universal_agentic_framework.heartbeat.tasks.dreaming import _strip_reasoning

    assert _strip_reasoning("<think>let me consider</think> User likes tea.") == "User likes tea."
    assert _strip_reasoning("User likes tea.") == "User likes tea."
    # An unterminated (token-truncated) think block collapses to empty.
    assert _strip_reasoning("<think>thinking but ran out of tokens") == ""


def test_greedy_cosine_clusters_pure():
    from universal_agentic_framework.heartbeat.tasks.dreaming import greedy_cosine_clusters

    items = _cluster_vectors()
    clusters = greedy_cosine_clusters(items, threshold=0.9, min_size=3)
    assert len(clusters) == 1
    assert {m["memory_id"] for m in clusters[0]} == {"e1", "e2", "e3"}

    # Below min_size → no cluster; a zero vector is ignored, not crashed on.
    assert greedy_cosine_clusters(items, threshold=0.9, min_size=4) == []
    assert greedy_cosine_clusters(
        [{"memory_id": "z", "text": "t", "vector": [0.0, 0.0, 0.0]}], threshold=0.5, min_size=1
    ) == []


@pytest.mark.asyncio
async def test_promotion_clusters_to_one_semantic_and_flags_contributors():
    reader = _FakeReader(vectors=_cluster_vectors())
    audit = _FakeAudit()
    task = _make_task(reader, audit=audit, synthesizer=_synth_ok)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "ok"
    assert len(reader.written) == 1
    written = reader.written[0]
    assert written["text"] == "User prefers oat milk in coffee."
    assert written["confidence"] == pytest.approx(0.6)
    assert set(written["sources"]) == {"e1", "e2", "e3"}
    # Each source episodic is flagged as an epiphany contributor (spared by GC).
    assert set(reader.flagged) == {"e1", "e2", "e3"}
    promote_rows = [r for r in audit.records if r["action"] == "promote"]
    assert len(promote_rows) == 1
    assert promote_rows[0]["after_state"]["source_episodic_ids"] == ["e1", "e2", "e3"]
    assert promote_rows[0]["reversible_until"] == NOW + timedelta(days=7)
    assert any(r["cycle"] == "promotion" and r["action"] == "cycle_run" for r in audit.records)


@pytest.mark.asyncio
async def test_promotion_skips_cluster_already_covered():
    reader = _FakeReader(vectors=_cluster_vectors(), sources=[["e1", "e2", "e3"]])
    task = _make_task(reader, synthesizer=_synth_ok)

    await task.tick(TickContext(user_id="u1"))

    assert reader.written == []   # already covered by an existing semantic
    assert reader.flagged == []


@pytest.mark.asyncio
async def test_promotion_respects_interval_days():
    # Promotion ran 2 days ago; interval is 7 → not due, no clustering at all.
    reader = _FakeReader(vectors=_cluster_vectors())
    audit = _FakeAudit(last={"promotion": NOW - timedelta(days=2), "drift": NOW})
    task = _make_task(reader, audit=audit, synthesizer=_synth_ok)

    await task.tick(TickContext(user_id="u1"))

    assert reader.written == []
    assert reader.vectors_fetched == 0  # not even read when the cycle isn't due


@pytest.mark.asyncio
async def test_promotion_offline_defers_without_writing():
    reader = _FakeReader(vectors=_cluster_vectors())
    audit = _FakeAudit()
    run_store = _FakeRunStore()
    task = _make_task(reader, audit=audit, run_store=run_store,
                      synthesizer=_synth_ok, health=_never_reachable)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "partial"
    assert reader.written == []
    assert "promotion" in run_store.runs[-1]["detail"]["deferred"]
    assert not any(r["cycle"] == "promotion" and r["action"] == "cycle_run" for r in audit.records)


@pytest.mark.asyncio
async def test_promotion_cap_limits_writes_per_run():
    # Two disjoint clusters of 3; cap = 1 → only one promotion this run.
    vectors = [
        {"memory_id": "a1", "text": "a one", "vector": [1.0, 0.0, 0.0]},
        {"memory_id": "a2", "text": "a two", "vector": [0.99, 0.01, 0.0]},
        {"memory_id": "a3", "text": "a three", "vector": [0.98, 0.02, 0.0]},
        {"memory_id": "b1", "text": "b one", "vector": [0.0, 1.0, 0.0]},
        {"memory_id": "b2", "text": "b two", "vector": [0.0, 0.99, 0.01]},
        {"memory_id": "b3", "text": "b three", "vector": [0.0, 0.98, 0.02]},
    ]
    reader = _FakeReader(vectors=vectors)
    task = _make_task(reader, synthesizer=_synth_ok,
                      settings=CognitiveMemorySettings(max_promotions_per_run=1))

    await task.tick(TickContext(user_id="u1"))
    assert len(reader.written) == 1


@pytest.mark.asyncio
async def test_promotion_rejects_empty_synthesis():
    reader = _FakeReader(vectors=_cluster_vectors())

    async def _synth_empty(_texts):
        return {"text": "   ", "tokens": 3}

    task = _make_task(reader, synthesizer=_synth_empty)
    await task.tick(TickContext(user_id="u1"))
    assert reader.written == []   # garbage synthesis is skipped, nothing written


@pytest.mark.asyncio
async def test_promotion_contributor_not_forgotten_same_tick():
    # An episodic that is both GC-eligible (old, untouched) AND a promotion source
    # must survive this tick (its contributor flag isn't persisted until act()).
    old_contrib = _episodic("e1", age_days=99)  # old + untouched → GC-eligible
    reader = _FakeReader(
        items=[old_contrib],
        vectors=[
            {"memory_id": "e1", "text": "x", "vector": [1.0, 0.0, 0.0]},
            {"memory_id": "e2", "text": "y", "vector": [0.99, 0.01, 0.0]},
            {"memory_id": "e3", "text": "z", "vector": [0.98, 0.02, 0.0]},
        ],
    )
    task = _make_task(reader, synthesizer=_synth_ok)

    await task.tick(TickContext(user_id="u1"))

    assert len(reader.written) == 1
    assert "e1" not in reader.deleted   # de-conflicted: promoted source not GC'd


# --------------------------------------------------------------------------- #
# Cycle D — Procedural learning (Phase 5)
# --------------------------------------------------------------------------- #
def _proc_settings(**over):
    base = dict(procedural_tier1_window_days=2, procedural_tier1_min_samples=2,
                procedural_tier2_window_days=3, procedural_tier2_min_samples=2)
    base.update(over)
    return CognitiveMemorySettings(**base)


def _proc_reader():
    # One recent episodic so observe() has a behavioural corpus; no semantic so drift is inert.
    return _FakeReader(items=[_episodic("r1", age_days=0, text="please use bullet points")])


def _proposer_for(rules):
    async def _p(_observations):
        return {"rules": list(rules), "tokens": 7}
    return _p


@pytest.mark.asyncio
async def test_procedural_matures_to_proposed_over_two_days():
    proc = _FakeProcedural()
    proposer = _proposer_for([{"rule_key": "format.bullets", "rule_text": "Use bullet lists", "samples": 1}])
    settings = _proc_settings()

    # Day 1: observed once → stays observing (window not met).
    day1 = _make_task(_proc_reader(), proposer=proposer, procedural=proc,
                      settings=settings, clock=lambda: NOW)
    await day1.tick(TickContext(user_id="u1"))
    assert proc.rows["format.bullets"]["status"] == "observing"

    # Day 2: second distinct day, samples now 2 ≥ min → matures to proposed.
    day2 = _make_task(_proc_reader(), proposer=proposer, procedural=proc,
                      settings=settings, clock=lambda: NOW + timedelta(days=1))
    await day2.tick(TickContext(user_id="u1"))
    assert proc.rows["format.bullets"]["status"] == "proposed"
    assert proc.rows["format.bullets"]["evidence"]["observation_days"] == ["2026-06-29", "2026-06-30"]
    # A proposed rule is NOT active → never reaches the prompt until approved.
    assert proc.list_active("u1") == []


@pytest.mark.asyncio
async def test_procedural_tier3_is_never_written():
    proc = _FakeProcedural()
    audit = _FakeAudit()
    proposer = _proposer_for([{"rule_key": "logic.always_cite", "rule_text": "Always cite sources", "samples": 5}])

    task = _make_task(_proc_reader(), audit=audit, proposer=proposer, procedural=proc,
                      settings=_proc_settings())
    await task.tick(TickContext(user_id="u1"))

    assert proc.rows == {}  # Tier-3 locked: no rule row at all
    suggestions = [r for r in audit.records if r["action"] == "tier3_suggestion"]
    assert len(suggestions) == 1 and suggestions[0]["target_id"] == "logic.always_cite"


@pytest.mark.asyncio
async def test_procedural_unknown_namespace_ignored():
    proc = _FakeProcedural()
    proposer = _proposer_for([{"rule_key": "random.thing", "rule_text": "x", "samples": 9}])
    task = _make_task(_proc_reader(), proposer=proposer, procedural=proc, settings=_proc_settings())
    await task.tick(TickContext(user_id="u1"))
    assert proc.rows == {}  # unknown rule_key namespace → dropped


@pytest.mark.asyncio
async def test_procedural_offline_defers():
    proc = _FakeProcedural()
    run_store = _FakeRunStore()
    proposer = _proposer_for([{"rule_key": "format.bullets", "rule_text": "Use bullets", "samples": 5}])
    task = _make_task(_proc_reader(), run_store=run_store, proposer=proposer, procedural=proc,
                      settings=_proc_settings(), health=_never_reachable)

    res = await task.tick(TickContext(user_id="u1"))

    assert res["status"] == "partial"
    assert proc.rows == {}  # nothing written when provider is offline
    assert "procedural" in run_store.runs[-1]["detail"]["deferred"]


@pytest.mark.asyncio
async def test_procedural_tier2_needs_longer_window():
    # style.* is Tier 2 (window 3 days here); two days is not yet enough.
    proc = _FakeProcedural()
    proposer = _proposer_for([{"rule_key": "style.concise", "rule_text": "Be concise", "samples": 3}])
    settings = _proc_settings()  # tier2 window = 3

    for day in (0, 1):
        task = _make_task(_proc_reader(), proposer=proposer, procedural=proc,
                          settings=settings, clock=lambda d=day: NOW + timedelta(days=d))
        await task.tick(TickContext(user_id="u1"))

    assert proc.rows["style.concise"]["status"] == "observing"  # 2 of 3 days → not yet proposed

