"""Phase 0 tests: cognitive-memory metadata foundation + tier tagging.

Covers the concept §6 payload contract on Mem0MemoryBackend:
- new upserts carry the four contract fields (user_id, cognitive_tier,
  confidence, last_accessed),
- legacy records (written before tagging) read-normalize without crashing,
- round-trip persistence of ``cognitive_tier`` (Mem0 unknown-key risk),
- blended semantic/episodic split is correct and de-duped,
- ``update_metadata`` preserves the memory text,
- ``get_all_for_dreaming`` returns the user's normalized memories.

Uses a minimal in-process fake Mem0 client — no Qdrant/LLM required.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from universal_agentic_framework.memory.mem0_backend import Mem0MemoryBackend


class _FakeMemory:
    """Minimal Mem0 Memory API surface — no network calls."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        text: Any,
        *,
        user_id: str,
        metadata: Optional[dict] = None,
        infer: bool = True,
    ) -> dict:
        if isinstance(text, list):
            text_value = "\n".join(
                str(item.get("content") or "") for item in text if isinstance(item, dict)
            ).strip()
        else:
            text_value = str(text)
        memory_id = uuid.uuid4().hex[:12]
        self._store[memory_id] = {
            "id": memory_id,
            "memory": text_value,
            "user_id": user_id,
            "metadata": dict(metadata or {}),
            "score": 1.0,
        }
        return {"results": [{"id": memory_id}]}

    def search(self, query: str, *, filters=None, top_k=None, **kwargs) -> dict:
        user_id = (filters or {}).get("user_id")
        limit = top_k if top_k is not None else 10
        results = [v for v in self._store.values() if v["user_id"] == user_id][:limit]
        for r in results:
            r["score"] = 1.0
        return {"results": results}

    def get_all(self, *, filters=None, limit=None, **kwargs) -> dict:
        user_id = (filters or {}).get("user_id")
        effective = limit if limit is not None else 100
        results = [v for v in self._store.values() if v["user_id"] == user_id][:effective]
        return {"results": results}

    def get(self, memory_id: str) -> Optional[dict]:
        return self._store.get(memory_id)

    def update(self, *, memory_id: str, data: str = "", **kwargs: Any) -> None:
        if memory_id in self._store:
            self._store[memory_id]["memory"] = data
            metadata = kwargs.get("metadata")
            if isinstance(metadata, dict):
                existing = dict(self._store[memory_id].get("metadata") or {})
                self._store[memory_id]["metadata"] = {**existing, **metadata}

    def delete(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)


_COMMON_KWARGS = dict(
    host="localhost",
    port=6333,
    collection_prefix="test",
    embedding_model="text-embedding-ada-002",
    dimension=768,
    embedding_remote_endpoint=None,
    llm_model="gpt-4o-mini",
    llm_api_base=None,
    llm_temperature=0.0,
    llm_max_tokens=None,
    llm_api_key="test-key",
)


def _make_backend(**overrides: Any) -> Mem0MemoryBackend:
    kwargs = {**_COMMON_KWARGS, **overrides, "client": _FakeMemory()}
    return Mem0MemoryBackend(**kwargs)


_CONTRACT_FIELDS = {"user_id", "cognitive_tier", "confidence", "last_accessed"}


# ---------------------------------------------------------------------------
# Write path: contract fields present + defaulted
# ---------------------------------------------------------------------------

def test_upsert_carries_four_contract_fields():
    backend = _make_backend()
    rec = backend.upsert("u1", "episodic by default")

    for field in _CONTRACT_FIELDS:
        assert field in rec.metadata, f"missing contract field: {field}"
    assert rec.metadata["user_id"] == "u1"
    assert rec.metadata["cognitive_tier"] == "episodic"
    assert rec.metadata["confidence"] == 1.0
    assert rec.metadata["last_accessed"] == rec.metadata["created_at"]


def test_upsert_does_not_override_explicit_tier_and_confidence():
    """Engine-written semantic memories pass tier/confidence explicitly."""
    backend = _make_backend()
    rec = backend.upsert(
        "u1",
        "synthesized fact",
        metadata={"cognitive_tier": "semantic", "confidence": 0.6},
    )
    assert rec.metadata["cognitive_tier"] == "semantic"
    assert rec.metadata["confidence"] == 0.6


# ---------------------------------------------------------------------------
# Read path: legacy records normalize
# ---------------------------------------------------------------------------

def test_legacy_record_without_contract_fields_normalizes():
    backend = _make_backend()
    # Simulate a pre-tagging record: no cognitive_tier/confidence/last_accessed.
    legacy_id = uuid.uuid4().hex[:12]
    backend._memory._store[legacy_id] = {
        "id": legacy_id,
        "memory": "legacy memory",
        "user_id": "u1",
        "metadata": {"user_id": "u1"},
        "score": 1.0,
    }

    records = backend.load("u1", top_k=5)
    assert len(records) == 1
    meta = records[0].metadata
    assert meta["cognitive_tier"] == "episodic"
    assert meta["confidence"] == 1.0
    assert meta["last_accessed"]  # defaulted, not missing


# ---------------------------------------------------------------------------
# Round-trip persistence of cognitive_tier (checkpoint #7 style)
# ---------------------------------------------------------------------------

def test_cognitive_tier_round_trips_through_persistence():
    backend = _make_backend()
    rec = backend.upsert(
        "u1", "semantic fact", metadata={"cognitive_tier": "semantic", "confidence": 0.6}
    )
    mid = rec.metadata["memory_id"]

    point = backend.find_memory_point(mid)
    assert point is not None
    assert point["payload"]["metadata"]["cognitive_tier"] == "semantic"
    assert point["payload"]["metadata"]["confidence"] == 0.6


# ---------------------------------------------------------------------------
# Blended retrieval split
# ---------------------------------------------------------------------------

def test_load_blended_prepends_semantic_before_episodic():
    backend = _make_backend()
    backend.upsert("u1", "episodic one")
    backend.upsert("u1", "episodic two")
    backend.upsert(
        "u1", "semantic insight", metadata={"cognitive_tier": "semantic", "confidence": 0.8}
    )

    records = backend.load_blended("u1", query="anything", semantic_top_k=3, episodic_top_k=5)
    tiers = [r.metadata.get("cognitive_tier") for r in records]

    assert "semantic" in tiers
    # Semantic memories are prepended before any episodic memory.
    first_episodic = tiers.index("episodic")
    assert all(t == "semantic" for t in tiers[:first_episodic])


def test_load_blended_respects_per_tier_caps_and_dedupes():
    backend = _make_backend()
    for i in range(4):
        backend.upsert("u1", f"episodic {i}")
    for i in range(3):
        backend.upsert(
            "u1", f"semantic {i}", metadata={"cognitive_tier": "semantic", "confidence": 0.7}
        )

    records = backend.load_blended("u1", query="x", semantic_top_k=2, episodic_top_k=2)
    tiers = [r.metadata.get("cognitive_tier") for r in records]
    assert tiers.count("semantic") <= 2
    assert tiers.count("episodic") <= 2

    ids = [r.metadata.get("memory_id") for r in records]
    assert len(ids) == len(set(ids)), "blended results must be de-duped by memory_id"


def test_load_blended_all_episodic_when_no_semantic():
    backend = _make_backend()
    backend.upsert("u1", "only episodic a")
    backend.upsert("u1", "only episodic b")

    records = backend.load_blended("u1", query="x", semantic_top_k=3, episodic_top_k=5)
    assert records
    assert all(r.metadata.get("cognitive_tier") == "episodic" for r in records)


# ---------------------------------------------------------------------------
# get_all_for_dreaming
# ---------------------------------------------------------------------------

def test_get_all_for_dreaming_returns_user_scoped_normalized_items():
    backend = _make_backend()
    backend.upsert("u1", "mine one")
    backend.upsert("u1", "mine two")
    backend.upsert("u2", "not mine")

    items = backend.get_all_for_dreaming("u1", limit=100)
    assert len(items) == 2
    for item in items:
        assert _CONTRACT_FIELDS.issubset(item["metadata"].keys())
        assert item["metadata"]["user_id"] == "u1"


# ---------------------------------------------------------------------------
# update_metadata text preservation
# ---------------------------------------------------------------------------

def test_update_metadata_preserves_text():
    backend = _make_backend()
    rec = backend.upsert("u1", "do not wipe this text")
    mid = rec.metadata["memory_id"]

    ok = backend.update_metadata(mid, {"confidence": 0.4, "cognitive_tier": "semantic"})
    assert ok is True

    point = backend.find_memory_point(mid)
    assert point is not None
    assert point["payload"]["text"] == "do not wipe this text"
    assert point["payload"]["metadata"]["confidence"] == 0.4
    assert point["payload"]["metadata"]["cognitive_tier"] == "semantic"


def test_update_metadata_missing_memory_returns_false():
    backend = _make_backend()
    assert backend.update_metadata("nonexistent-id", {"confidence": 0.1}) is False


def test_dreaming_reads_degrade_when_collection_missing():
    # A fresh deployment / a user with no memories => Qdrant collection doesn't
    # exist yet. The engine's batch reads must no-op (return []), not raise, so the
    # dreaming tick doesn't error every beat.
    backend = _make_backend()

    def _missing(*_a, **_k):
        raise Exception("Unexpected Response: 404 (Not Found) Collection `test_memory` doesn't exist!")

    backend._memory.get_all = _missing
    backend._memory.search = _missing

    assert backend.get_all_for_dreaming("u1") == []
    assert backend.find_nearest_semantic("u1", "anything") == []
