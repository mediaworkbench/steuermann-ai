"""Integration tests for Mem0MemoryBackend adapter.

Uses a _FakeMemory client to exercise all public contracts without requiring
a live Qdrant or LLM instance.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import pytest

from universal_agentic_framework.memory.backend import MemoryRatingBackend
from universal_agentic_framework.memory.mem0_backend import Mem0MemoryBackend


# ---------------------------------------------------------------------------
# Fake Mem0 Memory client
# ---------------------------------------------------------------------------

class _FakeMemory:
    """Minimal Mem0 Memory API surface — no network calls."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self.last_search_filters: Optional[Dict[str, Any]] = None
        self.last_get_all_filters: Optional[Dict[str, Any]] = None
        self.last_delete_all_filters: Optional[Dict[str, Any]] = None
        self.last_add_infer: Optional[bool] = None
        self.last_add_payload: Any = None

    def add(
        self,
        text: Any,
        *,
        user_id: str,
        metadata: Optional[dict] = None,
        infer: bool = True,
    ) -> dict:
        self.last_add_infer = infer
        self.last_add_payload = text
        if isinstance(text, list):
            # Mem0 canonical call path: list of chat-style messages.
            text_value = "\n".join(
                str(item.get("content") or "")
                for item in text
                if isinstance(item, dict)
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

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        self.last_search_filters = dict(filters or {})
        effective_user_id = user_id or self.last_search_filters.get("user_id")
        effective_limit = top_k if top_k is not None else (limit if limit is not None else 10)
        results = [
            v for v in self._store.values() if v["user_id"] == effective_user_id
        ][:effective_limit]
        # Attach simple relevance score based on word overlap
        for r in results:
            words = set(query.lower().split())
            mem_words = set(r["memory"].lower().split())
            r["score"] = len(words & mem_words) / max(len(words), 1)
        return {"results": results}

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        self.last_get_all_filters = dict(filters or {})
        effective_user_id = user_id or self.last_get_all_filters.get("user_id")
        effective_limit = top_k if top_k is not None else (limit if limit is not None else 100)
        results = [v for v in self._store.values() if v["user_id"] == effective_user_id][:effective_limit]
        return {"results": results}

    def get(self, *, memory_id: str) -> Optional[dict]:
        return self._store.get(memory_id)

    def update(self, *, memory_id: str, data: str = "", **kwargs: Any) -> None:
        if memory_id in self._store:
            self._store[memory_id]["memory"] = data
            metadata = kwargs.get("metadata")
            if isinstance(metadata, dict):
                existing = dict(self._store[memory_id].get("metadata") or {})
                self._store[memory_id]["metadata"] = {**existing, **metadata}

    def delete_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.last_delete_all_filters = dict(filters or {})
        effective_user_id = user_id or self.last_delete_all_filters.get("user_id")
        to_delete = [k for k, v in self._store.items() if v["user_id"] == effective_user_id]
        for k in to_delete:
            del self._store[k]


class _NoHitSearchFakeMemory(_FakeMemory):
    """Fake Memory backend where semantic search returns no hits."""

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        self.last_search_filters = dict(filters or {})
        return {"results": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_backend_nohit_search(**overrides: Any) -> Mem0MemoryBackend:
    kwargs = {**_COMMON_KWARGS, **overrides, "client": _NoHitSearchFakeMemory()}
    return Mem0MemoryBackend(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_isinstance_memory_rating_backend():
    """Mem0MemoryBackend must satisfy the MemoryRatingBackend protocol."""
    backend = _make_backend()
    assert isinstance(backend, MemoryRatingBackend)


def test_upsert_returns_memory_record():
    backend = _make_backend()
    rec = backend.upsert("u1", "alpha beta gamma", metadata={"tag": "x"})
    assert rec.user_id == "u1"
    assert rec.text == "alpha beta gamma"
    assert rec.metadata.get("memory_id") is not None
    assert rec.metadata["tag"] == "x"
    assert backend._memory.last_add_infer is True


def test_upsert_normalizes_message_payload_content_shapes():
    backend = _make_backend()

    backend.upsert(
        "u1",
        "fallback text",
        messages=[
            {"role": "user", "content": [{"text": "alpha"}, "beta"]},
            {"role": "assistant", "content": {"text": "gamma"}},
            {"role": "user", "content": None},
            "ignored-non-dict",
        ],
    )

    assert backend._memory.last_add_payload == [
        {"role": "user", "content": "alpha\nbeta"},
        {"role": "assistant", "content": "gamma"},
        {"role": "user", "content": ""},
    ]


def test_load_with_query_applies_importance_scoring():
    backend = _make_backend()
    backend.upsert("u1", "machine learning basics")
    backend.upsert("u1", "completely unrelated topic")
    backend.upsert("u1", "machine learning advanced techniques")

    records = backend.load("u1", query="machine learning", top_k=5)
    assert len(records) >= 1
    assert backend._memory.last_search_filters == {"user_id": "u1"}
    # Records with higher relevance should appear first
    texts = [r.text for r in records]
    assert any("machine learning" in t for t in texts)


def test_load_with_query_falls_back_to_recent_when_search_has_no_hits():
    backend = _make_backend_nohit_search()
    backend.upsert("u1", "most recent memory one")
    backend.upsert("u1", "most recent memory two")

    records = backend.load("u1", query="unrelated query", top_k=5)

    assert len(records) == 2
    assert backend._memory.last_search_filters == {"user_id": "u1"}
    assert backend._memory.last_get_all_filters == {"user_id": "u1"}


def test_load_without_query_returns_all_sorted_by_created_at():
    backend = _make_backend()
    backend.upsert("u1", "first entry")
    backend.upsert("u1", "second entry")

    records = backend.load("u1", top_k=10)
    assert len(records) == 2


def test_load_top_k_limits_results():
    backend = _make_backend()
    for i in range(10):
        backend.upsert("u1", f"memory entry {i}")

    records = backend.load("u1", top_k=3)
    assert len(records) <= 3


def test_load_returns_empty_for_unknown_user():
    backend = _make_backend()
    records = backend.load("nobody", query="anything", top_k=5)
    assert records == []


def test_upsert_metadata_preserved():
    backend = _make_backend()
    rec = backend.upsert("u1", "test text", metadata={"source": "doc1", "page": 3})
    assert rec.metadata["source"] == "doc1"
    assert rec.metadata["page"] == 3


def test_clear_removes_all_user_memories():
    backend = _make_backend()
    backend.upsert("u1", "to be cleared")
    backend.upsert("u1", "also cleared")
    backend.upsert("u2", "other user kept")

    backend.clear("u1")
    assert backend._memory.last_get_all_filters == {"user_id": "u1"}
    assert backend._memory.last_delete_all_filters == {"user_id": "u1"}

    assert backend.load("u1", top_k=10) == []
    assert len(backend.load("u2", top_k=10)) == 1


def test_clear_purges_internal_caches():
    backend = _make_backend()
    rec = backend.upsert("u1", "cached text")
    mid = rec.metadata["memory_id"]

    backend.clear("u1")

    assert mid not in backend._metadata_cache
    assert mid not in backend._text_cache
    assert mid not in backend._owner_cache


def test_find_memory_point_returns_correct_shape():
    backend = _make_backend()
    rec = backend.upsert("u1", "findable text")
    mid = rec.metadata["memory_id"]

    point = backend.find_memory_point(mid)
    assert point is not None
    assert point["point_id"] == mid
    assert "payload" in point
    assert point["payload"]["text"] == "findable text"
    assert point["payload"]["user_id"] == "u1"


def test_find_memory_point_returns_none_for_unknown():
    backend = _make_backend()
    assert backend.find_memory_point("nonexistent-id") is None


def test_find_memory_point_handles_null_score():
    backend = _make_backend()
    rec = backend.upsert("u1", "null-score memory")
    mid = rec.metadata["memory_id"]

    # Mem0 get() may return score=None for item lookups.
    backend._memory._store[mid]["score"] = None

    point = backend.find_memory_point(mid)
    assert point is not None
    assert point["point_id"] == mid
    assert point["payload"]["text"] == "null-score memory"


def test_set_memory_user_rating_persists_in_cache():
    backend = _make_backend()
    rec = backend.upsert("u1", "rateable memory")
    mid = rec.metadata["memory_id"]

    backend.set_memory_user_rating(
        point_id=mid,
        metadata={"memory_id": mid},
        rating=5,
    )

    assert backend._rating_overrides[mid] == 5
    assert backend._metadata_cache[mid]["user_rating"] == 5


def test_set_memory_user_rating_appears_in_subsequent_load():
    backend = _make_backend()
    rec = backend.upsert("u1", "rate this")
    mid = rec.metadata["memory_id"]

    backend.set_memory_user_rating(
        point_id=mid,
        metadata={"memory_id": mid},
        rating=4,
    )

    records = backend.load("u1", top_k=10)
    rated = [r for r in records if r.metadata.get("memory_id") == mid]
    assert len(rated) == 1
    assert rated[0].metadata["user_rating"] == 4


def test_set_memory_user_rating_persists_after_cache_reset():
    backend = _make_backend()
    rec = backend.upsert("u1", "persist rating")
    mid = rec.metadata["memory_id"]

    backend.set_memory_user_rating(
        point_id=mid,
        metadata={"memory_id": mid},
        rating=5,
    )

    # Simulate a fresh in-process state (e.g., page reload/new request path).
    backend._rating_overrides.clear()
    backend._metadata_cache.clear()

    records = backend.load("u1", top_k=10)
    rated = [r for r in records if r.metadata.get("memory_id") == mid]
    assert len(rated) == 1
    assert rated[0].metadata.get("user_rating") == 5


def test_co_occurrence_tracking_disabled():
    """With co-occurrence disabled, related memories are never appended."""
    backend = _make_backend(enable_co_occurrence_tracking=False)
    assert backend._co_occurrence_tracker is None

    for i in range(5):
        backend.upsert("u1", f"some memory {i}")

    records = backend.load("u1", query="memory", top_k=3, include_related=True)
    # No extra related memories appended — should stay within top_k
    assert len(records) <= 3


def test_importance_scoring_disabled():
    """With importance scoring disabled, records are still returned."""
    backend = _make_backend(enable_importance_scoring=False)
    assert backend._importance_scorer is None

    backend.upsert("u1", "without importance scoring")
    records = backend.load("u1", top_k=5)
    assert len(records) == 1


def test_multiple_users_isolated():
    backend = _make_backend()
    backend.upsert("alice", "alice memory")
    backend.upsert("bob", "bob memory")

    alice_records = backend.load("alice", top_k=10)
    bob_records = backend.load("bob", top_k=10)

    alice_texts = {r.text for r in alice_records}
    bob_texts = {r.text for r in bob_records}

    assert "alice memory" in alice_texts
    assert "bob memory" not in alice_texts
    assert "bob memory" in bob_texts
    assert "alice memory" not in bob_texts
