"""Tests for Phase 5 Step 14: memory retrieval quality feedback loop instrumentation."""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import memories as memories_module
from backend.routers import metrics as metrics_module
from backend.routers.memories import _memory_was_recently_retrieved, router as memories_router
from backend.routers.metrics import router as metrics_router
from universal_agentic_framework.monitoring.metrics import (
    _rating_bucket,
    track_memory_rated_after_retrieval,
    track_memory_retrieval_signal,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


class FakeMemoryBackend:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {
            "mem-1": {
                "point_id": "p-1",
                "payload": {
                    "user_id": "u1",
                    "text": "memory one",
                    "metadata": {
                        "memory_id": "mem-1",
                        "user_rating": 4,
                        "importance_score": 0.7,
                        "is_related": False,
                    },
                },
            },
            "mem-2": {
                "point_id": "p-2",
                "payload": {
                    "user_id": "u1",
                    "text": "memory two",
                    "metadata": {
                        "memory_id": "mem-2",
                        "user_rating": None,
                        "importance_score": 0.3,
                        "is_related": False,
                    },
                },
            },
        }

    def load(self, user_id, query=None, top_k=5, include_related=False, session_id=None):
        from universal_agentic_framework.memory.backend import MemoryRecord

        return [
            MemoryRecord(
                user_id=user_id,
                text=str((r.get("payload") or {}).get("text") or ""),
                metadata=dict((r.get("payload") or {}).get("metadata") or {}),
            )
            for r in self.records.values()
            if (r.get("payload") or {}).get("user_id") == user_id
        ][:top_k]

    def find_memory_point(self, memory_id: str):
        return self.records.get(memory_id)

    def set_memory_user_rating(self, *, point_id, metadata, rating):
        if metadata is not None:
            metadata["user_rating"] = rating

    def delete_memory(self, *, memory_id, user_id):
        self.records.pop(memory_id, None)


class _FakePromClient:
    async def query(self, promql: str):
        if "langgraph_memory_retrieval_signal_total" in promql:
            if 'rated="yes"' in promql:
                return [{"metric": {}, "value": [0, "10"]}]
            if 'rated="no"' in promql:
                return [{"metric": {}, "value": [0, "5"]}]
            if "rating_bucket" in promql:
                return [
                    {"metric": {"rating_bucket": "high"}, "value": [0, "8"]},
                    {"metric": {"rating_bucket": "mid"}, "value": [0, "2"]},
                    {"metric": {"rating_bucket": "none"}, "value": [0, "5"]},
                ]
            # total
            return [{"metric": {}, "value": [0, "15"]}]
        if "langgraph_memory_rated_after_retrieval_total" in promql:
            return [{"metric": {}, "value": [0, "3"]}]
        return []

    async def query_range(self, *args, **kwargs):
        return []


@pytest.fixture()
def rate_client(monkeypatch):
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    fake_backend = FakeMemoryBackend()
    monkeypatch.setattr(memories_module, "load_core_config", lambda: object())
    monkeypatch.setattr(memories_module, "build_memory_backend", lambda cfg: fake_backend)

    app = FastAPI()
    app.include_router(memories_router)
    return TestClient(app), fake_backend


# ---------------------------------------------------------------------------
# Unit tests: _rating_bucket helper
# ---------------------------------------------------------------------------


def test_rating_bucket_none():
    assert _rating_bucket(None) == "none"


def test_rating_bucket_low():
    assert _rating_bucket(1) == "low"
    assert _rating_bucket(2) == "low"


def test_rating_bucket_mid():
    assert _rating_bucket(3) == "mid"


def test_rating_bucket_high():
    assert _rating_bucket(4) == "high"
    assert _rating_bucket(5) == "high"


def test_rating_bucket_invalid_string():
    assert _rating_bucket("bad") == "none"


# ---------------------------------------------------------------------------
# Unit tests: track_memory_retrieval_signal
# ---------------------------------------------------------------------------


def test_track_memory_retrieval_signal_with_rating():
    """Calling the tracker with a non-None rating should not raise."""
    track_memory_retrieval_signal("starter", 4)


def test_track_memory_retrieval_signal_without_rating():
    """Calling the tracker with None rating should not raise."""
    track_memory_retrieval_signal("starter", None)


# ---------------------------------------------------------------------------
# Unit tests: track_memory_rated_after_retrieval
# ---------------------------------------------------------------------------


def test_track_memory_rated_after_retrieval_does_not_raise():
    track_memory_rated_after_retrieval("starter")


# ---------------------------------------------------------------------------
# Unit tests: _memory_was_recently_retrieved helper
# ---------------------------------------------------------------------------


def _make_request_with_conversation(memories_used_ids: list[str]) -> Any:
    """Build a mock FastAPI Request whose app.state.conversation_store has
    a conversation containing the given memory IDs in messages_used."""
    mock_request = MagicMock()
    conv_store = MagicMock()
    mock_request.app.state.conversation_store = conv_store
    conv_store.list_conversations.return_value = [{"id": "conv-1"}]
    conv_store.get_messages.return_value = [
        {
            "role": "assistant",
            "metadata": {
                "memories_used": [{"memory_id": mid} for mid in memories_used_ids]
            },
        }
    ]
    return mock_request


def test_memory_was_recently_retrieved_found():
    request = _make_request_with_conversation(["mem-abc", "mem-def"])
    assert _memory_was_recently_retrieved(request, "mem-abc", "u1") is True


def test_memory_was_recently_retrieved_not_found():
    request = _make_request_with_conversation(["mem-xyz"])
    assert _memory_was_recently_retrieved(request, "mem-abc", "u1") is False


def test_memory_was_recently_retrieved_no_conversation_store():
    mock_request = MagicMock()
    mock_request.app.state.conversation_store = None
    assert _memory_was_recently_retrieved(mock_request, "mem-abc", "u1") is False


def test_memory_was_recently_retrieved_swallows_exceptions():
    mock_request = MagicMock()
    mock_request.app.state.conversation_store = MagicMock(
        list_conversations=MagicMock(side_effect=RuntimeError("db down"))
    )
    # Should not raise, returns False
    assert _memory_was_recently_retrieved(mock_request, "mem-abc", "u1") is False


# ---------------------------------------------------------------------------
# Integration-style: rate endpoint fires feedback signal when retrieved
# ---------------------------------------------------------------------------


def test_rate_memory_fires_feedback_signal_when_retrieved(rate_client, monkeypatch):
    test_client, _ = rate_client

    fired: list[str] = []

    def _fake_track(fork_name: str) -> None:
        fired.append(fork_name)

    monkeypatch.setattr(memories_module, "track_memory_rated_after_retrieval", _fake_track)
    monkeypatch.setattr(
        memories_module,
        "_memory_was_recently_retrieved",
        lambda req, memory_id, user_id: True,
    )

    response = test_client.post("/api/memories/mem-1/rate", json={"rating": 5})

    assert response.status_code == 200
    assert len(fired) == 1


def test_rate_memory_no_feedback_signal_when_not_retrieved(rate_client, monkeypatch):
    test_client, _ = rate_client

    fired: list[str] = []

    def _fake_track(fork_name: str) -> None:
        fired.append(fork_name)

    monkeypatch.setattr(memories_module, "track_memory_rated_after_retrieval", _fake_track)
    monkeypatch.setattr(
        memories_module,
        "_memory_was_recently_retrieved",
        lambda req, memory_id, user_id: False,
    )

    response = test_client.post("/api/memories/mem-1/rate", json={"rating": 5})

    assert response.status_code == 200
    assert len(fired) == 0


# ---------------------------------------------------------------------------
# Analytics endpoint: /api/analytics/memory-retrieval-quality
# ---------------------------------------------------------------------------


def test_memory_retrieval_quality_endpoint_shape(monkeypatch):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(metrics_module, "PrometheusClient", _FakePromClient)

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    response = client.get("/api/analytics/memory-retrieval-quality")

    assert response.status_code == 200
    body = response.json()

    expected_keys = {
        "retrieval_signals_total",
        "retrieved_with_prior_rating",
        "retrieved_without_prior_rating",
        "prior_rating_coverage",
        "rating_bucket_distribution",
        "rated_after_retrieval_total",
        "feedback_coverage",
        "timestamp",
    }
    assert set(body.keys()) == expected_keys


def test_memory_retrieval_quality_feedback_coverage(monkeypatch):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(metrics_module, "PrometheusClient", _FakePromClient)

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    response = client.get("/api/analytics/memory-retrieval-quality")
    body = response.json()

    # rated_after=3, total=15 → coverage = 3/15 = 0.2
    assert body["rated_after_retrieval_total"] == 3.0
    assert body["retrieval_signals_total"] == 15.0
    assert abs(body["feedback_coverage"] - 0.2) < 1e-4


def test_memory_retrieval_quality_bucket_distribution(monkeypatch):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(metrics_module, "PrometheusClient", _FakePromClient)

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    response = client.get("/api/analytics/memory-retrieval-quality")
    body = response.json()

    assert body["rating_bucket_distribution"]["high"] == 8.0
    assert body["rating_bucket_distribution"]["mid"] == 2.0
    assert body["rating_bucket_distribution"]["none"] == 5.0
