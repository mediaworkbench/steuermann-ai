"""Unit tests for the admin RAG explorer router (backend/routers/rag_search.py).

External dependencies (config, embedding provider, Qdrant) are monkeypatched in the
router's own namespace, so these tests run without live services (not @integration).
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.rag_search as rag_module


# ── Fakes ───────────────────────────────────────────────────────────────────

class _FakeEmbedder:
    def encode(self, text):  # noqa: D401 - matches RemoteEmbeddingProvider.encode(str)
        return [0.1, 0.2, 0.3]


class _RaisingEmbedder:
    def __init__(self, exc):
        self._exc = exc

    def encode(self, text):
        raise self._exc


def _fake_config(collection_name="framework", threshold=None):
    rag = SimpleNamespace(
        collection_name=collection_name,
        top_k=5,
        pill_score_threshold=threshold,
        with_payload=True,
        with_vectors=False,
        timeout_seconds=30,
    )
    memory = SimpleNamespace(vector_store=SimpleNamespace(host="qdrant", port=6333))
    return SimpleNamespace(rag=rag, memory=memory)


# Three hits straddling the default production cutoff (0.72): 0.91 / 0.75 above, 0.60 below.
_HITS = [
    {
        "id": "a",
        "score": 0.91,
        "payload": {
            "text": "Alpha chunk about taxes",
            "file_path": "/data/a.md",
            "chunk_index": 0,
            "chunk_count": 3,
            "detected_language": "en",
            "language_confidence": 0.99,
        },
    },
    {
        "id": "b",
        "score": 0.75,
        "payload": {"text": "Beta chunk", "file_path": "/data/sub/b.md", "chunk_index": 1, "chunk_count": 2},
    },
    {
        "id": "c",
        "score": 0.60,
        "payload": {"text": "Gamma chunk", "file_name": "c.md", "chunk_index": 0},
    },
]


def _client(monkeypatch, *, config=None, search=None, embedder=None):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(rag_module, "load_core_config", lambda: config or _fake_config())
    monkeypatch.setattr(
        rag_module,
        "get_routing_embedding_provider",
        lambda cfg: (embedder or _FakeEmbedder(), "model"),
    )
    if search is not None:
        monkeypatch.setattr(rag_module, "search_qdrant", search)
    app = FastAPI()
    app.include_router(rag_module.router)
    return TestClient(app)


def _recording_search(hits):
    calls: list[dict] = []

    def _search(qdrant_url, collection_name, vector, top_k, with_payload, with_vector,
                score_threshold, timeout_seconds, label):
        calls.append({"collection": collection_name, "threshold": score_threshold, "label": label})
        return list(hits)

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


# ── Tests ───────────────────────────────────────────────────────────────────

def test_returns_all_hits_with_cutoff_flags_sorted(monkeypatch):
    client = _client(monkeypatch, search=_recording_search(_HITS))
    resp = client.get("/api/rag/search", params={"q": "taxes"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)  # sorted desc
    flags = {i["file_name"]: i["above_cutoff"] for i in body["items"]}
    assert flags == {"a.md": True, "b.md": True, "c.md": False}
    # file_name derived from file_path basename; legacy file_name field also handled.
    assert body["items"][0]["file_name"] == "a.md"
    assert body["production_threshold"] == 0.72


def test_passes_no_threshold_to_qdrant_by_default(monkeypatch):
    search = _recording_search(_HITS)
    client = _client(monkeypatch, search=search)
    client.get("/api/rag/search", params={"q": "taxes"})
    assert search.calls[0]["threshold"] is None  # shows everything by default


def test_score_threshold_param_is_forwarded(monkeypatch):
    search = _recording_search(_HITS)
    client = _client(monkeypatch, search=search)
    client.get("/api/rag/search", params={"q": "taxes", "score_threshold": "0.5"})
    assert search.calls[0]["threshold"] == 0.5


def test_collection_fallback_to_framework_when_unset(monkeypatch):
    search = _recording_search(_HITS)
    client = _client(monkeypatch, config=_fake_config(collection_name=None), search=search)
    client.get("/api/rag/search", params={"q": "taxes"})
    assert search.calls[0]["collection"] == "framework"


def test_collection_override_param_wins(monkeypatch):
    search = _recording_search(_HITS)
    client = _client(monkeypatch, search=search)
    client.get("/api/rag/search", params={"q": "taxes", "collection": "custom"})
    assert search.calls[0]["collection"] == "custom"


def test_empty_query_is_rejected(monkeypatch):
    client = _client(monkeypatch, search=_recording_search(_HITS))
    assert client.get("/api/rag/search", params={"q": ""}).status_code == 422
    assert client.get("/api/rag/search").status_code == 422


def test_missing_collection_maps_to_404(monkeypatch):
    def _raise(*args, **kwargs):
        request = httpx.Request("POST", "http://qdrant:6333/collections/x/points/search")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    client = _client(monkeypatch, search=_raise)
    resp = client.get("/api/rag/search", params={"q": "taxes"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_other_qdrant_error_maps_to_502(monkeypatch):
    def _raise(*args, **kwargs):
        request = httpx.Request("POST", "http://qdrant:6333/collections/x/points/search")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    client = _client(monkeypatch, search=_raise)
    assert client.get("/api/rag/search", params={"q": "taxes"}).status_code == 502


def test_qdrant_unreachable_maps_to_502(monkeypatch):
    # ConnectError is a RequestError but NOT a TimeoutException/HTTPStatusError —
    # without an explicit handler this would surface as an unhandled 500.
    def _raise(*args, **kwargs):
        raise httpx.ConnectError(
            "connection refused",
            request=httpx.Request("POST", "http://qdrant:6333/collections/x/points/search"),
        )

    client = _client(monkeypatch, search=_raise)
    resp = client.get("/api/rag/search", params={"q": "taxes"})
    assert resp.status_code == 502
    assert "reach" in resp.json()["detail"].lower()


def test_embedding_failure_maps_to_502(monkeypatch):
    from universal_agentic_framework.embeddings import EmbeddingProviderUnavailableError

    embedder = _RaisingEmbedder(EmbeddingProviderUnavailableError("provider down"))
    client = _client(monkeypatch, search=_recording_search(_HITS), embedder=embedder)
    resp = client.get("/api/rag/search", params={"q": "taxes"})
    assert resp.status_code == 502
    assert "embedding" in resp.json()["detail"].lower()


def test_collections_lists_names_with_counts(monkeypatch):
    def _fake_get(url, timeout=None):
        if url.endswith("/collections"):
            payload = {"result": {"collections": [{"name": "framework"}, {"name": "other"}]}}
        else:  # /collections/{name}
            payload = {"result": {"points_count": 42}}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(rag_module.httpx, "get", _fake_get)
    client = _client(monkeypatch, search=_recording_search(_HITS))
    resp = client.get("/api/rag/collections")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_collection"] == "framework"
    assert {c["name"]: c["points_count"] for c in body["collections"]} == {
        "framework": 42,
        "other": 42,
    }
