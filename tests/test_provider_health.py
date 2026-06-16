"""Tests for provider-offline resilience:

- ``GET /api/llm/health`` live reachability aggregation (online/degraded/offline).
- The non-fatal LangGraph startup embedding probe (must never raise).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.settings import router


# --------------------------------------------------------------------------- #
# GET /api/llm/health
# --------------------------------------------------------------------------- #

def _make_core(chat_base, embed_base, configured_roles=("chat",)):
    """Fake core config whose llm helpers mirror the real resolution surface.

    Optional roles that aren't configured raise (like the real
    ``get_role_provider``), so the endpoint must skip them gracefully.
    """

    class _LLM:
        def get_role_provider(self, role):
            if role in configured_roles:
                return SimpleNamespace(api_base=chat_base if role == "chat" else embed_base)
            raise ValueError(f"llm.roles.{role} not configured")

        def get_embedding_remote_endpoint(self):
            return embed_base

    return SimpleNamespace(llm=_LLM())


def _fake_client_factory(reachable: dict[str, bool]):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            base = url[: -len("/models")] if url.endswith("/models") else url
            if reachable.get(base, False):
                return SimpleNamespace(status_code=200)
            raise httpx.ConnectError("connection refused")

    return _FakeAsyncClient


def _client(monkeypatch, core, reachable):
    monkeypatch.setattr("backend.routers.settings.load_core_config", lambda: core)
    monkeypatch.setattr(
        "backend.routers.settings.httpx.AsyncClient", _fake_client_factory(reachable)
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_online_when_all_reachable(monkeypatch):
    core = _make_core("http://chat/v1", "http://embed/v1")
    client = _client(monkeypatch, core, {"http://chat/v1": True, "http://embed/v1": True})

    body = client.get("/api/llm/health").json()

    assert body["status"] == "online"
    # chat + embedding endpoints deduped to two distinct bases, both reachable.
    assert {p["api_base"] for p in body["providers"]} == {"http://chat/v1", "http://embed/v1"}
    assert all(p["reachable"] for p in body["providers"])


def test_health_offline_when_chat_unreachable(monkeypatch):
    core = _make_core("http://chat/v1", "http://embed/v1")
    client = _client(monkeypatch, core, {"http://chat/v1": False, "http://embed/v1": True})

    body = client.get("/api/llm/health").json()

    assert body["status"] == "offline"


def test_health_degraded_when_only_secondary_unreachable(monkeypatch):
    core = _make_core("http://chat/v1", "http://embed/v1")
    client = _client(monkeypatch, core, {"http://chat/v1": True, "http://embed/v1": False})

    body = client.get("/api/llm/health").json()

    assert body["status"] == "degraded"


def test_health_skips_unconfigured_optional_roles(monkeypatch):
    # Only chat + embedding configured; vision/auxiliary raise and must be skipped
    # without leaking a 500.
    core = _make_core("http://chat/v1", "http://embed/v1")
    resp = _client(monkeypatch, core, {"http://chat/v1": True, "http://embed/v1": True}).get(
        "/api/llm/health"
    )

    assert resp.status_code == 200
    roles = {r for p in resp.json()["providers"] for r in p["roles"]}
    assert "vision" not in roles and "auxiliary" not in roles


def test_health_offline_when_no_api_base(monkeypatch):
    class _LLM:
        def get_role_provider(self, role):
            raise ValueError("not configured")

        def get_embedding_remote_endpoint(self):
            return None

    core = SimpleNamespace(llm=_LLM())
    client = _client(monkeypatch, core, {})

    body = client.get("/api/llm/health").json()

    assert body["status"] == "offline"
    assert body["providers"] == []
    assert body["error"] == "no_api_base_configured"


# --------------------------------------------------------------------------- #
# Non-fatal startup embedding probe
# --------------------------------------------------------------------------- #

@pytest.fixture
def server_module(monkeypatch):
    """Import server.py with graph construction and the import-time probe stubbed.

    server.py builds the graph at import time and runs a best-effort embedding
    probe; both are neutralised here so importing is fast and offline-safe.
    """
    import universal_agentic_framework.orchestration.graph_builder as gb
    import universal_agentic_framework.orchestration.helpers.embedding_provider as ep

    monkeypatch.setattr(gb, "build_graph", lambda: MagicMock(), raising=True)

    def _boom(_config, *args, **kwargs):
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(ep, "get_routing_embedding_provider", _boom, raising=True)

    server = importlib.import_module("universal_agentic_framework.server")
    importlib.reload(server)
    return server


def test_startup_probe_nonfatal_returns_false_on_failure(server_module):
    # Provider unreachable → returns False, never raises (stack must still boot).
    assert server_module.probe_embedding_provider_nonfatal(server_module.CONFIG) is False


def test_startup_probe_returns_true_on_success(server_module, monkeypatch):
    import universal_agentic_framework.orchestration.helpers.embedding_provider as ep

    fake_embedder = MagicMock()
    monkeypatch.setattr(
        ep,
        "get_routing_embedding_provider",
        lambda cfg, *a, **k: (fake_embedder, "model-x"),
        raising=True,
    )

    assert server_module.probe_embedding_provider_nonfatal(server_module.CONFIG) is True
    fake_embedder.encode.assert_called_once()
