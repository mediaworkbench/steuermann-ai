import asyncio

import pytest

from universal_agentic_framework.orchestration.graph_builder import build_graph
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.memory import InMemoryMemoryManager
from types import SimpleNamespace


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Out:
            content = f"LLM: {prompt}"

        return _Out()


@pytest.fixture(autouse=True)
def patch_factories(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS_OLLAMA_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_PROVIDERS_LMSTUDIO_API_BASE", "http://localhost:1234/v1")

    # Patch LLMFactory to return fake model (both get_model and get_router_model)
    def _fake_get_model(self, language: str, prefer_local: bool = True):
        return _FakeChatModel()

    def _fake_get_router_model(self, language: str, preferred_model=None):
        return _FakeChatModel()

    monkeypatch.setattr(LLMFactory, "get_model", _fake_get_model)
    monkeypatch.setattr(LLMFactory, "get_router_model", _fake_get_router_model)
    monkeypatch.setattr(LLMFactory, "create_auxiliary_llm", lambda self: _FakeChatModel())

    # node_summarize calls get_auxiliary_model() directly (not via LLMFactory)
    monkeypatch.setattr(
        "universal_agentic_framework.orchestration.graph_builder.get_auxiliary_model",
        lambda config, language="en": (_FakeChatModel(), "fake-provider", "fake-model"),
    )

    # Patch memory backend to in-memory
    def _fake_build_backend(config, client=None, embedder=None):
        return InMemoryMemoryManager()

    monkeypatch.setattr("universal_agentic_framework.orchestration.graph_builder.build_memory_backend", _fake_build_backend)


def test_langgraph_pipeline_runs_and_updates_memory():
    graph = build_graph()

    inputs = {
        "messages": [{"role": "user", "content": "hello memory"}],
        "user_id": "u1",
        "language": "en",
    }

    # The graph's performance nodes are native async coroutines, so it must be driven
    # via the async API (matching production's GRAPH.ainvoke in server.py).
    result = asyncio.run(graph.ainvoke(inputs))

    # Assistant message appended
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"].startswith("LLM: ")

    # Ensure tokens tracked
    assert result.get("tokens_used", 0) > 0

    # Summary should be produced and stored
    assert result.get("summary_text", "").startswith("LLM: ")


