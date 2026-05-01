import pytest

from universal_agentic_framework.orchestration.graph_builder import build_graph
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.memory import InMemoryMemoryManager
from universal_agentic_framework.llm.budget import TokenBudgetExceeded
from types import SimpleNamespace


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Out:
            content = f"LLM: {prompt}"

        return _Out()


@pytest.fixture(autouse=True)
def patch_factories(monkeypatch):
    monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:11434")

    # Patch LLMFactory to return fake model (both get_model and get_router_model)
    def _fake_get_model(self, language: str, prefer_local: bool = True):
        return _FakeChatModel()

    def _fake_get_router_model(self, language: str, preferred_model=None):
        return _FakeChatModel()

    monkeypatch.setattr(LLMFactory, "get_model", _fake_get_model)
    monkeypatch.setattr(LLMFactory, "get_router_model", _fake_get_router_model)

    # Patch memory backend to in-memory
    def _fake_build_backend(config, client=None, embedder=None):
        return InMemoryMemoryManager()

    monkeypatch.setattr("universal_agentic_framework.memory.factory.build_memory_backend", _fake_build_backend)


def test_langgraph_pipeline_runs_and_updates_memory():
    graph = build_graph()

    inputs = {
        "messages": [{"role": "user", "content": "hello memory"}],
        "user_id": "u1",
        "language": "en",
    }

    result = graph.invoke(inputs)

    # Assistant message appended
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"].startswith("LLM: ")

    # Ensure tokens tracked
    assert result.get("tokens_used", 0) > 0

    # Summary should be produced and stored
    assert result.get("summary_text", "").startswith("LLM: ")


def test_summarization_budget_enforced(monkeypatch):
    # Force a tiny summarization budget to trigger TokenBudgetExceeded
    tiny_budget_config = SimpleNamespace(
        fork=SimpleNamespace(language="en"),
        tokens=SimpleNamespace(default_budget=100, per_node_budgets={"summarization_node": 1}),
    )

    monkeypatch.setattr(
        "universal_agentic_framework.orchestration.graph_builder.load_core_config",
        lambda: tiny_budget_config,
    )
    # Patch memory backend builder at the graph_builder import site to avoid config.memory access
    monkeypatch.setattr(
        "universal_agentic_framework.orchestration.graph_builder.build_memory_backend",
        lambda *_, **__: InMemoryMemoryManager(),
    )

    graph = build_graph()

    inputs = {
        "messages": [{"role": "user", "content": "hello memory"}],
        "user_id": "u1",
        "language": "en",
    }

    with pytest.raises(TokenBudgetExceeded):
        graph.invoke(inputs)
