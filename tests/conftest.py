import os
import pytest


# Ensure config placeholder substitution succeeds in tests that call load_core_config() directly.
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _disable_checkpointer_for_tests(monkeypatch):
    """Disable checkpointing globally in tests.

    With conditional edges from START, LangGraph enforces thread_id when a checkpointer is
    attached.  Integration tests invoke the graph without a configurable thread_id, and tool
    instances stored in GraphState are not msgpack-serialisable anyway.  Patching
    build_checkpointer to return None keeps tests self-contained.
    """
    try:
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.graph_builder.build_checkpointer",
            lambda *_, **__: None,
        )
    except (ImportError, ModuleNotFoundError):
        pass  # langgraph not available in this environment (e.g. ingestion container)


@pytest.fixture(autouse=True)
def _disable_crews_for_tests(monkeypatch):
    """Disable multi-agent crews during tests to prevent real LLM initialization."""
    try:
        from universal_agentic_framework.orchestration import crew_nodes
    except (ImportError, ModuleNotFoundError):
        return  # orchestration deps not available in this environment (e.g. ingestion container)
    original_load = crew_nodes.load_features_config
    
    def mock_load_features(*args, **kwargs):
        cfg = original_load(*args, **kwargs)
        # Disable multi_agent_crews in the config
        cfg.multi_agent_crews = False
        return cfg
    
    monkeypatch.setattr(crew_nodes, "load_features_config", mock_load_features)
