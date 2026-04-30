import os
import pytest


# Ensure config placeholder substitution succeeds in tests that call load_core_config() directly.
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434")


@pytest.fixture(autouse=True)
def _disable_checkpointer_for_tests(monkeypatch):
    """Disable checkpointing globally in tests.

    With conditional edges from START, LangGraph enforces thread_id when a checkpointer is
    attached.  Integration tests invoke the graph without a configurable thread_id, and tool
    instances stored in GraphState are not msgpack-serialisable anyway.  Patching
    build_checkpointer to return None keeps tests self-contained.
    """
    monkeypatch.setattr(
        "universal_agentic_framework.orchestration.graph_builder.build_checkpointer",
        lambda *_, **__: None,
    )


@pytest.fixture(autouse=True)
def _disable_crews_for_tests(monkeypatch):
    """Disable multi-agent crews during tests to prevent real LLM initialization."""
    # Patch load_features_config in crew_nodes to disable crews
    from universal_agentic_framework.orchestration import crew_nodes
    original_load = crew_nodes.load_features_config
    
    def mock_load_features(*args, **kwargs):
        cfg = original_load(*args, **kwargs)
        # Disable multi_agent_crews in the config
        cfg.multi_agent_crews = False
        return cfg
    
    monkeypatch.setattr(crew_nodes, "load_features_config", mock_load_features)
