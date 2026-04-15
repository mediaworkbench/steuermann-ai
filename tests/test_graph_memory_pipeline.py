from universal_agentic_framework.graph import run_graph_with_memory
from universal_agentic_framework.memory import InMemoryMemoryManager
from universal_agentic_framework.memory.nodes import update_memory_node


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Out:
            content = f"LLM: {prompt}"

        return _Out()


def test_graph_with_memory_pipeline_updates_and_loads():
    backend = InMemoryMemoryManager()
    state = {"user_id": "u1"}

    # Seed a prior memory
    update_memory_node(state, text="question about A prior fact", metadata={"t": 1}, backend=backend)

    res = run_graph_with_memory(
        message="question about A",
        user_id="u1",
        language="en",
        backend_override=backend,
        model_override=_FakeChatModel(),
    )

    assert res["response"].startswith("LLM: ")
    assert res["tokens_used"] > 0
    # Loaded memory should include entries (query 'question about A' matches 'prior fact A')
    assert res["loaded_memory_count"] >= 1

    # Verify that a summary entry was appended
    entries = backend.load(user_id="u1")
    assert any(e.text.startswith("Summary: ") for e in entries)
