from universal_agentic_framework.memory.nodes import load_memory_node, update_memory_node
from universal_agentic_framework.memory import InMemoryMemoryManager


class _FakeBackend(InMemoryMemoryManager):
    pass


def test_load_memory_node_roundtrip():
    backend = _FakeBackend()
    state = {"user_id": "u1", "messages": [{"role": "user", "content": "first"}]}

    update_memory_node(state, text="first memory", metadata={"t": 1}, backend=backend)
    update_memory_node(state, text="second memory", metadata={"t": 2}, backend=backend)

    # Now load with a query that should match only second
    state["messages"] = [{"role": "user", "content": "second"}]
    new_state = load_memory_node(state, backend=backend)

    loaded = new_state.get("loaded_memory", [])
    assert len(loaded) == 1
    assert loaded[0]["text"] == "second memory"


def test_update_memory_node_appends():
    backend = _FakeBackend()
    state = {"user_id": "u2"}

    update_memory_node(state, text="A", backend=backend)
    update_memory_node(state, text="B", backend=backend)

    results = backend.load(user_id="u2")
    assert [r.text for r in results] == ["A", "B"]


def test_load_memory_node_extracts_digest_context():
    backend = _FakeBackend()
    state = {"user_id": "u3", "messages": [{"role": "user", "content": "alpha"}]}

    update_memory_node(
        state,
        text="Summary memory",
        metadata={
            "type": "summary",
            "digest_id": "d-1",
            "created_at": "2026-04-12T10:00:00+00:00",
        },
        backend=backend,
    )
    update_memory_node(
        state,
        text="Regular memory alpha",
        metadata={"type": "fact", "created_at": "2026-04-12T10:01:00+00:00"},
        backend=backend,
    )

    loaded_state = load_memory_node(state, backend=backend, top_k=5)
    digest_context = loaded_state.get("digest_context", [])
    analytics = loaded_state.get("memory_analytics", {})

    assert len(digest_context) == 1
    assert digest_context[0]["metadata"].get("digest_id") == "d-1"
    assert digest_context[0]["metadata"].get("is_digest") is True
    assert analytics.get("digest_count") == 1
