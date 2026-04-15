from universal_agentic_framework.memory import InMemoryMemoryManager


def test_memory_upsert_and_load():
    mgr = InMemoryMemoryManager()
    mgr.upsert(user_id="u1", text="Hello world", metadata={"k": 1})
    mgr.upsert(user_id="u1", text="Another note", metadata={"k": 2})

    results = mgr.load(user_id="u1", query="note", top_k=5)
    assert len(results) == 1
    assert results[0].text == "Another note"


def test_memory_clear():
    mgr = InMemoryMemoryManager()
    mgr.upsert(user_id="u1", text="Hello")
    mgr.clear("u1")
    assert mgr.load(user_id="u1") == []
