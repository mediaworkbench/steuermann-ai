"""Phase 2 tests: blended retrieval + retrieval-touch persistence (flagged off).

Asserts the three plan guarantees on ``load_memory_node``:
- flag ON  → retrieval uses ``load_blended`` and semantic memories prepend episodic;
- flag OFF → byte-identical to today (legacy ``store.load``; no semantic split; no touch);
- the retrieval-touch fires once per memory then debounces, and never blocks the turn.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from universal_agentic_framework.memory.backend import MemoryRecord
from universal_agentic_framework.memory import nodes as memory_nodes
from universal_agentic_framework.memory.nodes import load_memory_node


class _FakeStore:
    def __init__(
        self,
        *,
        plain: Optional[List[MemoryRecord]] = None,
        blended: Optional[List[MemoryRecord]] = None,
    ) -> None:
        self._plain = plain or []
        self._blended = blended or []
        self.load_calls: List[Dict[str, Any]] = []
        self.load_blended_calls: List[Dict[str, Any]] = []
        self.update_metadata_calls: List[tuple[str, Dict[str, Any]]] = []

    def load(self, user_id, query=None, top_k=5, **kwargs):
        self.load_calls.append({"user_id": user_id, "query": query, "top_k": top_k, **kwargs})
        return list(self._plain)

    def load_blended(self, user_id, query=None, semantic_top_k=3, episodic_top_k=5):
        self.load_blended_calls.append(
            {
                "user_id": user_id,
                "query": query,
                "semantic_top_k": semantic_top_k,
                "episodic_top_k": episodic_top_k,
            }
        )
        return list(self._blended)

    def update_metadata(self, memory_id, patch):
        self.update_metadata_calls.append((memory_id, dict(patch)))
        return True


def _rec(text: str, *, tier: str = "episodic", mid: Optional[str] = None, access: int = 0) -> MemoryRecord:
    return MemoryRecord(
        user_id="u1",
        text=text,
        metadata={
            "memory_id": mid or uuid.uuid4().hex[:12],
            "cognitive_tier": tier,
            "access_count": access,
        },
    )


def _state() -> Dict[str, Any]:
    return {"user_id": "u1", "messages": [{"role": "user", "content": "tell me about tea"}]}


def test_flag_off_uses_plain_load_and_no_semantic_or_touch():
    plain = [_rec("episodic a"), _rec("episodic b")]
    store = _FakeStore(plain=plain)

    out = load_memory_node(_state(), backend=store, cognitive_enabled=False)

    assert store.load_blended_calls == []  # legacy path only
    assert store.load_calls  # plain load was used
    assert out["semantic_memory"] == []
    assert store.update_metadata_calls == []  # no touch writes when off
    assert [m["text"] for m in out["loaded_memory"]] == ["episodic a", "episodic b"]


def test_flag_on_blends_with_semantic_prepended():
    sem = _rec("semantic insight", tier="semantic")
    epi1 = _rec("episodic one")
    epi2 = _rec("episodic two")
    # load_blended already returns semantic-prepended order.
    store = _FakeStore(blended=[sem, epi1, epi2])

    out = load_memory_node(
        _state(), backend=store, cognitive_enabled=True, semantic_top_k=3, episodic_top_k=5
    )

    assert store.load_blended_calls and store.load_blended_calls[0]["semantic_top_k"] == 3
    assert store.load_blended_calls[0]["episodic_top_k"] == 5
    # Semantic prepends episodic in the injected order.
    assert out["loaded_memory"][0]["metadata"]["cognitive_tier"] == "semantic"
    assert [m["text"] for m in out["semantic_memory"]] == ["semantic insight"]
    assert out["memory_analytics"]["semantic_count"] == 1


def test_retrieval_touch_fires_once_then_debounces():
    mid_sem = uuid.uuid4().hex[:12]
    mid_epi = uuid.uuid4().hex[:12]
    blended = [_rec("sem", tier="semantic", mid=mid_sem, access=2), _rec("epi", mid=mid_epi, access=0)]
    store = _FakeStore(blended=blended)

    # First retrieval: both injected memories are touched once.
    load_memory_node(_state(), backend=store, cognitive_enabled=True)
    touched_first = {mid for mid, _ in store.update_metadata_calls}
    assert touched_first == {mid_sem, mid_epi}

    # The patch persists access_count + a fresh last_accessed (text-preserving update).
    patch = dict(store.update_metadata_calls)[mid_sem]
    assert patch["access_count"] == 2
    assert "last_accessed" in patch

    # Second retrieval of the same memories within the debounce window: no new writes.
    store.update_metadata_calls.clear()
    load_memory_node(_state(), backend=store, cognitive_enabled=True)
    assert store.update_metadata_calls == []


def test_touch_never_raises_when_backend_update_fails(monkeypatch):
    class _BoomStore(_FakeStore):
        def update_metadata(self, memory_id, patch):
            raise RuntimeError("qdrant down")

    store = _BoomStore(blended=[_rec("sem", tier="semantic", mid=uuid.uuid4().hex[:12])])
    # Force the debounce to always allow a touch so the failing path is exercised.
    monkeypatch.setattr(memory_nodes, "_should_touch", lambda _mid: True)

    # Must not raise — touch is best-effort.
    out = load_memory_node(_state(), backend=store, cognitive_enabled=True)
    assert out["loaded_memory"]


def test_flag_on_without_blended_backend_falls_back_to_plain():
    """A backend lacking load_blended (cognitive on) must not crash — legacy load."""
    class _NoBlend:
        def __init__(self):
            self.load_calls = 0

        def load(self, user_id, query=None, top_k=5, **kwargs):
            self.load_calls += 1
            return [_rec("episodic only")]

    store = _NoBlend()
    out = load_memory_node(_state(), backend=store, cognitive_enabled=True)
    assert store.load_calls >= 1
    assert out["semantic_memory"] == []
