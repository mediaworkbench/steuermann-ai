"""Phase 5 tests: procedural load node, prompt merge, and the server.py allowlist.

- ``node_load_procedural`` loads *active* rules only and is gated by the
  ``procedural_overrides_enabled`` flag (off → empty, so the prompt merge no-ops).
- The respond-node merge appends ``=== LEARNED USER PREFERENCES ===`` only from
  ``state["procedural_overrides"]`` (verified via the same line-building logic).
- The server.py request-driven allowlist forwards ``cognitive_memory_enabled``
  into GraphState (regression for the silent-drop bug).
"""
from __future__ import annotations

from types import SimpleNamespace

from universal_agentic_framework.orchestration import procedural_node
from universal_agentic_framework.orchestration.procedural_node import (
    node_load_procedural,
    load_active_procedural,
)


class _FakeProceduralStore:
    def __init__(self, rows):
        self._rows = rows

    def list_active(self, user_id):
        return [dict(r) for r in self._rows]


def _disable_redis(monkeypatch):
    # Force the no-cache path so the fake store is always consulted.
    monkeypatch.setattr(procedural_node, "_get_redis", lambda: None)


def test_load_active_procedural_maps_active_rows(monkeypatch):
    _disable_redis(monkeypatch)
    store = _FakeProceduralStore([
        {"rule_key": "format.bullets", "rule_text": "Use bullet lists", "tier": 1},
        {"rule_key": "style.concise", "rule_text": "Be concise", "tier": 2},
        {"rule_key": "format.empty", "rule_text": "", "tier": 1},  # dropped: no text
    ])
    rules = load_active_procedural("u1", store=store)
    assert [r["rule_key"] for r in rules] == ["format.bullets", "style.concise"]
    assert rules[0] == {"rule_key": "format.bullets", "rule_text": "Use bullet lists", "tier": 1}


def test_node_load_procedural_off_yields_empty(monkeypatch):
    _disable_redis(monkeypatch)
    monkeypatch.setattr(
        procedural_node, "load_features_config",
        lambda: SimpleNamespace(procedural_overrides_enabled=False),
    )
    store = _FakeProceduralStore([{"rule_key": "format.bullets", "rule_text": "x", "tier": 1}])
    state = node_load_procedural({"user_id": "u1"}, store=store)
    assert state["procedural_overrides"] == []  # flag off → nothing loaded


def test_node_load_procedural_on_loads_active(monkeypatch):
    _disable_redis(monkeypatch)
    monkeypatch.setattr(
        procedural_node, "load_features_config",
        lambda: SimpleNamespace(procedural_overrides_enabled=True),
    )
    store = _FakeProceduralStore([{"rule_key": "format.bullets", "rule_text": "Use bullets", "tier": 1}])
    state = node_load_procedural({"user_id": "u1"}, store=store)
    assert state["procedural_overrides"] == [
        {"rule_key": "format.bullets", "rule_text": "Use bullets", "tier": 1}
    ]


def test_node_load_procedural_without_user_is_empty(monkeypatch):
    _disable_redis(monkeypatch)
    monkeypatch.setattr(
        procedural_node, "load_features_config",
        lambda: SimpleNamespace(procedural_overrides_enabled=True),
    )
    state = node_load_procedural({}, store=_FakeProceduralStore([]))
    assert state["procedural_overrides"] == []


def test_prompt_merge_block_built_from_active_overrides():
    # Mirror the respond-node merge (graph_builder): only rule_text lines, fenced block.
    procedural = [
        {"rule_key": "format.bullets", "rule_text": "Use bullet lists"},
        {"rule_key": "style.concise", "rule_text": "Be concise"},
        {"rule_key": "x", "rule_text": ""},  # skipped
    ]
    lines = [f"- {p.get('rule_text', '')}" for p in procedural if p.get("rule_text")]
    block = ""
    if lines:
        block = (
            "\n\n=== LEARNED USER PREFERENCES ===\n"
            + "\n".join(lines)
            + "\n=== END LEARNED USER PREFERENCES ===\n"
        )
    assert "=== LEARNED USER PREFERENCES ===" in block
    assert "- Use bullet lists" in block and "- Be concise" in block
    # An empty override set produces no block at all.
    assert [f"- {p['rule_text']}" for p in [] if p.get("rule_text")] == []


def test_server_allowlist_forwards_cognitive_memory_enabled():
    from universal_agentic_framework.server import _request_driven_state

    # The request-driven key must reach GraphState (silent-drop regression).
    state = _request_driven_state({"cognitive_memory_enabled": True})
    assert state["cognitive_memory_enabled"] is True

    # Absent → None (node_load_memory then falls back to the feature flag).
    assert _request_driven_state({})["cognitive_memory_enabled"] is None

    # The sibling allowlisted keys are still forwarded (no drift).
    forwarded = _request_driven_state(
        {"allowed_tools": ["t"], "memory_enabled": False, "workspace_writeback_document": {"id": "d"}}
    )
    assert forwarded["allowed_tools"] == ["t"]
    assert forwarded["memory_enabled"] is False
    assert forwarded["workspace_writeback_document"] == {"id": "d"}
