from types import SimpleNamespace

from universal_agentic_framework.orchestration import graph_builder


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_core_config():
    return SimpleNamespace(
        fork=SimpleNamespace(name="starter", language="en"),
        tokens=SimpleNamespace(
            default_budget=10000,
            per_node_budgets={"summarization_node": 2000, "update_memory": 2000},
        ),
    )


def test_node_summarize_normalizes_and_prunes_digest_chain(monkeypatch):
    monkeypatch.setattr(graph_builder, "load_core_config", lambda: _fake_core_config())
    monkeypatch.setattr(
        graph_builder,
        "load_features_config",
        lambda: SimpleNamespace(long_term_memory=True, memory_digest_chain_enabled=True),
    )
    monkeypatch.setattr(graph_builder, "safe_get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        graph_builder,
        "resolve_initial_model_metadata",
        lambda *args, **kwargs: ("fake_provider", "fake_model"),
    )
    monkeypatch.setattr(
        graph_builder,
        "_invoke_with_model_fallback",
        lambda **kwargs: ("user fact summary", "fake_provider", "fake_model", None, None),
    )
    monkeypatch.setattr(graph_builder, "track_node_execution", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(graph_builder, "track_llm_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_builder, "track_tokens", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_builder, "estimate_tokens", lambda text: max(1, len(str(text)) // 10))
    monkeypatch.setattr(
        graph_builder,
        "get_budget_context",
        lambda *args, **kwargs: {"per_turn_budget": 5000, "turn_remaining": 5000},
    )
    monkeypatch.setattr(graph_builder, "get_node_budget", lambda *args, **kwargs: 2000)
    monkeypatch.setattr(graph_builder, "per_node_hard_limit_enabled", lambda *args, **kwargs: False)
    monkeypatch.setattr(graph_builder, "require_tokens", lambda *args, **kwargs: None)

    messages = []
    for idx in range(1, 8):
        messages.append(
            {
                "role": "system",
                "type": "summary",
                "content": f"summary {idx}",
                "digest_id": f"d-{idx}",
                "previous_digest_id": f"d-{idx-1}" if idx > 1 else None,
                "timestamp": f"2026-05-15T10:0{idx}:00+00:00",
            }
        )
    messages.extend(
        [
            {"role": "user", "content": "I prefer concise answers."},
            {"role": "assistant", "content": "Understood."},
        ]
    )

    state = {
        "user_id": "u-digest",
        "language": "en",
        "messages": messages,
        "digest_chain": [{"digest_id": "legacy-d"}],
    }

    result = graph_builder.node_summarize(state)

    assert result.get("summary_text") == "user fact summary"
    assert len(result.get("digest_chain", [])) == 5
    assert [d.get("digest_id") for d in result["digest_chain"]] == [
        "d-7",
        "d-6",
        "d-5",
        "d-4",
        "d-3",
    ]


def test_node_update_memory_forwards_digest_chain_and_metadata(monkeypatch):
    captured = {}

    monkeypatch.setattr(graph_builder, "load_core_config", lambda: _fake_core_config())
    monkeypatch.setattr(
        graph_builder,
        "load_features_config",
        lambda: SimpleNamespace(long_term_memory=True, memory_digest_chain_enabled=True),
    )
    monkeypatch.setattr(graph_builder, "build_memory_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph_builder, "track_node_execution", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(graph_builder, "track_memory_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_builder, "estimate_tokens", lambda text: max(1, len(str(text)) // 10))
    monkeypatch.setattr(
        graph_builder,
        "get_budget_context",
        lambda *args, **kwargs: {"per_turn_budget": 5000, "turn_remaining": 5000},
    )
    monkeypatch.setattr(graph_builder, "get_node_budget", lambda *args, **kwargs: 2000)
    monkeypatch.setattr(graph_builder, "per_node_hard_limit_enabled", lambda *args, **kwargs: False)
    monkeypatch.setattr(graph_builder, "require_tokens", lambda *args, **kwargs: None)

    def _fake_update_memory_node(state, text, metadata=None, backend=None, messages=None, digest_chain=None):
        captured["text"] = text
        captured["metadata"] = metadata
        captured["digest_chain"] = digest_chain
        return state

    monkeypatch.setattr(graph_builder, "update_memory_node", _fake_update_memory_node)

    state = {
        "user_id": "u2",
        "language": "en",
        "messages": [
            {"role": "user", "content": "I like black coffee."},
            {"role": "assistant", "content": "Noted."},
        ],
        "summary_text": "User likes black coffee.",
        "digest_chain": [
            {"digest_id": "d-new", "previous_digest_id": "d-old"},
            {"digest_id": "d-old", "previous_digest_id": None},
        ],
    }

    graph_builder.node_update_memory(state)

    assert captured["digest_chain"][0]["digest_id"] == "d-new"
    assert captured["metadata"]["digest_id"] == "d-new"
    assert captured["metadata"]["previous_digest_id"] == "d-old"
    assert captured["metadata"]["digest_chain_ids"] == ["d-new", "d-old"]
    assert captured["metadata"]["digest_chain_length"] == 2


def test_node_update_memory_disables_digest_chain_via_feature_flag(monkeypatch):
    captured = {}

    monkeypatch.setattr(graph_builder, "load_core_config", lambda: _fake_core_config())
    monkeypatch.setattr(
        graph_builder,
        "load_features_config",
        lambda: SimpleNamespace(long_term_memory=True, memory_digest_chain_enabled=False),
    )
    monkeypatch.setattr(graph_builder, "build_memory_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph_builder, "track_node_execution", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(graph_builder, "track_memory_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_builder, "estimate_tokens", lambda text: max(1, len(str(text)) // 10))
    monkeypatch.setattr(
        graph_builder,
        "get_budget_context",
        lambda *args, **kwargs: {"per_turn_budget": 5000, "turn_remaining": 5000},
    )
    monkeypatch.setattr(graph_builder, "get_node_budget", lambda *args, **kwargs: 2000)
    monkeypatch.setattr(graph_builder, "per_node_hard_limit_enabled", lambda *args, **kwargs: False)
    monkeypatch.setattr(graph_builder, "require_tokens", lambda *args, **kwargs: None)

    def _fake_update_memory_node(state, text, metadata=None, backend=None, messages=None, digest_chain=None):
        captured["metadata"] = metadata
        captured["digest_chain"] = digest_chain
        return state

    monkeypatch.setattr(graph_builder, "update_memory_node", _fake_update_memory_node)

    state = {
        "user_id": "u3",
        "language": "en",
        "messages": [
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "Okay."},
        ],
        "summary_text": "remember this",
        "digest_chain": [{"digest_id": "d-hidden", "previous_digest_id": None}],
    }

    graph_builder.node_update_memory(state)

    assert captured["digest_chain"] == []
    assert "digest_id" not in captured["metadata"]
    assert "digest_chain_ids" not in captured["metadata"]
