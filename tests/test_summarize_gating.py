"""Tests for node_summarize gating (W2.2) and failure handling (W1.4).

W2.2: the auxiliary-model fact-extraction call is skipped when the turn won't be
persisted (memory disabled or a trivial exchange) — node_update_memory would discard
the result anyway.

W1.4: on a blank/failed summary the node must NOT fall back to echoing the extraction
prompt; node_update_memory then builds a meaningful exchange-based summary instead of
persisting the instruction text as a user fact.
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from universal_agentic_framework.orchestration import graph_builder


class _RecordingModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        class _Out:
            content = "Extracted: user likes Postgres."
            usage_metadata = {"input_tokens": 5, "output_tokens": 7}

        return _Out()


class _RaisingModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError("auxiliary model unreachable")


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(profile=SimpleNamespace(language="en", name="test-profile"))


def _features(long_term_memory: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        long_term_memory=long_term_memory,
        memory_digest_chain_enabled=False,
    )


def _patch(monkeypatch, model, features) -> None:
    monkeypatch.setattr(graph_builder, "load_core_config", _fake_config)
    monkeypatch.setattr(graph_builder, "load_features_config", lambda: features)
    monkeypatch.setattr(
        graph_builder, "get_auxiliary_model",
        lambda config, language="en": (model, "fake-provider", "fake-model"),
    )
    monkeypatch.setattr(graph_builder, "track_node_execution", lambda *a, **k: nullcontext())
    monkeypatch.setattr(graph_builder, "track_tokens", lambda *a, **k: None)
    monkeypatch.setattr(graph_builder, "track_llm_call", lambda *a, **k: None)
    monkeypatch.setattr(graph_builder, "count_tokens_for_model", lambda *a, **k: 3)


def test_summarize_skips_llm_for_trivial_exchange(monkeypatch) -> None:
    model = _RecordingModel()
    _patch(monkeypatch, model, _features(long_term_memory=True))

    state = {
        "messages": [
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You're welcome."},
        ],
        "language": "en",
    }
    result = graph_builder.node_summarize(state)

    assert model.calls == 0  # no auxiliary-model call for a trivial turn
    assert result["summary_text"] == ""


def test_summarize_skips_llm_when_memory_disabled(monkeypatch) -> None:
    model = _RecordingModel()
    _patch(monkeypatch, model, _features(long_term_memory=False))

    state = {
        "messages": [
            {"role": "user", "content": "Tell me about quarterly revenue trends"},
            {"role": "assistant", "content": "Revenue rose 12%."},
        ],
        "language": "en",
    }
    result = graph_builder.node_summarize(state)

    assert model.calls == 0
    assert result["summary_text"] == ""


def test_summarize_runs_llm_for_substantive_exchange(monkeypatch) -> None:
    model = _RecordingModel()
    _patch(monkeypatch, model, _features(long_term_memory=True))

    state = {
        "messages": [
            {"role": "user", "content": "I prefer Postgres over MySQL for this project"},
            {"role": "assistant", "content": "Noted."},
        ],
        "language": "en",
    }
    result = graph_builder.node_summarize(state)

    assert model.calls == 1
    assert result["summary_text"] == "Extracted: user likes Postgres."


def test_summarize_failure_yields_empty_not_prompt_echo(monkeypatch) -> None:
    """W1.4: a failed summary must not echo the extraction prompt into summary_text."""
    model = _RaisingModel()
    _patch(monkeypatch, model, _features(long_term_memory=True))

    state = {
        "messages": [
            {"role": "user", "content": "Remember that my company is named Steuermann GmbH"},
            {"role": "assistant", "content": "Got it."},
        ],
        "language": "en",
    }
    result = graph_builder.node_summarize(state)

    assert model.calls == 1
    summary = result["summary_text"]
    assert summary == ""
    assert "Extract facts about the user" not in summary  # the prompt text never leaks
