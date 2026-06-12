"""Tests for compress_state threshold resolution, fill measurement, and status reporting."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from universal_agentic_framework.orchestration import performance_nodes as pn


def _messages(n: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i}"}
        for i in range(n)
    ]


def _install_summarizer(monkeypatch, *, content="a digest", fail=False):
    """Wire a fake summarizer with a controllable auxiliary LLM."""
    mock_llm = AsyncMock()
    if fail:
        mock_llm.ainvoke.side_effect = RuntimeError("provider down")
    else:
        mock_llm.ainvoke.return_value = SimpleNamespace(content=content)
    mock_factory = MagicMock()
    mock_factory.create_auxiliary_llm.return_value = mock_llm
    from universal_agentic_framework.memory.summarization import ConversationSummarizer

    summarizer = ConversationSummarizer(llm_factory=mock_factory)
    monkeypatch.setattr(pn, "get_summarizer", lambda: summarizer)
    return summarizer


def test_resolve_context_window_prefers_config(monkeypatch):
    # Patch the lazily-imported loader inside _resolve_context_window.
    import universal_agentic_framework.config as cfg_mod

    fake_cfg = SimpleNamespace(
        llm=SimpleNamespace(roles=SimpleNamespace(chat=SimpleNamespace(context_window_tokens=12345)))
    )
    monkeypatch.setattr(cfg_mod, "load_core_config", lambda: fake_cfg)
    assert pn._resolve_context_window({}) == 12345


def test_resolve_context_window_falls_back_to_probe(monkeypatch):
    import universal_agentic_framework.config as cfg_mod

    # No config override -> use probe snapshot matching model_used.
    fake_cfg = SimpleNamespace(
        llm=SimpleNamespace(roles=SimpleNamespace(chat=SimpleNamespace(context_window_tokens=None)))
    )
    monkeypatch.setattr(cfg_mod, "load_core_config", lambda: fake_cfg)
    state = {
        "model_used": "openai/foo",
        "llm_capability_probes": [
            {"model_name": "openai/bar", "metadata": {"context_window_tokens": 4096}},
            {"model_name": "openai/foo", "metadata": {"context_window_tokens": 65536}},
        ],
    }
    assert pn._resolve_context_window(state) == 65536


def test_resolve_context_window_probe_top_level_key(monkeypatch):
    """The adapter forwards context_window_tokens as a top-level row key (not nested)."""
    import universal_agentic_framework.config as cfg_mod

    fake_cfg = SimpleNamespace(
        llm=SimpleNamespace(roles=SimpleNamespace(chat=SimpleNamespace(context_window_tokens=None)))
    )
    monkeypatch.setattr(cfg_mod, "load_core_config", lambda: fake_cfg)
    state = {
        "model_used": "openai/foo",
        "llm_capability_probes": [
            {"model_name": "openai/foo", "context_window_tokens": 8192},
        ],
    }
    assert pn._resolve_context_window(state) == 8192


def test_resolve_context_window_default(monkeypatch):
    import universal_agentic_framework.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_core_config", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert pn._resolve_context_window({}) == 32768


def test_compress_state_skips_short_conversation(monkeypatch):
    _install_summarizer(monkeypatch)
    state = {"messages": _messages(4), "user_id": "u"}
    result = asyncio.run(pn.compress_state(state, force=False))
    assert result["last_compression_status"] == "skipped"
    assert len(result["messages"]) == 4


def test_compress_state_skips_when_below_threshold(monkeypatch):
    _install_summarizer(monkeypatch)
    monkeypatch.setattr(pn, "_resolve_context_window", lambda s: 100000)
    state = {
        "messages": _messages(20),
        "user_id": "u",
        "last_input_tokens": 100,  # well below 0.75 * 100000
    }
    result = asyncio.run(pn.compress_state(state, force=False))
    assert result["last_compression_status"] == "skipped"
    assert len(result["messages"]) == 20


def test_compress_state_compresses_when_fill_exceeds_threshold(monkeypatch):
    _install_summarizer(monkeypatch, content="rolling digest")
    monkeypatch.setattr(pn, "_resolve_context_window", lambda s: 1000)
    state = {
        "messages": _messages(20),
        "user_id": "u",
        "last_input_tokens": 900,  # above 0.75 * 1000 = 750
    }
    result = asyncio.run(pn.compress_state(state, force=False))
    assert result["last_compression_status"] == "ok"
    assert len(result["messages"]) < 20
    assert result["messages"][0].get("type") == "summary"


def test_compress_state_reports_error_on_summary_failure(monkeypatch):
    _install_summarizer(monkeypatch, fail=True)
    state = {"messages": _messages(20), "user_id": "u"}
    result = asyncio.run(pn.compress_state(state, force=True))
    assert result["last_compression_status"] == "error"
    # History untouched on failure.
    assert len(result["messages"]) == 20


def test_compress_state_force_ignores_threshold(monkeypatch):
    _install_summarizer(monkeypatch, content="forced digest")
    state = {"messages": _messages(20), "user_id": "u", "last_input_tokens": 1}
    result = asyncio.run(pn.compress_state(state, force=True))
    assert result["last_compression_status"] == "ok"
    assert len(result["messages"]) < 20
