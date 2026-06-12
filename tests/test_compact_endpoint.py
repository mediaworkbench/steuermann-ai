"""Tests for the LangGraph /compact endpoint status handling.

server.py builds the graph (and a Postgres checkpointer) at import time, so we
patch build_graph before importing and then inject a fake GRAPH.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def server_module(monkeypatch):
    # Patch graph construction so importing server.py does not require Postgres
    # or a running event loop.
    import universal_agentic_framework.orchestration.graph_builder as gb

    monkeypatch.setattr(gb, "build_graph", lambda: MagicMock(), raising=True)
    import importlib

    server = importlib.import_module("universal_agentic_framework.server")
    importlib.reload(server)
    return server


def _checkpoint_tuple(messages):
    ct = MagicMock()
    ct.checkpoint = {"channel_values": {"messages": messages}}
    ct.config = {"configurable": {"checkpoint_ns": ""}}
    ct.metadata = {}
    return ct


def test_compact_ok(server_module, monkeypatch):
    server = server_module
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    fake_graph = MagicMock()
    fake_graph.checkpointer.aget_tuple = AsyncMock(return_value=_checkpoint_tuple(messages))
    fake_graph.aupdate_state = AsyncMock(return_value={})
    monkeypatch.setattr(server, "GRAPH", fake_graph)

    async def _fake_compress(state, force=False):
        state["messages"] = messages[:6]
        state["tokens_used"] = 123
        state["digest_chain"] = []
        state["last_compression_status"] = "ok"
        return state

    monkeypatch.setattr(server, "compress_state", _fake_compress)

    result = asyncio.run(server.compact_conversation({"session_id": "s1", "user_id": "u"}))
    assert result["status"] == "ok"
    assert result["messages_before"] == 20
    assert result["messages_after"] == 6
    assert result["estimated_tokens"] == 123
    fake_graph.aupdate_state.assert_awaited_once()


def test_compact_skipped(server_module, monkeypatch):
    server = server_module
    messages = [{"role": "user", "content": "hi"}]

    fake_graph = MagicMock()
    fake_graph.checkpointer.aget_tuple = AsyncMock(return_value=_checkpoint_tuple(messages))
    fake_graph.aupdate_state = AsyncMock()
    monkeypatch.setattr(server, "GRAPH", fake_graph)

    async def _fake_compress(state, force=False):
        state["last_compression_status"] = "skipped"
        return state

    monkeypatch.setattr(server, "compress_state", _fake_compress)

    result = asyncio.run(server.compact_conversation({"session_id": "s1", "user_id": "u"}))
    assert result["status"] == "skipped"
    fake_graph.aupdate_state.assert_not_awaited()


def test_compact_error_leaves_checkpoint_untouched(server_module, monkeypatch):
    server = server_module
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    fake_graph = MagicMock()
    fake_graph.checkpointer.aget_tuple = AsyncMock(return_value=_checkpoint_tuple(messages))
    fake_graph.aupdate_state = AsyncMock()
    monkeypatch.setattr(server, "GRAPH", fake_graph)

    async def _fake_compress(state, force=False):
        state["last_compression_status"] = "error"
        return state

    monkeypatch.setattr(server, "compress_state", _fake_compress)

    result = asyncio.run(server.compact_conversation({"session_id": "s1", "user_id": "u"}))
    assert result["status"] == "error"
    fake_graph.aupdate_state.assert_not_awaited()


def test_compact_requires_session_id(server_module):
    from fastapi import HTTPException

    server = server_module
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.compact_conversation({"user_id": "u"}))
    assert exc.value.status_code == 400
