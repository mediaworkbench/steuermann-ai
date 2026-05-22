"""Tests for the streaming chat path.

FastAPI /api/chat/stream endpoint is tested here (mocking httpx). The
LangGraph /stream endpoint requires live services and is marked integration.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.attachments import (
    AttachmentManagerConfig,
    ChatAttachmentManager,
    ChatWorkspaceManager,
    UserWorkspaceFileManager,
    WorkspaceManagerConfig,
)
from backend.circuit_breaker import CircuitState
from backend.rate_limit import limiter
from backend.routers import chat as chat_module
from backend.routers.chat import router as chat_router


# ── Shared fakes (identical to test_chat_router.py) ──────────────────────────


class FakeSettingsStore:
    def get_user_settings(self, user_id: str) -> Dict[str, Any] | None:
        return {
            "tool_toggles": {},
            "rag_config": {"collection": "", "top_k": 5},
            "preferred_model": None,
            "theme": "auto",
            "language": "en",
        }


class FakeConversationStore:
    def __init__(self) -> None:
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: list[Dict[str, Any]] = []

    def create_conversation(self, conversation_id: str, user_id: str) -> None:
        self._conversations[conversation_id] = {"id": conversation_id, "user_id": user_id}

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def add_message(self, **kwargs) -> Dict[str, Any]:
        self.messages.append(kwargs)
        return kwargs

    def get_messages(self, conversation_id: str, limit: int = 500, offset: int = 0):
        rows = [r for r in self.messages if r.get("conversation_id") == conversation_id]
        return rows[offset : offset + limit]


class FakeAttachmentStore:
    def get_attachments_by_ids(self, conversation_id: str, attachment_ids: list[str], include_inactive: bool = False):
        return []


class FakeWorkspaceStore:
    def log_workspace_operation(self, **kwargs) -> Dict[str, Any]:
        return {}


class FakeWorkspaceDocumentStore:
    def __init__(self) -> None:
        self._documents: list[Dict[str, Any]] = []

    def get_documents_by_ids(self, user_id: str, document_ids: list[str]):
        return []

    def get_document(self, document_id: str, user_id: str):
        return None

    def list_documents(self, user_id: str, limit: int = 200, offset: int = 0):
        return []


class FakeLLMCapabilityProbeStore:
    def get_latest_probes(self) -> list[Dict[str, Any]]:
        return []


# ── SSE helper ────────────────────────────────────────────────────────────────

def parse_sse_stream(raw: str) -> list[Dict[str, Any]]:
    """Parse a raw SSE body into a list of {event, data} dicts."""
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = "message"
        data_line = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_line = line[6:].strip()
        events.append({"event": event_type, "data": data_line})
    return events


# ── Streaming SSE payload the fake LangGraph returns ─────────────────────────

_SSE_RESPONSE = (
    "event: token\ndata: {\"delta\": \"Hello\"}\n\n"
    "event: token\ndata: {\"delta\": \" world\"}\n\n"
    "event: node\ndata: {\"node\": \"retrieve_knowledge\", \"label\": \"Searching knowledge base...\"}\n\n"
    "event: tool_call\ndata: {\"name\": \"web_search\", \"status\": \"start\", \"label\": \"Searching the web...\"}\n\n"
    "event: tool_call\ndata: {\"name\": \"web_search\", \"status\": \"end\", \"label\": \"Searching the web...\"}\n\n"
    "event: metadata\ndata: {\"tokens_used\": 20, \"input_tokens\": 8, \"output_tokens\": 12, "
    "\"model_used\": \"test-model\", \"tool_results\": {}, \"sources\": [], "
    "\"rag_attempted\": false, \"rag_doc_count\": 0, \"loaded_memory\": []}\n\n"
    "data: [DONE]\n\n"
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def stream_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    # Fake httpx that returns the SSE payload as a stream
    class _FakeStreamContext:
        def __init__(self, status_code: int = 200, body: str = _SSE_RESPONSE) -> None:
            self.status_code = status_code
            self.headers = {"content-type": "text/event-stream"}
            self._body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aread(self) -> bytes:
            return self._body.encode()

        async def aiter_text(self) -> AsyncGenerator[str, None]:
            # Yield the full SSE body in one chunk for simplicity
            yield self._body

    class _FakeAsyncClientStream:
        def __init__(self, *args, **kwargs) -> None:
            self.last_json: Dict[str, Any] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method: str, url: str, json: Dict[str, Any]):
            _FakeAsyncClientStream.last_request_json = json
            return _FakeStreamContext()

    # Also patch the non-stream post (used by _validate_preferred_model and _classify_workspace_intent_llm)
    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": '{"edit": false, "save": false, "new_version": false}'}}]}

    class _FakeHttpx:
        AsyncClient = _FakeAsyncClientStream
        RequestError = Exception
        HTTPStatusError = Exception

    monkeypatch.setattr(chat_module, "httpx", _FakeHttpx)

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=_noop_lifespan)
    app.state.limiter = limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)

    conversation_store = FakeConversationStore()
    conversation_store.create_conversation("conv-1", "u1")
    app.state.conversation_store = conversation_store
    app.state.settings_store = FakeSettingsStore()
    app.state.llm_capability_probe_store = FakeLLMCapabilityProbeStore()
    app.state.conversation_attachment_store = FakeAttachmentStore()
    app.state.conversation_workspace_store = FakeWorkspaceStore()
    app.state.workspace_document_store = FakeWorkspaceDocumentStore()

    attachment_manager = ChatAttachmentManager(
        AttachmentManagerConfig(root_dir=tmp_path / "chat-workspaces")
    )
    app.state.chat_attachment_manager = attachment_manager
    app.state.chat_workspace_manager = ChatWorkspaceManager(
        attachment_manager=attachment_manager,
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24),
    )
    app.state.user_workspace_file_manager = UserWorkspaceFileManager(
        tmp_path / "user-workspace"
    )
    app.state.analytics_store = None

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, conversation_store, _FakeAsyncClientStream


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_chat_stream_returns_event_stream_content_type(stream_client) -> None:
    test_client, _, _ = stream_client

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1", "language": "en"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_chat_stream_forwards_token_events(stream_client) -> None:
    test_client, _, _ = stream_client

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1"},
    )

    assert response.status_code == 200
    events = parse_sse_stream(response.text)
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) == 2
    assert json.loads(token_events[0]["data"])["delta"] == "Hello"
    assert json.loads(token_events[1]["data"])["delta"] == " world"


def test_chat_stream_forwards_node_and_tool_events(stream_client) -> None:
    test_client, _, _ = stream_client

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Search something", "user_id": "u1"},
    )

    events = parse_sse_stream(response.text)
    node_events = [e for e in events if e["event"] == "node"]
    tool_events = [e for e in events if e["event"] == "tool_call"]

    assert any(json.loads(e["data"])["node"] == "retrieve_knowledge" for e in node_events)
    tool_names = [json.loads(e["data"])["name"] for e in tool_events]
    assert "web_search" in tool_names


def test_chat_stream_terminates_with_done(stream_client) -> None:
    test_client, _, _ = stream_client

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1"},
    )

    assert response.text.endswith("data: [DONE]\n\n") or "data: [DONE]" in response.text


def test_chat_stream_forwards_metadata_event(stream_client) -> None:
    test_client, _, _ = stream_client

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1"},
    )

    events = parse_sse_stream(response.text)
    metadata_events = [e for e in events if e["event"] == "metadata"]
    assert metadata_events, "Expected at least one metadata event"
    meta = json.loads(metadata_events[0]["data"])
    assert meta["model_used"] == "test-model"
    assert meta["tokens_used"] == 20


def test_chat_stream_persists_messages_when_conversation_id_given(stream_client) -> None:
    test_client, conversation_store, _ = stream_client

    test_client.post(
        "/api/chat/stream",
        json={
            "message": "Hello",
            "user_id": "u1",
            "conversation_id": "conv-1",
        },
    )

    user_msgs = [m for m in conversation_store.messages if m.get("role") == "user"]
    assistant_msgs = [m for m in conversation_store.messages if m.get("role") == "assistant"]
    assert len(user_msgs) >= 1
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[0]["content"] == "Hello world"


def test_chat_stream_no_persistence_without_conversation_id(stream_client) -> None:
    test_client, conversation_store, _ = stream_client

    test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1"},
    )

    assert len(conversation_store.messages) == 0


def test_chat_stream_circuit_breaker_open_returns_error_event(
    stream_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _, _ = stream_client

    monkeypatch.setattr(
        chat_module.LANGGRAPH_CIRCUIT_BREAKER,
        "_state",
        CircuitState.OPEN,
    )
    import time
    monkeypatch.setattr(
        chat_module.LANGGRAPH_CIRCUIT_BREAKER,
        "_last_failure_at",
        time.time(),  # recent — won't transition to HALF_OPEN yet
    )

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "Hello", "user_id": "u1"},
    )

    assert response.status_code == 200
    events = parse_sse_stream(response.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert error_events, "Expected an error SSE event when circuit breaker is open"
    assert "unavailable" in json.loads(error_events[0]["data"])["message"].lower()


def test_chat_stream_workspace_writeback_uses_streaming_path(
    stream_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When workspace_writeback_requested is true the endpoint still streams.

    Writeback is performed inside _proxy_stream after [DONE] arrives — there is
    no sync fallback.  The response must be SSE and forward token events normally.
    """
    test_client, _, _FakeAsyncClientStream = stream_client

    monkeypatch.setattr(chat_module, "_should_check_workspace_intent", lambda docs, atts: True)

    async def _fake_classify(message: str, language: str = "en") -> Dict[str, Any]:
        return {"edit": True, "save": True, "new_version": False}

    monkeypatch.setattr(chat_module, "_classify_workspace_intent_llm", _fake_classify)

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "rewrite and save this document", "user_id": "u1"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = parse_sse_stream(response.text)
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) >= 1, "Expected token events in SSE response"
