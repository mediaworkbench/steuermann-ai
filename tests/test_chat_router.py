from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.attachments import AttachmentManagerConfig, ChatAttachmentManager, ChatWorkspaceManager, UserWorkspaceFileManager, WorkspaceManagerConfig
from backend.db import WorkspaceVersionConflictError
from backend.rate_limit import limiter
from backend.routers import chat as chat_module
from backend.routers.chat import router as chat_router


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

    def get_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conv = self._conversations.get(conversation_id)
        if conv is None or (user_id is not None and conv["user_id"] != user_id):
            return None
        return conv

    def add_message(self, **kwargs) -> Dict[str, Any]:
        self.messages.append(kwargs)
        return kwargs

    def get_messages(
        self, conversation_id: str, limit: int = 500, offset: int = 0, *, latest: bool = False
    ) -> list[Dict[str, Any]]:
        rows = [row for row in self.messages if row.get("conversation_id") == conversation_id]
        if latest:
            end = len(rows) - offset
            return rows[max(0, end - limit) : max(0, end)]
        return rows[offset : offset + limit]


class FakeAttachmentStore:
    def __init__(self, attachments: Optional[list[Dict[str, Any]]] = None) -> None:
        self._attachments = attachments or []

    def get_attachments_by_ids(self, conversation_id: str, attachment_ids: list[str], include_inactive: bool = False):
        rows_by_id = {
            row["id"]: row
            for row in self._attachments
            if row["conversation_id"] == conversation_id and (include_inactive or row.get("status") == "active")
        }
        return [rows_by_id[attachment_id] for attachment_id in attachment_ids if attachment_id in rows_by_id]

    def get_attachment(self, attachment_id: str, conversation_id: Optional[str] = None, include_inactive: bool = False):
        for row in self._attachments:
            if row["id"] != attachment_id:
                continue
            if conversation_id is not None and row["conversation_id"] != conversation_id:
                continue
            if not include_inactive and row.get("status") != "active":
                continue
            return row
        return None


class FakeWorkspaceStore:
    def __init__(self) -> None:
        self._workspaces: Dict[str, Dict[str, Any]] = {}
        self._operations: list[Dict[str, Any]] = []

    def get_workspace(self, conversation_id: str, include_inactive: bool = False):
        workspace = self._workspaces.get(conversation_id)
        if not workspace:
            return None
        if not include_inactive and workspace.get("status") != "active":
            return None
        return dict(workspace)

    def upsert_workspace(self, conversation_id: str, user_id: str, root_path: str, expires_at=None, status: str = "active"):
        now = datetime.now(timezone.utc).isoformat()
        workspace = self._workspaces.get(conversation_id, {
            "conversation_id": conversation_id,
            "created_at": now,
        })
        workspace.update(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "root_path": root_path,
                "status": status,
                "updated_at": now,
                "last_activity_at": now,
                "expires_at": expires_at.isoformat() if hasattr(expires_at, "isoformat") else expires_at,
            }
        )
        self._workspaces[conversation_id] = workspace
        return dict(workspace)

    def touch_workspace(self, conversation_id: str, expires_at=None):
        workspace = self._workspaces.get(conversation_id)
        if not workspace:
            return None
        workspace["status"] = "active"
        workspace["updated_at"] = datetime.now(timezone.utc).isoformat()
        workspace["last_activity_at"] = workspace["updated_at"]
        if expires_at is not None:
            workspace["expires_at"] = expires_at.isoformat() if hasattr(expires_at, "isoformat") else expires_at
        return dict(workspace)

    def log_workspace_operation(self, **kwargs):
        entry = {
            "id": len(self._operations) + 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._operations.append(entry)
        return entry


class FakeWorkspaceDocumentStore:
    def __init__(self, documents: Optional[list[Dict[str, Any]]] = None) -> None:
        self._documents = documents or []

    def get_documents_by_ids(self, user_id: str, document_ids: list[str]):
        rows_by_id = {
            row["id"]: row
            for row in self._documents
            if row["user_id"] == user_id
        }
        return [rows_by_id[document_id] for document_id in document_ids if document_id in rows_by_id]

    def get_document(self, document_id: str, user_id: str):
        for row in self._documents:
            if row["id"] == document_id and row["user_id"] == user_id:
                return row
        return None

    def list_documents(self, user_id: str, limit: int = 200, offset: int = 0):
        _ = limit
        _ = offset
        return [row for row in self._documents if row["user_id"] == user_id]

    def update_document_content(
        self,
        document_id: str,
        user_id: str,
        content_text: str,
        size_bytes: int,
        sha256: str,
        expected_version: Optional[int] = None,
        source: str = "user",
    ):
        for row in self._documents:
            if row["id"] == document_id and row["user_id"] == user_id:
                if expected_version is not None and int(row.get("version", 0)) != expected_version:
                    raise WorkspaceVersionConflictError(int(row.get("version", 0)), expected_version)
                row["content_text"] = content_text
                row["size_bytes"] = size_bytes
                row["sha256"] = sha256
                row["version"] = int(row.get("version", 0)) + 1
                row["last_source"] = source
                return row
        return None

    def count_documents(self, user_id: str) -> int:
        return sum(1 for row in self._documents if row["user_id"] == user_id)


class FakeLLMCapabilityProbeStore:
    def list_probe_results(self, profile_id: str, limit: int = 100) -> list[Dict[str, Any]]:
        _ = limit
        return [
            {
                "profile_id": profile_id,
                "provider_id": "lmstudio",
                "model_name": "liquid/lfm2-24b-a2b",
                "configured_tool_calling_mode": "native",
                "supports_bind_tools": False,
                "supports_tool_schema": False,
                "capability_mismatch": True,
                "status": "warning",
                "error_message": "bind_tools_failed: mock",
                "probed_at": "2026-05-09T00:00:00+00:00",
            }
        ]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.setenv("PROFILE_ID", "safety")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    async def _fake_cb_call(func):
        return await func()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {
                "messages": [{"role": "assistant", "content": _FakeAsyncClient.response_content}],
                "tokens_used": 12,
                "input_tokens": 5,
                "output_tokens": 7,
                "model_used": "test-model",
                "profile_id": _FakeAsyncClient.response_profile_id,
                "tool_results": {},
                "sources": [],
            }

    class _FakeAsyncClient:
        response_content = "Attachment-aware answer"
        response_profile_id = "safety"

        def __init__(self, *args, **kwargs) -> None:
            self.last_json = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
            self.last_json = json
            _FakeAsyncClient.last_request_json = json
            return _FakeResponse()

    monkeypatch.setattr(chat_module.LANGGRAPH_CIRCUIT_BREAKER, "call", _fake_cb_call)
    monkeypatch.setattr(chat_module, "httpx", type("_HttpxModule", (), {"AsyncClient": _FakeAsyncClient, "RequestError": Exception, "HTTPStatusError": Exception}))

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
    app.state.conversation_attachment_store = FakeAttachmentStore(
        attachments=[
            {
                "id": "att-1",
                "conversation_id": "conv-1",
                "user_id": "u1",
                "original_name": "notes.md",
                "mime_type": "text/markdown",
                "size_bytes": 15,
                "extracted_text": "# Notes\nHello",
                "status": "active",
            }
        ]
    )
    attachment_manager = ChatAttachmentManager(
        AttachmentManagerConfig(root_dir=tmp_path / "chat-workspaces")
    )
    app.state.chat_attachment_manager = attachment_manager
    app.state.chat_workspace_manager = ChatWorkspaceManager(
        attachment_manager=attachment_manager,
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24),
    )
    app.state.conversation_workspace_store = FakeWorkspaceStore()
    user_workspace_manager = UserWorkspaceFileManager(
        WorkspaceManagerConfig(root_dir=tmp_path / "user-workspaces", retention_hours=24)
    )
    stored_metadata = user_workspace_manager.store_document_file(
        user_id="u1",
        document_id="916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        filename="flatsplit.md",
        content=b"# Flat split\nDraft text",
    )
    app.state.user_workspace_file_manager = user_workspace_manager
    app.state.workspace_document_store = FakeWorkspaceDocumentStore(
        documents=[
            {
                "id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
                "user_id": "u1",
                "filename": "flatsplit.md",
                "stored_path": stored_metadata["stored_path"],
                "mime_type": "text/markdown",
                "size_bytes": 42,
                "sha256": "abc",
                "content_text": "# Flat split\nDraft text",
                "version": 1,
            }
        ]
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, conversation_store, _FakeAsyncClient


def test_chat_forwards_attachment_context(client) -> None:
    test_client, conversation_store, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use the attachment",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "attachment_ids": ["att-1"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Attachment-aware answer"
    assert body["metadata"]["attachments_used"] == [{"id": "att-1", "original_name": "notes.md"}]
    assert body["metadata"]["documents_used"] == []
    assert body["metadata"]["memories_used"] == []
    assert body["metadata"]["workspace_document_writeback"] is None
    assert body["metadata"]["profile_id"] == "safety"
    assert fake_async_client.last_request_json["attachments"][0]["id"] == "att-1"
    assert fake_async_client.last_request_json["attachments"][0]["extracted_text"] == "# Notes\nHello"
    assert conversation_store.messages[0]["metadata"] == {
        "attachment_ids": ["att-1"],
        "document_ids": [],
    }
    assert conversation_store.messages[1]["metadata"] == {
        "attachments_used": [{"id": "att-1", "original_name": "notes.md"}],
        "documents_used": [],
        "memories_used": [],
        "workspace_document_writeback": None,
        "profile_id": "safety",
        "input_tokens": 5,
        "output_tokens": 7,
        "sources": [],
        "rag_attempted": False,
        "rag_doc_count": 0,
    }


def test_chat_rejects_other_users_conversation(client) -> None:
    """A user must not chat into (or read history from) another user's conversation."""
    test_client, conversation_store, _ = client
    conversation_store.create_conversation("conv-u2", "u2")

    response = test_client.post(
        "/api/chat",
        json={"message": "leak history", "language": "en", "conversation_id": "conv-u2"},
    )
    assert response.status_code == 404
    # Nothing was persisted into the other user's conversation.
    assert all(m.get("conversation_id") != "conv-u2" for m in conversation_store.messages)


def test_chat_stream_rejects_other_users_conversation(client) -> None:
    test_client, conversation_store, _ = client
    conversation_store.create_conversation("conv-u2b", "u2")

    response = test_client.post(
        "/api/chat/stream",
        json={"message": "leak", "language": "en", "conversation_id": "conv-u2b"},
    )
    assert response.status_code == 404


def test_chat_rejects_attachment_ids_without_conversation_id(client) -> None:
    test_client, _, _ = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use the attachment",
            "user_id": "u1",
            "language": "en",
            "attachment_ids": ["att-1"],
        },
    )

    assert response.status_code == 400
    assert "conversation_id" in response.json()["detail"]


def test_chat_forwards_llm_capability_probes(client) -> None:
    test_client, _, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Summarize this",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    probes = fake_async_client.last_request_json.get("llm_capability_probes")
    assert isinstance(probes, list)
    assert probes
    assert probes[0]["provider_id"] == "lmstudio"
    assert probes[0]["capability_mismatch"] is True


def test_chat_forwards_per_chat_disabled_tools(client) -> None:
    """Tools quick-disabled for this chat are overlaid as tool_toggles=False for this
    inference only (the user's saved settings are not modified)."""
    test_client, _, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "hi",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "disabled_tools": ["datetime_tool", "calculator_tool"],
        },
    )

    assert response.status_code == 200
    forwarded_toggles = fake_async_client.last_request_json["user_settings"]["tool_toggles"]
    assert forwarded_toggles["datetime_tool"] is False
    assert forwarded_toggles["calculator_tool"] is False


def test_resolve_chat_endpoint_uses_chat_role(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(api_base="http://localhost:1234/v1/")
    core_config = SimpleNamespace(
        llm=SimpleNamespace(get_role_provider=lambda role_name: provider if role_name == "chat" else None)
    )
    monkeypatch.setattr(chat_module, "load_core_config", lambda: core_config)

    assert chat_module._resolve_chat_endpoint() == "http://localhost:1234/v1"


def test_resolve_chat_endpoint_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    core_config = SimpleNamespace(
        llm=SimpleNamespace(get_role_provider=lambda role_name: SimpleNamespace(api_base=None))
    )
    monkeypatch.setattr(chat_module, "load_core_config", lambda: core_config)

    with pytest.raises(ValueError, match="No api_base configured for the chat role"):
        chat_module._resolve_chat_endpoint()


def test_chat_uses_user_settings_language_over_request_language(client) -> None:
    test_client, _, fake_async_client = client

    class _GermanSettingsStore:
        def get_user_settings(self, user_id: str) -> Dict[str, Any] | None:
            return {
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "preferred_model": None,
                "theme": "auto",
                "language": "de",
            }

    test_client.app.state.settings_store = _GermanSettingsStore()
    chat_module._settings_cache.clear()

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Language precedence check",
            "user_id": "u1",
            "language": "en",
        },
    )

    assert response.status_code == 200
    assert fake_async_client.last_request_json["language"] == "de"


def test_chat_tracks_profile_mismatch_metric_when_langgraph_profile_differs(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, fake_async_client = client
    fake_async_client.response_profile_id = "medical"
    monkeypatch.setattr(chat_module, "ACTIVE_PROFILE_ID", "safety")

    tracked: list[tuple[str, str, str]] = []

    def _track_mismatch(*, profile_name: str, active_profile_id: str, reported_profile_id: str) -> None:
        tracked.append((profile_name, active_profile_id, reported_profile_id))

    monkeypatch.setattr(chat_module, "track_profile_id_mismatch", _track_mismatch)

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Simple check",
            "user_id": "u1",
            "language": "en",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["profile_id"] == "medical"
    assert tracked == [(chat_module.PROFILE_ID, "safety", "medical")]


def test_chat_falls_back_to_active_profile_when_langgraph_profile_missing(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, fake_async_client = client
    fake_async_client.response_profile_id = None
    monkeypatch.setattr(chat_module, "ACTIVE_PROFILE_ID", "safety")

    tracked: list[tuple[str, str, str]] = []

    def _track_mismatch(*, profile_name: str, active_profile_id: str, reported_profile_id: str) -> None:
        tracked.append((profile_name, active_profile_id, reported_profile_id))

    monkeypatch.setattr(chat_module, "track_profile_id_mismatch", _track_mismatch)

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Simple check",
            "user_id": "u1",
            "language": "en",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["profile_id"] == "safety"
    assert tracked == []


def test_chat_rejects_missing_attachment_ids(client) -> None:
    test_client, _, _ = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use the attachment",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "attachment_ids": ["missing"],
        },
    )

    assert response.status_code == 400
    assert "Invalid attachment_ids" in response.json()["detail"]


def test_chat_forwards_multiple_attachments_in_requested_order(client) -> None:
    test_client, conversation_store, fake_async_client = client

    # Add a second active attachment to the same conversation.
    test_client.app.state.conversation_attachment_store._attachments.append(
        {
            "id": "att-2",
            "conversation_id": "conv-1",
            "user_id": "u1",
            "original_name": "second.md",
            "mime_type": "text/markdown",
            "size_bytes": 21,
            "extracted_text": "Second attachment body",
            "status": "active",
        }
    )

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use both attachments",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "attachment_ids": ["att-2", "att-1"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [a["id"] for a in fake_async_client.last_request_json["attachments"]] == ["att-2", "att-1"]
    assert conversation_store.messages[0]["metadata"] == {
        "attachment_ids": ["att-2", "att-1"],
        "document_ids": [],
    }
    assert body["metadata"]["attachments_used"] == [
        {"id": "att-2", "original_name": "second.md"},
        {"id": "att-1", "original_name": "notes.md"},
    ]
    assert body["metadata"]["documents_used"] == []
    assert body["metadata"]["workspace_document_writeback"] is None


def test_chat_infers_workspace_document_by_id_from_message(client) -> None:
    test_client, _, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Please improve workspace document (id: 916f651c-fbe5-4df6-a7c9-f156fb96e9fa)",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    workspace_docs = fake_async_client.last_request_json["workspace_documents"]
    assert len(workspace_docs) == 1
    assert workspace_docs[0]["id"] == "916f651c-fbe5-4df6-a7c9-f156fb96e9fa"
    assert workspace_docs[0]["filename"] == "flatsplit.md"
    body = response.json()
    assert body["metadata"]["documents_used"] == [
        {
            "id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
            "filename": "flatsplit.md",
            "version": 1,
        }
    ]


def test_chat_infers_workspace_document_by_quoted_filename_from_message(client) -> None:
    test_client, _, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": 'Look at workspace document "flatsplit.md" and improve it',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    workspace_docs = fake_async_client.last_request_json["workspace_documents"]
    assert len(workspace_docs) == 1
    assert workspace_docs[0]["filename"] == "flatsplit.md"
    body = response.json()
    assert body["metadata"]["documents_used"] == [
        {
            "id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
            "filename": "flatsplit.md",
            "version": 1,
        }
    ]


def _fake_models_client(model_ids: list[str]):
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"data": [{"id": model_id} for model_id in model_ids]}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            return _FakeResponse()

    return type("_HttpxModule", (), {"AsyncClient": _FakeAsyncClient})


@pytest.mark.asyncio
async def test_validate_preferred_model_matches_endpoint_ids_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_cb_call(func, *args):
        return await func(*args)

    monkeypatch.setattr(chat_module.MODEL_VALIDATION_CIRCUIT_BREAKER, "call", _fake_cb_call)
    monkeypatch.setattr(chat_module, "httpx", _fake_models_client(["liquid/lfm2-24b-a2b", "openai/gpt-5"]))
    monkeypatch.setattr(chat_module, "_resolve_chat_endpoint", lambda: "http://verbatim.test/v1")

    assert await chat_module._validate_preferred_model("liquid/lfm2-24b-a2b") == ("liquid/lfm2-24b-a2b", None)
    # An ID that genuinely starts with ``openai/`` (OpenRouter) is matched as-is.
    assert await chat_module._validate_preferred_model("openai/gpt-5") == ("openai/gpt-5", None)


@pytest.mark.asyncio
async def test_validate_preferred_model_does_not_alias_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport-prefixed ID is not the endpoint's ID: fall back to the default model."""
    async def _fake_cb_call(func, *args):
        return await func(*args)

    monkeypatch.setattr(chat_module.MODEL_VALIDATION_CIRCUIT_BREAKER, "call", _fake_cb_call)
    monkeypatch.setattr(chat_module, "httpx", _fake_models_client(["liquid/lfm2-24b-a2b"]))
    monkeypatch.setattr(chat_module, "_resolve_chat_endpoint", lambda: "http://alias.test/v1")

    validated, warning = await chat_module._validate_preferred_model("openai/liquid/lfm2-24b-a2b")
    assert validated is None
    assert "not found" in warning


@pytest.mark.asyncio
async def test_validate_preferred_model_caches_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back-to-back chat turns must not each pay a /models round-trip."""
    calls: list[str] = []

    async def _fake_cb_call(func, base):
        calls.append(base)
        return ["liquid/lfm2-24b-a2b"]

    monkeypatch.setattr(chat_module.MODEL_VALIDATION_CIRCUIT_BREAKER, "call", _fake_cb_call)
    monkeypatch.setattr(chat_module, "_resolve_chat_endpoint", lambda: "http://cache.test/v1")

    for _ in range(3):
        validated, warning = await chat_module._validate_preferred_model("liquid/lfm2-24b-a2b")
        assert (validated, warning) == ("liquid/lfm2-24b-a2b", None)
    assert calls == ["http://cache.test/v1"]


def test_cached_settings_are_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-request edits (validated model, overlays) must not leak into the cache."""
    monkeypatch.setattr(chat_module, "_settings_cache", {})

    class _Store:
        def get_user_settings(self, user_id: str):
            return {"preferred_model": "stale-model", "tool_toggles": {}}

    first = chat_module._get_cached_settings("u1", _Store())
    first["preferred_model"] = None
    assert chat_module._get_cached_settings("u1", _Store())["preferred_model"] == "stale-model"


def test_chat_clears_invalid_persisted_chat_model_override(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, fake_async_client = client

    class _InvalidPreferredModelStore:
        def __init__(self) -> None:
            self.upsert_calls: list[Dict[str, Any]] = []

        def get_user_settings(self, user_id: str) -> Dict[str, Any] | None:
            _ = user_id
            return {
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "user2-model",
                "preferred_models": {
                    "chat": "user2-model",
                    "embedding": "text-embedding-granite-embedding-278m-multilingual",
                },
                "theme": "auto",
                "language": "en",
            }

        def upsert_user_settings(self, **kwargs) -> Dict[str, Any]:
            self.upsert_calls.append(kwargs)
            return {
                "user_id": kwargs["user_id"],
                "tool_toggles": kwargs["tool_toggles"],
                "rag_config": kwargs["rag_config"],
                "analytics_preferences": kwargs["analytics_preferences"],
                "preferred_model": kwargs["preferred_model"],
                "preferred_models": kwargs["preferred_models"],
                "theme": kwargs["theme"],
                "language": kwargs["language"],
                "updated_at": None,
            }

    async def _invalid_model(*_args, **_kwargs):
        return None, "Preferred model 'user2-model' not found at configured LLM endpoint. Using default."

    invalid_store = _InvalidPreferredModelStore()
    test_client.app.state.settings_store = invalid_store
    chat_module._settings_cache.clear()
    monkeypatch.setattr(chat_module, "_validate_preferred_model", _invalid_model)

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use the default model please",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    assert len(invalid_store.upsert_calls) == 1
    assert invalid_store.upsert_calls[0]["preferred_model"] is None
    assert invalid_store.upsert_calls[0]["preferred_models"] == {
        "embedding": "text-embedding-granite-embedding-278m-multilingual"
    }
    assert fake_async_client.last_request_json["user_settings"]["preferred_model"] is None
    assert response.json()["metadata"]["model_warning"] == (
        "Preferred model 'user2-model' not found at configured LLM endpoint. Using default."
    )


def test_chat_infers_workspace_document_by_bare_filename_from_message(client) -> None:
    test_client, _, fake_async_client = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Please review flatsplit.md in your workspace, improve it, and save it back",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    workspace_docs = fake_async_client.last_request_json["workspace_documents"]
    assert len(workspace_docs) == 1
    assert workspace_docs[0]["filename"] == "flatsplit.md"


def test_chat_auto_saves_single_workspace_document_when_user_requests_save(client) -> None:
    test_client, conversation_store, fake_async_client = client
    fake_async_client.response_content = "# Flat split\nImproved text"

    response = test_client.post(
        "/api/chat",
        json={
            "message": 'Improve workspace document "flatsplit.md" and save it back to the workspace',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["workspace_document_writeback"] == {
        "status": "saved",
        "document_id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        "filename": "flatsplit.md",
        "version": 2,
        "size_bytes": len("# Flat split\nImproved text".encode("utf-8")),
    }
    assert body["metadata"]["documents_used"] == [
        {
            "id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
            "filename": "flatsplit.md",
            "version": 2,
        }
    ]
    stored_doc = test_client.app.state.workspace_document_store.get_document(
        "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        "u1",
    )
    assert stored_doc["content_text"] == "# Flat split\nImproved text"
    assert conversation_store.messages[1]["metadata"]["workspace_document_writeback"]["version"] == 2


def test_chat_infers_workspace_document_from_attachment_filename_for_follow_up_save(client) -> None:
    test_client, conversation_store, fake_async_client = client
    fake_async_client.response_content = "# Flat split\nSecond revision"

    test_client.app.state.conversation_attachment_store._attachments.append(
        {
            "id": "att-doc",
            "conversation_id": "conv-1",
            "user_id": "u1",
            "original_name": "flatsplit.md",
            "mime_type": "text/markdown",
            "size_bytes": 21,
            "extracted_text": "# Flat split\nDraft text",
            "status": "active",
        }
    )

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Please update the document in the workspace",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "attachment_ids": ["att-doc"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    workspace_docs = fake_async_client.last_request_json["workspace_documents"]
    assert len(workspace_docs) == 1
    assert workspace_docs[0]["filename"] == "flatsplit.md"
    assert body["metadata"]["workspace_document_writeback"] == {
        "status": "saved",
        "document_id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        "filename": "flatsplit.md",
        "version": 2,
        "size_bytes": len("# Flat split\nSecond revision".encode("utf-8")),
    }
    assert conversation_store.messages[1]["metadata"]["documents_used"] == [
        {
            "id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
            "filename": "flatsplit.md",
            "version": 2,
        }
    ]


def test_chat_reuses_recent_conversation_document_for_generic_follow_up_save(client) -> None:
    test_client, conversation_store, fake_async_client = client

    fake_async_client.response_content = "# Flat split\nFirst revision"
    first_response = test_client.post(
        "/api/chat",
        json={
            "message": 'Improve workspace document "flatsplit.md" and save it back to the workspace',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )
    assert first_response.status_code == 200
    assert first_response.json()["metadata"]["workspace_document_writeback"]["version"] == 2

    fake_async_client.response_content = "# Flat split\nSecond revision"
    second_response = test_client.post(
        "/api/chat",
        json={
            "message": "Can you elaborate the Integrated Calendar a bit more and save it back?",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert second_response.status_code == 200
    second_body = second_response.json()
    workspace_docs = fake_async_client.last_request_json["workspace_documents"]
    assert len(workspace_docs) == 1
    assert workspace_docs[0]["filename"] == "flatsplit.md"
    assert second_body["metadata"]["workspace_document_writeback"] == {
        "status": "saved",
        "document_id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        "filename": "flatsplit.md",
        "version": 3,
        "size_bytes": len("# Flat split\nSecond revision".encode("utf-8")),
    }
    assert conversation_store.messages[3]["metadata"]["workspace_document_writeback"]["version"] == 3


def test_chat_save_intent_does_not_trigger_on_wokspace_typo(client) -> None:
    test_client, conversation_store, fake_async_client = client

    fake_async_client.response_content = "# Flat split\nFirst revision"
    first_response = test_client.post(
        "/api/chat",
        json={
            "message": 'Improve workspace document "flatsplit.md" and save it back to the workspace',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )
    assert first_response.status_code == 200
    assert first_response.json()["metadata"]["workspace_document_writeback"]["version"] == 2

    fake_async_client.response_content = "# Flat split\nSecond revision"
    second_response = test_client.post(
        "/api/chat",
        json={
            "message": "please save your update to the wokspace",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["metadata"]["workspace_document_writeback"] is None
    assert conversation_store.messages[3]["metadata"]["workspace_document_writeback"] is None


def test_workspace_edit_intent_detection_examples() -> None:
    assert chat_module._has_explicit_workspace_edit_intent("rewrite the uploaded markdown file")
    assert chat_module._has_explicit_workspace_edit_intent("create a revised copy of this attachment")
    assert chat_module._has_explicit_workspace_edit_intent("Bitte überarbeite die Datei im Workspace")

    assert not chat_module._has_explicit_workspace_edit_intent("summarize this document")
    assert not chat_module._has_explicit_workspace_edit_intent("what does this attachment say?")


def test_writeback_records_assistant_source(client) -> None:
    test_client, _conversation_store, fake_async_client = client
    fake_async_client.response_content = "# Flat split\nImproved by AI"

    response = test_client.post(
        "/api/chat",
        json={
            "message": 'Improve workspace document "flatsplit.md" and save it back to the workspace',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["workspace_document_writeback"]["status"] == "saved"
    stored = test_client.app.state.workspace_document_store.get_document(
        "916f651c-fbe5-4df6-a7c9-f156fb96e9fa", "u1"
    )
    assert stored["last_source"] == "assistant"


def test_save_intent_with_only_image_document_warns_and_skips_writeback(client) -> None:
    test_client, _conversation_store, fake_async_client = client
    fake_async_client.response_content = "Here is a description of the image."

    # An image document is in context — images are immutable, so a save must not fire.
    test_client.app.state.workspace_document_store._documents.append(
        {
            "id": "img-1",
            "user_id": "u1",
            "filename": "diagram.png",
            "stored_path": "/tmp/diagram.png",
            "mime_type": "image/png",
            "size_bytes": 100,
            "sha256": "img",
            "content_text": "",
            "version": 1,
        }
    )

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Please save the updated version of this document",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "document_ids": ["img-1"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["workspace_document_writeback"] is None
    assert body["metadata"]["workspace_warning"]


def test_save_intent_with_two_text_documents_warns_and_skips_writeback(client) -> None:
    test_client, _conversation_store, fake_async_client = client
    fake_async_client.response_content = "Some answer."

    test_client.app.state.workspace_document_store._documents.append(
        {
            "id": "doc-2",
            "user_id": "u1",
            "filename": "second.md",
            "stored_path": "/tmp/second.md",
            "mime_type": "text/markdown",
            "size_bytes": 10,
            "sha256": "two",
            "content_text": "# Second",
            "version": 1,
        }
    )

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Please save the updated version of this document",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "document_ids": ["916f651c-fbe5-4df6-a7c9-f156fb96e9fa", "doc-2"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["workspace_document_writeback"] is None
    assert body["metadata"]["workspace_warning"]


def test_writeback_version_conflict_keeps_raw_answer(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, conversation_store, fake_async_client = client
    raw_answer = "SUMMARY:\nImproved it\n\nDOCUMENT:\n# Flat split\nConflict path"
    fake_async_client.response_content = raw_answer

    store = test_client.app.state.workspace_document_store

    def _raise_conflict(**kwargs):
        raise WorkspaceVersionConflictError(99, kwargs.get("expected_version") or 1)

    monkeypatch.setattr(store, "update_document_content", _raise_conflict)

    response = test_client.post(
        "/api/chat",
        json={
            "message": 'Improve workspace document "flatsplit.md" and save it back to the workspace',
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["workspace_document_writeback"] == {
        "status": "conflict",
        "document_id": "916f651c-fbe5-4df6-a7c9-f156fb96e9fa",
        "filename": "flatsplit.md",
    }
    # The model's raw answer is preserved (no "Saved ..." confirmation) and persisted.
    assert body["response"] == raw_answer
    assert conversation_store.messages[1]["content"] == raw_answer


def test_extract_writeback_summary_tolerates_single_newline() -> None:
    content = "SUMMARY:\nTightened the intro.\nDOCUMENT:\n# Doc\nBody"
    assert chat_module._extract_writeback_summary(content) == "Tightened the intro."


def test_workspace_write_revised_requires_explicit_intent(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, _ = client
    monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")

    tracked_intent_denials: list[tuple[str, str, str]] = []

    def _track_intent_denied(profile_name: str, operation: str, reason: str) -> None:
        tracked_intent_denials.append((profile_name, operation, reason))

    monkeypatch.setattr(chat_module, "track_workspace_intent_denied", _track_intent_denied)

    response = test_client.post(
        "/api/chat",
        json={
            "message": "summarize this draft",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "workspace_action": {
                "operation": "write_revised_copy",
                "path": "notes.md",
                "content": "Updated text",
            },
        },
    )

    assert response.status_code == 400
    assert "explicit edit intent" in response.json()["detail"]
    assert tracked_intent_denials == [
        (chat_module.PROFILE_ID, "write_revised_copy", "explicit_edit_intent_required")
    ]


def test_workspace_write_revised_creates_revised_copy(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, _ = client
    monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")

    workspace_dir = test_client.app.state.chat_workspace_manager.get_workspace_dir("conv-1")
    source_file = workspace_dir / "notes.md"
    source_file.write_text("Original text", encoding="utf-8")

    response = test_client.post(
        "/api/chat",
        json={
            "message": "rewrite the uploaded markdown file",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "workspace_action": {
                "operation": "write_revised_copy",
                "path": "notes.md",
                "content": "Updated text",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["workspace_action"]["revised_path"] == "notes.revised.md"
    assert source_file.read_text(encoding="utf-8") == "Original text"
    assert (workspace_dir / "notes.revised.md").read_text(encoding="utf-8") == "Updated text"


def test_workspace_read_rejects_path_escape(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, _ = client
    monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")

    response = test_client.post(
        "/api/chat",
        json={
            "message": "read the workspace file",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "workspace_action": {
                "operation": "read_workspace_file",
                "path": "../secrets.txt",
            },
        },
    )

    assert response.status_code == 400
    assert "escapes" in response.json()["detail"].lower()

def _doc_resolution_request(conv_store, doc_store):
    from types import SimpleNamespace

    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(conversation_store=conv_store, workspace_document_store=doc_store)
        )
    )


def _doc(doc_id: str, filename: str) -> Dict[str, Any]:
    return {"id": doc_id, "user_id": "u1", "filename": filename, "content_text": "x", "version": 1,
            "mime_type": "text/plain"}


def test_explicit_document_does_not_pull_in_history_document() -> None:
    from backend.routers.chat import ChatRequest, _resolve_workspace_documents

    conv = FakeConversationStore()
    conv.messages.append({"conversation_id": "c1", "role": "assistant",
                          "metadata": {"documents_used": [{"id": "doc-a"}]}})
    docs = FakeWorkspaceDocumentStore([_doc("doc-a", "a.md"), _doc("doc-b", "b.md")])
    body = ChatRequest(message="improve this and save", conversation_id="c1", document_ids=["doc-b"])

    resolved = _resolve_workspace_documents(_doc_resolution_request(conv, docs), body, "u1")

    assert [d["id"] for d in resolved] == ["doc-b"]


def test_history_document_used_when_nothing_named() -> None:
    from backend.routers.chat import ChatRequest, _resolve_workspace_documents

    conv = FakeConversationStore()
    conv.messages.append({"conversation_id": "c1", "role": "assistant",
                          "metadata": {"documents_used": [{"id": "doc-a"}]}})
    docs = FakeWorkspaceDocumentStore([_doc("doc-a", "a.md")])
    body = ChatRequest(message="improve it again and save", conversation_id="c1")

    resolved = _resolve_workspace_documents(_doc_resolution_request(conv, docs), body, "u1")

    assert [d["id"] for d in resolved] == ["doc-a"]


def test_history_inference_uses_newest_messages() -> None:
    from backend.routers.chat import ChatRequest, _resolve_workspace_documents

    conv = FakeConversationStore()
    conv.messages.append({"conversation_id": "c1", "role": "assistant",
                          "metadata": {"documents_used": [{"id": "doc-old"}]}})
    for _ in range(60):
        conv.messages.append({"conversation_id": "c1", "role": "user", "metadata": {}})
    conv.messages.append({"conversation_id": "c1", "role": "assistant",
                          "metadata": {"documents_used": [{"id": "doc-new"}]}})
    docs = FakeWorkspaceDocumentStore([_doc("doc-old", "old.md"), _doc("doc-new", "new.md")])
    body = ChatRequest(message="improve it again and save", conversation_id="c1")

    resolved = _resolve_workspace_documents(_doc_resolution_request(conv, docs), body, "u1")

    assert [d["id"] for d in resolved] == ["doc-new"]
