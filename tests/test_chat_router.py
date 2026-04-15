from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.attachments import AttachmentManagerConfig, ChatAttachmentManager, ChatWorkspaceManager, UserWorkspaceFileManager, WorkspaceManagerConfig
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

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def add_message(self, **kwargs) -> Dict[str, Any]:
        self.messages.append(kwargs)
        return kwargs

    def get_messages(self, conversation_id: str, limit: int = 500, offset: int = 0) -> list[Dict[str, Any]]:
        rows = [row for row in self.messages if row.get("conversation_id") == conversation_id]
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
    ):
        _ = expected_version
        for row in self._documents:
            if row["id"] == document_id and row["user_id"] == user_id:
                row["content_text"] = content_text
                row["size_bytes"] = size_bytes
                row["sha256"] = sha256
                row["version"] = int(row.get("version", 0)) + 1
                return row
        return None


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
        "workspace_document_writeback": None,
        "profile_id": "safety",
    }


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

    def _track_mismatch(*, fork_name: str, active_profile_id: str, reported_profile_id: str) -> None:
        tracked.append((fork_name, active_profile_id, reported_profile_id))

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

    def _track_mismatch(*, fork_name: str, active_profile_id: str, reported_profile_id: str) -> None:
        tracked.append((fork_name, active_profile_id, reported_profile_id))

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


def test_workspace_write_revised_requires_explicit_intent(client, monkeypatch: pytest.MonkeyPatch) -> None:
    test_client, _, _ = client
    monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")

    tracked_intent_denials: list[tuple[str, str, str]] = []

    def _track_intent_denied(fork_name: str, operation: str, reason: str) -> None:
        tracked_intent_denials.append((fork_name, operation, reason))

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