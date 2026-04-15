from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.attachments import UserWorkspaceFileManager, WorkspaceManagerConfig
from backend.rate_limit import limiter
from backend.routers import chat as chat_module
from backend.routers.chat import router as chat_router
from backend.routers.workspace import router as workspace_router


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


class FakeAttachmentStore:
    def get_attachments_by_ids(self, conversation_id: str, attachment_ids: list[str], include_inactive: bool = False):
        return []


class FakeWorkspaceDocumentStore:
    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, Any]] = {}

    def create_document(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        stored_path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        content_text: str,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "id": document_id,
            "user_id": user_id,
            "filename": filename,
            "stored_path": stored_path,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "content_text": content_text,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        self._docs[document_id] = row
        return dict(row)

    def get_document(self, document_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        row = self._docs.get(document_id)
        if not row or row["user_id"] != user_id:
            return None
        return dict(row)

    def list_documents(self, user_id: str, limit: int = 200, offset: int = 0) -> list[Dict[str, Any]]:
        docs = [dict(row) for row in self._docs.values() if row["user_id"] == user_id]
        docs.sort(key=lambda row: row["updated_at"], reverse=True)
        return docs[offset : offset + limit]

    def get_documents_by_ids(self, user_id: str, document_ids: list[str]) -> list[Dict[str, Any]]:
        docs: list[Dict[str, Any]] = []
        for document_id in document_ids:
            row = self._docs.get(document_id)
            if row and row["user_id"] == user_id:
                docs.append(dict(row))
        return docs

    def update_document_content(
        self,
        document_id: str,
        user_id: str,
        content_text: str,
        size_bytes: int,
        sha256: str,
        expected_version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        row = self._docs.get(document_id)
        if not row or row["user_id"] != user_id:
            return None
        if expected_version is not None and row["version"] != expected_version:
            return None

        row["content_text"] = content_text
        row["size_bytes"] = size_bytes
        row["sha256"] = sha256
        row["version"] += 1
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        return dict(row)

    def delete_document(self, document_id: str, user_id: str) -> bool:
        row = self._docs.get(document_id)
        if not row or row["user_id"] != user_id:
            return False
        del self._docs[document_id]
        return True


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AUTH_USERNAME", "u1")
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
                "messages": [{"role": "assistant", "content": "workspace-aware answer"}],
                "tokens_used": 20,
                "input_tokens": 11,
                "output_tokens": 9,
                "model_used": "test-model",
                "tool_results": {},
                "sources": [],
            }

    class _FakeAsyncClient:
        last_request_json: Dict[str, Any] | None = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
            _FakeAsyncClient.last_request_json = json
            return _FakeResponse()

    monkeypatch.setattr(chat_module.LANGGRAPH_CIRCUIT_BREAKER, "call", _fake_cb_call)
    monkeypatch.setattr(
        chat_module,
        "httpx",
        type(
            "_HttpxModule",
            (),
            {"AsyncClient": _FakeAsyncClient, "RequestError": Exception, "HTTPStatusError": Exception},
        ),
    )

    app = FastAPI(lifespan=_noop_lifespan)
    app.state.limiter = limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(workspace_router)
    app.include_router(chat_router)

    conversation_store = FakeConversationStore()
    conversation_store.create_conversation("conv-1", "u1")

    app.state.conversation_store = conversation_store
    app.state.settings_store = FakeSettingsStore()
    app.state.conversation_attachment_store = FakeAttachmentStore()
    app.state.workspace_document_store = FakeWorkspaceDocumentStore()
    app.state.user_workspace_file_manager = UserWorkspaceFileManager(
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24)
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, _FakeAsyncClient


def test_workspace_upload_rejects_binary_content(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/api/workspace/documents",
        files={"file": ("payload.md", b"abc\x00def", "text/markdown")},
    )

    assert response.status_code == 400
    assert "Binary content" in response.json()["detail"]


def test_workspace_upload_rejects_unsupported_extension(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/api/workspace/documents",
        files={"file": ("payload.pdf", b"not allowed", "application/pdf")},
    )

    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_workspace_upload_rejects_disallowed_mime_type(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/api/workspace/documents",
        files={"file": ("payload.md", b"hello", "application/json")},
    )

    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]


def test_workspace_operations_return_404_for_missing_document(client) -> None:
    test_client, _ = client

    get_response = test_client.get("/api/workspace/documents/missing-doc")
    assert get_response.status_code == 404

    update_response = test_client.put(
        "/api/workspace/documents/missing-doc",
        files={"file": ("notes.md", b"updated", "text/markdown")},
    )
    assert update_response.status_code == 404

    delete_response = test_client.delete("/api/workspace/documents/missing-doc")
    assert delete_response.status_code == 404

    download_response = test_client.get("/api/workspace/documents/missing-doc/download")
    assert download_response.status_code == 404


def test_chat_rejects_invalid_workspace_document_ids(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/api/chat",
        json={
            "message": "Use this missing document",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "document_ids": ["missing-doc"],
        },
    )

    assert response.status_code == 400
    assert "Invalid document_ids" in response.json()["detail"]


def test_workspace_requires_api_token_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.setenv("CHAT_ACCESS_TOKEN", "secret-token")

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
    app.include_router(workspace_router)

    app.state.workspace_document_store = FakeWorkspaceDocumentStore()
    app.state.user_workspace_file_manager = UserWorkspaceFileManager(
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24)
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        unauthorized = test_client.get("/api/workspace/documents")
        assert unauthorized.status_code == 401

        authorized = test_client.get(
            "/api/workspace/documents",
            headers={"x-chat-token": "secret-token"},
        )
        assert authorized.status_code == 200


def test_workspace_end_to_end_upload_attach_overwrite_reopen_download(client) -> None:
    test_client, fake_async_client = client

    upload_response = test_client.post(
        "/api/workspace/documents",
        files={"file": ("notes.md", b"Original content", "text/markdown")},
    )
    assert upload_response.status_code == 201
    uploaded_document = upload_response.json()["document"]
    document_id = uploaded_document["id"]

    chat_response = test_client.post(
        "/api/chat",
        json={
            "message": "Use this document",
            "user_id": "u1",
            "language": "en",
            "conversation_id": "conv-1",
            "document_ids": [document_id],
        },
    )
    assert chat_response.status_code == 200
    assert fake_async_client.last_request_json is not None
    forwarded_docs = fake_async_client.last_request_json["workspace_documents"]
    assert [doc["id"] for doc in forwarded_docs] == [document_id]
    assert forwarded_docs[0]["content_text"] == "Original content"

    updated_content = b"Edited content for QA"
    update_response = test_client.put(
        f"/api/workspace/documents/{document_id}",
        files={"file": ("notes.md", updated_content, "text/markdown")},
    )
    assert update_response.status_code == 200
    assert update_response.json()["document"]["version"] == 2
    assert update_response.json()["document"]["sha256"] == hashlib.sha256(updated_content).hexdigest()

    reopen_response = test_client.get(f"/api/workspace/documents/{document_id}")
    assert reopen_response.status_code == 200
    reopened = reopen_response.json()
    assert reopened["content_text"] == "Edited content for QA"

    download_response = test_client.get(f"/api/workspace/documents/{document_id}/download")
    assert download_response.status_code == 200
    assert download_response.content == updated_content
    assert "attachment; filename=\"notes.md\"" in download_response.headers.get("content-disposition", "")


def test_workspace_update_recreates_missing_file_from_existing_metadata(client) -> None:
    test_client, _ = client

    upload_response = test_client.post(
        "/api/workspace/documents",
        files={"file": ("draft.md", b"Original draft", "text/markdown")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document"]["id"]

    existing_doc = test_client.app.state.workspace_document_store.get_document(document_id, "u1")
    assert existing_doc is not None
    stored_path = Path(existing_doc["stored_path"])
    assert stored_path.exists()

    stored_path.unlink()
    assert not stored_path.exists()

    update_response = test_client.put(
        f"/api/workspace/documents/{document_id}",
        files={"file": ("draft.md", b"Recovered content", "text/markdown")},
    )
    assert update_response.status_code == 200
    assert update_response.json()["document"]["version"] == 2
    assert stored_path.exists()
    assert stored_path.read_text(encoding="utf-8") == "Recovered content"
