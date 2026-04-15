"""Tests for conversations REST API and ConversationStore.

These tests use an in-process SQLite-like mock via psycopg2 against a real
PostgreSQL database (requires the docker compose postgres service to be running)
or can be run with a lightweight mock.

For CI / local fast-run we mock the ConversationStore instead of hitting the DB.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.attachments import AttachmentManagerConfig, ChatAttachmentManager, ChatWorkspaceManager, UserWorkspaceFileManager, WorkspaceManagerConfig
from backend.fastapi_app import _run_workspace_startup_cleanup

# ── Minimal ConversationStore mock ────────────────────────────────────


class FakeConversationStore:
    """In-memory ConversationStore that mirrors the real API."""

    def __init__(self) -> None:
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._messages: List[Dict[str, Any]] = []
        self._attachments: List[Dict[str, Any]] = []
        self._msg_id_counter = 0

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str = "New conversation",
        language: str = "en",
        fork_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        conv = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "language": language,
            "fork_name": fork_name,
            "archived": False,
            "pinned": False,
            "metadata": {},
            "last_message": None,
            "message_count": None,
            "created_at": now,
            "updated_at": now,
        }
        self._conversations[conversation_id] = conv
        return conv

    def list_conversations(
        self,
        user_id: str,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        result = [
            c for c in self._conversations.values()
            if c["user_id"] == user_id and (include_archived or not c["archived"])
        ]
        result.sort(key=lambda c: (not c["pinned"], c["updated_at"]), reverse=False)
        total = len(result)
        return result[offset : offset + limit], total

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def update_conversation(
        self, conversation_id: str, **fields: Any
    ) -> Optional[Dict[str, Any]]:
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None
        for k, v in fields.items():
            if k in {"title", "archived", "pinned", "language", "metadata"}:
                conv[k] = v
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        return conv

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._messages = [
                m for m in self._messages if m["conversation_id"] != conversation_id
            ]
            return True
        return False

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        model_name: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        tools_used: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        self._msg_id_counter += 1
        msg = {
            "id": self._msg_id_counter,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "tokens_used": tokens_used,
            "model_name": model_name,
            "response_time_ms": response_time_ms,
            "tools_used": tools_used,
            "feedback": None,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._messages.append(msg)
        return msg

    def get_messages(
        self, conversation_id: str, limit: int = 500, offset: int = 0
    ) -> List[Dict[str, Any]]:
        msgs = [m for m in self._messages if m["conversation_id"] == conversation_id]
        msgs.sort(key=lambda m: m["created_at"])
        return msgs[offset : offset + limit]

    def update_message_feedback(
        self, message_id: int, feedback: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        for m in self._messages:
            if m["id"] == message_id:
                m["feedback"] = feedback
                return m
        return None

    def search_messages(
        self, user_id: str, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        results = []
        for m in self._messages:
            conv = self._conversations.get(m["conversation_id"])
            if conv and conv["user_id"] == user_id and query.lower() in m["content"].lower():
                results.append(
                    {
                        "message_id": m["id"],
                        "conversation_id": m["conversation_id"],
                        "conversation_title": conv["title"],
                        "role": m["role"],
                        "content": m["content"],
                        "created_at": m["created_at"],
                    }
                )
        return results[:limit]

    def export_conversation(
        self, conversation_id: str, fmt: str = "json"
    ) -> Optional[Any]:
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None
        msgs = self.get_messages(conversation_id, limit=10000)
        if fmt == "markdown":
            lines = [f"# {conv['title']}\n"]
            for m in msgs:
                speaker = "**You**" if m["role"] == "user" else "**AI Agent**"
                lines.append(f"{speaker}: {m['content']}\n")
            return "\n".join(lines)
        return {"conversation": conv, "messages": msgs}

    def create_attachment(
        self,
        attachment_id: str,
        conversation_id: str,
        user_id: str,
        original_name: str,
        stored_path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        extracted_text: str,
        expires_at=None,
    ) -> Dict[str, Any]:
        attachment = {
            "id": attachment_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "original_name": original_name,
            "stored_path": stored_path,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "extracted_text": extracted_text,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at.isoformat() if hasattr(expires_at, "isoformat") else expires_at,
        }
        self._attachments.append(attachment)
        return attachment

    def list_attachments(
        self,
        conversation_id: str,
        include_inactive: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        attachments = [
            attachment for attachment in self._attachments
            if attachment["conversation_id"] == conversation_id
            and (include_inactive or attachment["status"] == "active")
        ]
        return attachments[offset : offset + limit]

    def get_attachment(
        self,
        attachment_id: str,
        conversation_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        for attachment in self._attachments:
            if attachment["id"] != attachment_id:
                continue
            if conversation_id is not None and attachment["conversation_id"] != conversation_id:
                continue
            if not include_inactive and attachment["status"] != "active":
                continue
            return attachment
        return None

    def mark_attachment_deleted(
        self,
        attachment_id: str,
        conversation_id: Optional[str] = None,
    ) -> bool:
        attachment = self.get_attachment(
            attachment_id,
            conversation_id=conversation_id,
            include_inactive=True,
        )
        if not attachment or attachment["status"] != "active":
            return False
        attachment["status"] = "deleted"
        return True

    def get_attachments_by_ids(
        self,
        conversation_id: str,
        attachment_ids: list[str],
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        rows_by_id = {
            attachment["id"]: attachment
            for attachment in self._attachments
            if attachment["conversation_id"] == conversation_id
            and (include_inactive or attachment["status"] == "active")
        }
        return [rows_by_id[attachment_id] for attachment_id in attachment_ids if attachment_id in rows_by_id]


class FakeWorkspaceStore:
    def __init__(self) -> None:
        self._workspaces: Dict[str, Dict[str, Any]] = {}
        self._operations: List[Dict[str, Any]] = []

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

    def get_workspace(self, conversation_id: str, include_inactive: bool = False):
        workspace = self._workspaces.get(conversation_id)
        if not workspace:
            return None
        if not include_inactive and workspace.get("status") != "active":
            return None
        return dict(workspace)

    def touch_workspace(self, conversation_id: str, expires_at=None):
        workspace = self._workspaces.get(conversation_id)
        if not workspace:
            return None
        workspace["updated_at"] = datetime.now(timezone.utc).isoformat()
        workspace["last_activity_at"] = workspace["updated_at"]
        workspace["status"] = "active"
        if expires_at is not None:
            workspace["expires_at"] = expires_at.isoformat() if hasattr(expires_at, "isoformat") else expires_at
        return dict(workspace)

    def list_expired_workspaces(self, reference_time=None, limit: int = 500):
        effective_reference = reference_time or datetime.now(timezone.utc)
        results = []
        for workspace in self._workspaces.values():
            expires_at = workspace.get("expires_at")
            if not expires_at or workspace.get("status") == "deleted":
                continue
            expires_dt = datetime.fromisoformat(expires_at) if isinstance(expires_at, str) else expires_at
            if expires_dt and expires_dt <= effective_reference:
                results.append(dict(workspace))
        return results[:limit]

    def delete_workspace_record(self, conversation_id: str) -> bool:
        return self._workspaces.pop(conversation_id, None) is not None

    def log_workspace_operation(self, **kwargs):
        entry = {
            "id": len(self._operations) + 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._operations.append(entry)
        return entry


class FakeWorkspaceDocumentStore:
    def __init__(self) -> None:
        self._documents: Dict[str, Dict[str, Any]] = {}

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
        document = {
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
        self._documents[document_id] = document
        return dict(document)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def fake_store():
    return FakeConversationStore()


@pytest.fixture()
def client(fake_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a FastAPI test client with mocked stores (no DB required)."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from backend.routers.conversations import router as conversations_router

    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=_noop_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(conversations_router)

    # Inject fake store into app state
    app.state.conversation_store = fake_store
    app.state.conversation_attachment_store = fake_store
    attachment_manager = ChatAttachmentManager(
        AttachmentManagerConfig(
            root_dir=tmp_path / "chat-workspaces",
            max_file_bytes=128,
            retention_hours=24,
        )
    )
    app.state.chat_attachment_manager = attachment_manager
    app.state.chat_workspace_manager = ChatWorkspaceManager(
        attachment_manager=attachment_manager,
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24),
    )
    app.state.conversation_workspace_store = FakeWorkspaceStore()
    app.state.workspace_document_store = FakeWorkspaceDocumentStore()
    app.state.user_workspace_file_manager = UserWorkspaceFileManager(
        attachment_manager=attachment_manager,
        config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24),
    )

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────


class TestCreateConversation:
    def test_create_returns_201(self, client):
        resp = client.post(
            "/api/conversations",
            json={"user_id": "u1", "title": "Test conv", "language": "en"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["title"] == "Test conv"
        assert body["user_id"] == "u1"
        assert body["archived"] is False
        assert body["pinned"] is False

    def test_create_default_values(self, client):
        resp = client.post("/api/conversations", json={})
        assert resp.status_code == 201
        body = resp.json()
        assert body["title"] == "New conversation"
        assert body["user_id"] == "u1"


class TestListConversations:
    def test_list_empty(self, client):
        resp = client.get("/api/conversations?user_id=u1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["conversations"] == []
        assert body["total"] == 0

    def test_list_returns_created(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "First")
        fake_store.create_conversation("c2", "u1", "Second")
        resp = client.get("/api/conversations?user_id=u1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["conversations"]) == 2

    def test_list_excludes_archived(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Live")
        fake_store.create_conversation("c2", "u1", "Old")
        fake_store.update_conversation("c2", archived=True)
        resp = client.get("/api/conversations?user_id=u1")
        assert resp.json()["total"] == 1

    def test_list_includes_archived(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Live")
        fake_store.create_conversation("c2", "u1", "Old")
        fake_store.update_conversation("c2", archived=True)
        resp = client.get("/api/conversations?user_id=u1&include_archived=true")
        assert resp.json()["total"] == 2


class TestGetConversation:
    def test_get_existing(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Hello")
        fake_store.add_message("c1", "user", "hi")
        resp = client.get("/api/conversations/c1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["conversation"]["title"] == "Hello"
        assert len(body["messages"]) == 1

    def test_get_nonexistent_404(self, client):
        resp = client.get("/api/conversations/nope")
        assert resp.status_code == 404


class TestUpdateConversation:
    def test_rename(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Old")
        resp = client.patch("/api/conversations/c1", json={"title": "New title"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New title"

    def test_pin(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Test")
        resp = client.patch("/api/conversations/c1", json={"pinned": True})
        assert resp.json()["pinned"] is True

    def test_archive(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Test")
        resp = client.patch("/api/conversations/c1", json={"archived": True})
        assert resp.json()["archived"] is True

    def test_update_nonexistent_404(self, client):
        resp = client.patch("/api/conversations/nope", json={"title": "X"})
        assert resp.status_code == 404


class TestDeleteConversation:
    def test_delete_existing(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Bye")
        resp = client.delete("/api/conversations/c1")
        assert resp.status_code == 204

    def test_delete_nonexistent_404(self, client):
        resp = client.delete("/api/conversations/nope")
        assert resp.status_code == 404


class TestMessages:
    def test_add_message(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        resp = client.post(
            "/api/conversations/c1/messages",
            json={"role": "user", "content": "Hello world"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["role"] == "user"
        assert body["content"] == "Hello world"

    def test_add_assistant_with_metadata(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        resp = client.post(
            "/api/conversations/c1/messages",
            json={
                "role": "assistant",
                "content": "Hi!",
                "tokens_used": 42,
                "model_name": "llama-3.1",
                "response_time_ms": 1200,
                "tools_used": [{"name": "web_search", "status": "success"}],
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["tokens_used"] == 42
        assert body["model_name"] == "llama-3.1"

    def test_add_to_nonexistent_conv_404(self, client):
        resp = client.post(
            "/api/conversations/nope/messages",
            json={"role": "user", "content": "hello"},
        )
        assert resp.status_code == 404

    def test_feedback(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        fake_store.add_message("c1", "assistant", "answer")
        msg_id = fake_store._messages[0]["id"]
        resp = client.patch(
            f"/api/conversations/c1/messages/{msg_id}/feedback",
            json={"feedback": "up"},
        )
        assert resp.status_code == 200
        assert resp.json()["feedback"] == "up"

    def test_feedback_clear(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        fake_store.add_message("c1", "assistant", "answer")
        msg_id = fake_store._messages[0]["id"]
        # Set then clear
        client.patch(
            f"/api/conversations/c1/messages/{msg_id}/feedback",
            json={"feedback": "down"},
        )
        resp = client.patch(
            f"/api/conversations/c1/messages/{msg_id}/feedback",
            json={"feedback": None},
        )
        assert resp.json()["feedback"] is None


class TestSearch:
    def test_search_finds_match(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        fake_store.add_message("c1", "user", "What is tumor marker")
        fake_store.add_message("c1", "assistant", "Tumor markers are proteins")
        resp = client.get("/api/conversations/search?user_id=u1&q=tumor")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2

    def test_search_no_results(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Chat")
        fake_store.add_message("c1", "user", "Hello")
        resp = client.get("/api/conversations/search?user_id=u1&q=nonexistent")
        assert resp.json() == []


class TestExport:
    def test_export_json(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Export test")
        fake_store.add_message("c1", "user", "Hello")
        resp = client.get("/api/conversations/c1/export?fmt=json")
        assert resp.status_code == 200
        body = resp.json()
        assert "conversation" in body
        assert "messages" in body

    def test_export_markdown(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Export test")
        fake_store.add_message("c1", "user", "Hello")
        resp = client.get("/api/conversations/c1/export?fmt=markdown")
        assert resp.status_code == 200
        assert "Export test" in resp.text

    def test_export_nonexistent_404(self, client):
        resp = client.get("/api/conversations/nope/export")
        assert resp.status_code == 404


class TestAttachments:
    def test_upload_attachment(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Upload test")

        resp = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u1"},
            files={"file": ("notes.md", b"# Notes\nHello\n", "text/markdown")},
        )

        assert resp.status_code == 201
        body = resp.json()["attachment"]
        assert body["conversation_id"] == "c1"
        assert body["original_name"] == "notes.md"
        assert body["mime_type"] == "text/markdown"
        assert body["status"] == "active"
        assert len(fake_store._attachments) == 1

    def test_upload_attachment_normalizes_requested_user_to_single_user(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Upload test")

        resp = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u2"},
            files={"file": ("notes.md", b"# Notes\nHello\n", "text/markdown")},
        )

        assert resp.status_code == 201
        assert resp.json()["attachment"]["user_id"] == "u1"

    def test_list_attachments(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Upload test")
        fake_store.create_attachment(
            attachment_id="att-1",
            conversation_id="c1",
            user_id="u1",
            original_name="notes.md",
            stored_path="/tmp/notes.md",
            mime_type="text/markdown",
            size_bytes=12,
            sha256="abc",
            extracted_text="# Notes",
        )

        resp = client.get("/api/conversations/c1/attachments")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["attachments"]) == 1
        assert body["attachments"][0]["id"] == "att-1"

    def test_delete_attachment(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Upload test")
        upload = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u1"},
            files={"file": ("notes.txt", b"Hello", "text/plain")},
        )
        attachment_id = upload.json()["attachment"]["id"]

        resp = client.delete(f"/api/conversations/c1/attachments/{attachment_id}")

        assert resp.status_code == 204
        assert fake_store.get_attachment(attachment_id, conversation_id="c1", include_inactive=True)["status"] == "deleted"

    def test_upload_attachment_persists_in_user_workspace_only(self, client, fake_store, tmp_path: Path):
        fake_store.create_conversation("c1", "u1", "Upload test")

        resp = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u1"},
            files={"file": ("notes.md", b"# Notes\nHello\n", "text/markdown")},
        )

        assert resp.status_code == 201
        attachment = resp.json()["attachment"]
        stored_path = Path(fake_store.get_attachment(attachment["id"], conversation_id="c1")["stored_path"])
        assert stored_path.exists()
        assert "user-workspaces" in stored_path.parts
        assert "attachments" not in stored_path.parts

    def test_delete_missing_attachment_returns_404(self, client, fake_store):
        fake_store.create_conversation("c1", "u1", "Upload test")

        resp = client.delete("/api/conversations/c1/attachments/missing")

        assert resp.status_code == 404


class TestWorkspace:
    def test_materialize_attachment_creates_workspace_file(self, client, fake_store, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")
        fake_store.create_conversation("c1", "u1", "Workspace test")

        upload = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u1"},
            files={"file": ("notes.md", b"# Notes\nHello\n", "text/markdown")},
        )
        attachment_id = upload.json()["attachment"]["id"]

        resp = client.post(
            "/api/conversations/c1/workspace/materialize",
            json={"attachment_id": attachment_id},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["file"]["relative_path"] == "notes.md"
        assert body["workspace"]["conversation_id"] == "c1"

    def test_get_workspace_lists_materialized_files(self, client, fake_store, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHAT_WORKSPACE_ENABLED", "true")
        fake_store.create_conversation("c1", "u1", "Workspace test")

        upload = client.post(
            "/api/conversations/c1/attachments",
            data={"user_id": "u1"},
            files={"file": ("draft.txt", b"Hello", "text/plain")},
        )
        attachment_id = upload.json()["attachment"]["id"]
        client.post(
            "/api/conversations/c1/workspace/materialize",
            json={"attachment_id": attachment_id},
        )

        resp = client.get("/api/conversations/c1/workspace")

        assert resp.status_code == 200
        files = resp.json()["files"]
        assert len(files) == 1
        assert files[0]["relative_path"] == "draft.txt"


class TestWorkspaceCleanup:
    def test_startup_cleanup_removes_expired_workspace(self, tmp_path: Path):
        from fastapi import FastAPI

        app = FastAPI()
        attachment_manager = ChatAttachmentManager(
            AttachmentManagerConfig(root_dir=tmp_path / "chat-workspaces", max_file_bytes=128, retention_hours=24)
        )
        workspace_manager = ChatWorkspaceManager(
            attachment_manager=attachment_manager,
            config=WorkspaceManagerConfig(root_dir=tmp_path / "chat-workspaces", retention_hours=24),
        )
        workspace_store = FakeWorkspaceStore()

        workspace_dir = workspace_manager.get_workspace_dir("expired-conv")
        (workspace_dir / "old.md").write_text("stale", encoding="utf-8")
        workspace_store.upsert_workspace(
            conversation_id="expired-conv",
            user_id="u1",
            root_path=str(workspace_dir),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        app.state.chat_workspace_manager = workspace_manager
        app.state.conversation_workspace_store = workspace_store

        _run_workspace_startup_cleanup(app)

        assert not workspace_dir.exists()
        assert workspace_store.get_workspace("expired-conv", include_inactive=True) is None
        assert any(op["operation"] == "cleanup_workspace" for op in workspace_store._operations)
