from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from backend.db import ConversationAttachmentStore, _normalize_attachment_row


class FakeCursor:
    def __init__(self, *, fetchone_result=None, fetchall_result=None, rowcount: int = 0) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.rowcount = rowcount
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, statement, params=None) -> None:
        self.executed.append((statement, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self) -> None:
        self.committed = True


class FakeDbPool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


def test_normalize_attachment_row_serializes_timestamps() -> None:
    created_at = datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)
    expires_at = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)

    normalized = _normalize_attachment_row(
        {
            "id": "att-1",
            "conversation_id": "conv-1",
            "user_id": "u1",
            "original_name": "notes.md",
            "stored_path": "/tmp/notes.md",
            "mime_type": "text/markdown",
            "size_bytes": 42,
            "sha256": "abc123",
            "extracted_text": "hello",
            "status": "active",
            "created_at": created_at,
            "expires_at": expires_at,
        }
    )

    assert normalized["id"] == "att-1"
    assert normalized["created_at"] == created_at.isoformat()
    assert normalized["expires_at"] == expires_at.isoformat()


def test_create_attachment_returns_normalized_row_and_commits() -> None:
    row = {
        "id": "att-1",
        "conversation_id": "conv-1",
        "user_id": "u1",
        "original_name": "notes.md",
        "stored_path": "/tmp/notes.md",
        "mime_type": "text/markdown",
        "size_bytes": 42,
        "sha256": "abc123",
        "extracted_text": "hello",
        "status": "active",
        "created_at": datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc),
        "expires_at": None,
    }
    cursor = FakeCursor(fetchone_result=row)
    connection = FakeConnection(cursor)
    store = ConversationAttachmentStore(FakeDbPool(connection))

    created = store.create_attachment(
        attachment_id="att-1",
        conversation_id="conv-1",
        user_id="u1",
        original_name="notes.md",
        stored_path="/tmp/notes.md",
        mime_type="text/markdown",
        size_bytes=42,
        sha256="abc123",
        extracted_text="hello",
    )

    assert created["id"] == "att-1"
    assert created["original_name"] == "notes.md"
    assert connection.committed is True
    assert "INSERT INTO chat_document_refs" in cursor.executed[0][0]


def test_get_attachments_by_ids_preserves_requested_order() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {
                "id": "att-2",
                "conversation_id": "conv-1",
                "user_id": "u1",
                "original_name": "b.md",
                "stored_path": "/tmp/b.md",
                "mime_type": "text/markdown",
                "size_bytes": 2,
                "sha256": "bbb",
                "extracted_text": "b",
                "status": "active",
                "created_at": None,
                "expires_at": None,
            },
            {
                "id": "att-1",
                "conversation_id": "conv-1",
                "user_id": "u1",
                "original_name": "a.md",
                "stored_path": "/tmp/a.md",
                "mime_type": "text/markdown",
                "size_bytes": 1,
                "sha256": "aaa",
                "extracted_text": "a",
                "status": "active",
                "created_at": None,
                "expires_at": None,
            },
        ]
    )
    store = ConversationAttachmentStore(FakeDbPool(FakeConnection(cursor)))

    attachments = store.get_attachments_by_ids("conv-1", ["att-1", "att-2"])

    assert [attachment["id"] for attachment in attachments] == ["att-1", "att-2"]
    assert "id = ANY(%s)" in cursor.executed[0][0]


def test_mark_attachment_deleted_returns_true_when_row_updated() -> None:
    cursor = FakeCursor(rowcount=1)
    connection = FakeConnection(cursor)
    store = ConversationAttachmentStore(FakeDbPool(connection))

    deleted = store.mark_attachment_deleted("att-1", conversation_id="conv-1")

    assert deleted is True
    assert connection.committed is True
    assert "DELETE FROM chat_document_refs" in cursor.executed[0][0]


def test_conversation_belongs_to_user_returns_boolean() -> None:
    cursor = FakeCursor(fetchone_result=(1,))
    store = ConversationAttachmentStore(FakeDbPool(FakeConnection(cursor)))

    assert store.conversation_belongs_to_user("conv-1", "u1") is True
