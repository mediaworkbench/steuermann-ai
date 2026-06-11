from __future__ import annotations

import pytest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, cast

from backend.db import WorkspaceDocumentStore, WorkspaceVersionConflictError, _normalize_workspace_document_row


class FakeCursor:
    def __init__(
        self,
        *,
        fetchone_result=None,
        fetchone_results: list | None = None,
        fetchall_result=None,
        rowcount: int = 0,
    ) -> None:
        self.fetchone_result = fetchone_result
        # Queue of results for multi-call fetchone (e.g. SELECT FOR UPDATE then UPDATE RETURNING).
        # When set, pops from the front on each call; falls back to fetchone_result when exhausted.
        self._fetchone_queue: list | None = list(fetchone_results) if fetchone_results is not None else None
        self.fetchall_result = fetchall_result or []
        self.rowcount = rowcount
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, statement, params=None) -> None:
        self.executed.append((statement, params))

    def fetchone(self):
        if self._fetchone_queue is not None:
            return self._fetchone_queue.pop(0) if self._fetchone_queue else None
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
        self.rolled_back = False

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class FakeDbPool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


def test_normalize_workspace_document_row_serializes_timestamps() -> None:
    created_at = datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc)
    updated_at = datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc)

    normalized = _normalize_workspace_document_row(
        {
            "id": "doc-1",
            "user_id": "u1",
            "filename": "notes.md",
            "stored_path": "/tmp/u1/notes.md",
            "mime_type": "text/markdown",
            "size_bytes": 10,
            "sha256": "abc",
            "content_text": "hello",
            "version": 1,
            "created_at": created_at,
            "updated_at": updated_at,
        }
    )

    assert normalized["id"] == "doc-1"
    assert normalized["created_at"] == created_at.isoformat()
    assert normalized["updated_at"] == updated_at.isoformat()


def test_create_document_returns_normalized_row_and_commits() -> None:
    row = {
        "id": "doc-1",
        "user_id": "u1",
        "filename": "notes.md",
        "stored_path": "/tmp/u1/notes.md",
        "mime_type": "text/markdown",
        "size_bytes": 42,
        "sha256": "abc123",
        "content_text": "hello",
        "version": 1,
        "created_at": datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
    }
    cursor = FakeCursor(fetchone_result=row)
    connection = FakeConnection(cursor)
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(connection)))

    created = store.create_document(
        document_id="doc-1",
        user_id="u1",
        filename="notes.md",
        stored_path="/tmp/u1/notes.md",
        mime_type="text/markdown",
        size_bytes=42,
        sha256="abc123",
        content_text="hello",
    )

    assert created["id"] == "doc-1"
    assert created["filename"] == "notes.md"
    assert connection.committed is True
    assert "INSERT INTO workspace_documents" in cursor.executed[0][0]


def test_get_documents_by_ids_preserves_requested_order() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {
                "id": "doc-2",
                "user_id": "u1",
                "filename": "b.md",
                "stored_path": "/tmp/u1/b.md",
                "mime_type": "text/markdown",
                "size_bytes": 2,
                "sha256": "bbb",
                "content_text": "b",
                "version": 1,
                "created_at": None,
                "updated_at": None,
            },
            {
                "id": "doc-1",
                "user_id": "u1",
                "filename": "a.md",
                "stored_path": "/tmp/u1/a.md",
                "mime_type": "text/markdown",
                "size_bytes": 1,
                "sha256": "aaa",
                "content_text": "a",
                "version": 1,
                "created_at": None,
                "updated_at": None,
            },
        ]
    )
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(FakeConnection(cursor))))

    documents = store.get_documents_by_ids("u1", ["doc-1", "doc-2"])

    assert [document["id"] for document in documents] == ["doc-1", "doc-2"]
    assert "id = ANY(%s)" in cursor.executed[0][0]


def test_update_document_content_returns_none_when_document_not_found() -> None:
    # SELECT FOR UPDATE finds no row → return None without committing.
    cursor = FakeCursor(fetchone_result=None)
    connection = FakeConnection(cursor)
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(connection)))

    updated = store.update_document_content(
        document_id="doc-1",
        user_id="u1",
        content_text="new",
        size_bytes=3,
        sha256="newhash",
    )

    assert updated is None
    assert connection.committed is False
    assert connection.rolled_back is True
    lock_sqls = [sql for sql, _ in cursor.executed if "FOR UPDATE" in sql]
    assert lock_sqls, "Expected a SELECT … FOR UPDATE query"


def test_update_document_content_raises_conflict_on_version_mismatch() -> None:
    # SELECT FOR UPDATE returns version=5, but caller expects version=3 → conflict.
    cursor = FakeCursor(fetchone_result={"version": 5})
    connection = FakeConnection(cursor)
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(connection)))

    with pytest.raises(WorkspaceVersionConflictError) as exc_info:
        store.update_document_content(
            document_id="doc-1",
            user_id="u1",
            content_text="new",
            size_bytes=3,
            sha256="newhash",
            expected_version=3,
        )

    assert exc_info.value.current_version == 5
    assert exc_info.value.expected_version == 3
    assert connection.committed is False
    assert connection.rolled_back is True


def test_update_document_content_success_sets_source_and_commits() -> None:
    # SELECT FOR UPDATE returns version=1, no mismatch → snapshot + update + commit.
    doc_row = {
        "id": "doc-1",
        "user_id": "u1",
        "filename": "notes.md",
        "stored_path": "/tmp/u1/notes.md",
        "mime_type": "text/markdown",
        "size_bytes": 11,
        "sha256": "newhash",
        "content_text": "updated",
        "version": 2,
        "last_source": "assistant",
        "created_at": datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc),
    }
    # Two fetchone calls: (1) FOR UPDATE lock row, (2) UPDATE RETURNING row.
    cursor = FakeCursor(fetchone_results=[{"version": 1}, doc_row])
    connection = FakeConnection(cursor)
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(connection)))

    updated = store.update_document_content(
        document_id="doc-1",
        user_id="u1",
        content_text="updated",
        size_bytes=11,
        sha256="newhash",
        expected_version=1,
        source="assistant",
    )

    assert updated is not None
    assert updated["id"] == "doc-1"
    assert updated["last_source"] == "assistant"
    assert connection.committed is True
    assert connection.rolled_back is False
    # Snapshot INSERT should precede the UPDATE
    sqls = [sql for sql, _ in cursor.executed]
    snapshot_idx = next((i for i, s in enumerate(sqls) if "INSERT INTO workspace_document_versions" in s), None)
    update_idx = next((i for i, s in enumerate(sqls) if "UPDATE workspace_documents" in s), None)
    assert snapshot_idx is not None, "Snapshot INSERT expected"
    assert update_idx is not None, "UPDATE expected"
    assert snapshot_idx < update_idx, "Snapshot must precede the UPDATE"


def test_delete_document_returns_true_when_row_updated() -> None:
    cursor = FakeCursor(rowcount=1)
    connection = FakeConnection(cursor)
    store = WorkspaceDocumentStore(cast(Any, FakeDbPool(connection)))

    deleted = store.delete_document("doc-1", user_id="u1")

    assert deleted is True
    assert connection.committed is True
    assert "DELETE FROM workspace_documents" in cursor.executed[0][0]
