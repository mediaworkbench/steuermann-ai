from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.attachments import (
    AttachmentManagerConfig,
    AttachmentValidationError,
    ChatAttachmentManager,
)


@pytest.fixture()
def manager(tmp_path: Path) -> ChatAttachmentManager:
    return ChatAttachmentManager(
        AttachmentManagerConfig(
            root_dir=tmp_path / "chat-workspaces",
            max_file_bytes=64,
            retention_hours=24,
        )
    )


def test_sanitize_filename_strips_paths_and_invalid_chars(manager: ChatAttachmentManager) -> None:
    safe_name = manager.sanitize_filename("../../Draft Notes (v1).md")

    assert safe_name == "Draft_Notes_v1.md"


def test_store_attachment_writes_file_and_returns_metadata(manager: ChatAttachmentManager) -> None:
    content = b"# Title\nHello world\n"

    stored = manager.store_attachment(
        conversation_id="conv_123",
        attachment_id="att_123",
        filename="notes.md",
        content=content,
        content_type="text/markdown",
    )

    assert stored["original_name"] == "notes.md"
    assert stored["mime_type"] == "text/markdown"
    assert stored["size_bytes"] == len(content)
    assert stored["sha256"] == hashlib.sha256(content).hexdigest()
    assert stored["extracted_text"] == "# Title\nHello world\n"
    assert Path(stored["stored_path"]).exists()
    assert "/attachments/" in str(stored["stored_path"])


def test_store_attachment_rejects_invalid_extension(manager: ChatAttachmentManager) -> None:
    with pytest.raises(AttachmentValidationError, match="Extension"):
        manager.store_attachment(
            conversation_id="conv_123",
            attachment_id="att_123",
            filename="notes.pdf",
            content=b"hello",
            content_type="application/pdf",
        )


def test_store_attachment_rejects_oversized_file(manager: ChatAttachmentManager) -> None:
    with pytest.raises(AttachmentValidationError, match="File too large"):
        manager.store_attachment(
            conversation_id="conv_123",
            attachment_id="att_123",
            filename="notes.txt",
            content=b"a" * 65,
            content_type="text/plain",
        )


def test_store_attachment_rejects_binary_content(manager: ChatAttachmentManager) -> None:
    with pytest.raises(AttachmentValidationError, match="Binary content"):
        manager.store_attachment(
            conversation_id="conv_123",
            attachment_id="att_123",
            filename="notes.txt",
            content=b"hello\x00world",
            content_type="text/plain",
        )


def test_delete_stored_file_removes_file(manager: ChatAttachmentManager) -> None:
    stored = manager.store_attachment(
        conversation_id="conv_123",
        attachment_id="att_123",
        filename="notes.txt",
        content=b"hello",
        content_type="text/plain",
    )

    deleted = manager.delete_stored_file(str(stored["stored_path"]))

    assert deleted is True
    assert not Path(stored["stored_path"]).exists()


def test_delete_stored_file_rejects_paths_outside_root(manager: ChatAttachmentManager, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("hello", encoding="utf-8")

    with pytest.raises(AttachmentValidationError, match="escapes attachments root"):
        manager.delete_stored_file(str(outside))