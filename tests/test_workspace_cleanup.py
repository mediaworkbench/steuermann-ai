"""Conversation delete removes the conversation's on-disk workspace directory."""
from __future__ import annotations

import pytest

from backend.attachments import (
    AttachmentManagerConfig,
    ChatAttachmentManager,
    ChatWorkspaceManager,
    WorkspaceManagerConfig,
    WorkspaceValidationError,
)


def _manager(root):
    return ChatWorkspaceManager(
        attachment_manager=ChatAttachmentManager(AttachmentManagerConfig(root_dir=root)),
        config=WorkspaceManagerConfig(root_dir=root, retention_hours=24),
    )


def test_remove_conversation_dir_only_touches_that_conversation(tmp_path):
    m = _manager(tmp_path)
    (m.get_workspace_dir("conv-1") / "a.txt").write_text("hello")
    keep = m.get_workspace_dir("conv-2")

    assert m.remove_conversation_dir("conv-1") == 5
    assert not (tmp_path / "conv-1").exists()
    assert keep.exists()
    assert m.remove_conversation_dir("missing") == 0


def test_remove_conversation_dir_rejects_traversal(tmp_path):
    with pytest.raises(WorkspaceValidationError):
        _manager(tmp_path).remove_conversation_dir("../etc")
