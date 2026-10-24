"""Conversation-workspace file operations behind the chat ``workspace_action`` path."""
from __future__ import annotations

import pytest

from backend.workspace_file_ops import WorkspaceFileOps


def test_requires_an_explicit_sandbox():
    # The removed file_ops_tool fell back to the working directory (the app source tree).
    with pytest.raises(ValueError):
        WorkspaceFileOps("")


def test_read_existing_file(tmp_path):
    (tmp_path / "hello.txt").write_text("Hello World")
    assert "Hello World" in WorkspaceFileOps(str(tmp_path)).run("read", "hello.txt")


def test_read_missing_or_disallowed(tmp_path):
    ops = WorkspaceFileOps(str(tmp_path))
    assert ops.run("read", "nope.txt").startswith("Error:")
    (tmp_path / "secret.bin").write_bytes(b"\x00")
    assert "not allowed" in ops.run("read", "secret.bin")


def test_read_too_large(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 2_000_000)
    assert "too large" in WorkspaceFileOps(str(tmp_path)).run("read", "big.txt").lower()


def test_write_creates_nested_file(tmp_path):
    result = WorkspaceFileOps(str(tmp_path)).run("write", "sub/dir/file.txt", "nested")
    assert result.startswith("Written:")
    assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"


def test_write_requires_content_and_respects_cap(tmp_path):
    ops = WorkspaceFileOps(str(tmp_path))
    assert ops.run("write", "x.txt").startswith("Error:")
    assert "too large" in ops.run("write", "big.txt", "x" * 2_000_000).lower()


def test_write_revised_never_overwrites(tmp_path):
    (tmp_path / "notes.md").write_text("v1")
    ops = WorkspaceFileOps(str(tmp_path))
    first = ops.run("write_revised", "notes.md", "v2")
    second = ops.run("write_revised", "notes.md", "v3")
    assert first == "Written revised copy: notes.revised.md (2 bytes)"
    assert second == "Written revised copy: notes.revised.1.md (2 bytes)"
    assert (tmp_path / "notes.md").read_text() == "v1"


@pytest.mark.parametrize("op", ["read", "write"])
def test_path_traversal_blocked(tmp_path, op):
    sandbox = tmp_path / "ws"
    sandbox.mkdir()
    assert WorkspaceFileOps(str(sandbox)).run(op, "../escape.txt", "pwned").startswith("Error:")
    assert not (tmp_path / "escape.txt").exists()


def test_sibling_directory_with_shared_prefix_is_blocked(tmp_path):
    # A bare startswith() check let "/ws-evil" pass for sandbox "/ws".
    sandbox = tmp_path / "ws"
    sandbox.mkdir()
    (tmp_path / "ws-evil").mkdir()
    (tmp_path / "ws-evil" / "loot.txt").write_text("other user's data")
    assert WorkspaceFileOps(str(sandbox)).run("read", "../ws-evil/loot.txt").startswith("Error:")


def test_symlink_out_of_sandbox_blocked(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("confidential")
    sandbox = tmp_path / "ws"
    sandbox.mkdir()
    (sandbox / "link.txt").symlink_to(outside / "secret.txt")
    assert WorkspaceFileOps(str(sandbox)).run("read", "link.txt").startswith("Error:")
