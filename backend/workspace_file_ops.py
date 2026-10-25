"""Conversation-workspace file operations behind the chat ``workspace_action`` fast path.

User-initiated API operations (read / write / write a revised copy) confined to one
conversation's workspace directory. This is deliberately NOT a model tool: the former
``file_ops_tool`` exposed the same operations to the LLM with a sandbox that defaulted to
the process working directory (the app's own source tree) and was shared by every user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

_ALLOWED_EXTENSIONS = (
    ".txt", ".md", ".json", ".yaml", ".yml", ".csv",
    ".xml", ".html", ".log", ".py", ".js", ".ts",
)
_MAX_READ_SIZE_BYTES = 1_048_576  # 1 MB
_MAX_WRITE_SIZE_BYTES = 1_048_576  # 1 MB


class WorkspaceFileOps:
    """Read, write and write-revised operations inside one workspace directory.

    Results are human-readable strings; failures start with ``"Error:"`` (the chat
    router maps those to HTTP errors).
    """

    def __init__(self, sandbox_dir: str) -> None:
        if not str(sandbox_dir or "").strip():
            raise ValueError("WorkspaceFileOps requires an explicit sandbox directory")
        self._sandbox = Path(sandbox_dir).resolve()

    def run(
        self,
        operation: str,
        path: Optional[str] = None,
        content: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> str:
        try:
            if operation == "read":
                return self._read(path, encoding)
            if operation == "write":
                return self._write(path, content, encoding)
            if operation == "write_revised":
                return self._write_revised(path, content, encoding)
            return f"Error: Unknown operation '{operation}'. Available: read, write, write_revised"
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller as an error string
            logger.error("Workspace file operation failed", error=str(exc), operation=operation, path=path)
            return f"Error: {exc}"

    # ── Path safety ───────────────────────────────────────────────────

    def _resolve_safe_path(self, user_path: str) -> Path:
        """Resolve ``user_path`` inside the sandbox (symlinks followed); raise if it escapes."""
        resolved = (self._sandbox / user_path).resolve()
        if resolved != self._sandbox and not resolved.is_relative_to(self._sandbox):
            raise ValueError(f"Path escapes sandbox: {user_path}")
        return resolved

    @staticmethod
    def _check_extension(path: Path) -> None:
        if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Extension '{path.suffix}' not allowed. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}"
            )

    # ── Operations ────────────────────────────────────────────────────

    def _read(self, path: Optional[str], encoding: str) -> str:
        if not path:
            return "Error: 'path' is required for read operation."
        resolved = self._resolve_safe_path(path)
        self._check_extension(resolved)
        if not resolved.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        size = resolved.stat().st_size
        if size > _MAX_READ_SIZE_BYTES:
            return f"Error: File too large ({size:,} bytes, max {_MAX_READ_SIZE_BYTES:,})."
        return f"File: {path} ({size:,} bytes)\n\n{resolved.read_text(encoding=encoding)}"

    def _write(self, path: Optional[str], content: Optional[str], encoding: str) -> str:
        if not path:
            return "Error: 'path' is required for write operation."
        if content is None:
            return "Error: 'content' is required for write operation."
        resolved = self._resolve_safe_path(path)
        self._check_extension(resolved)
        content_bytes = len(content.encode(encoding))
        if content_bytes > _MAX_WRITE_SIZE_BYTES:
            return f"Error: Content too large ({content_bytes:,} bytes, max {_MAX_WRITE_SIZE_BYTES:,})."
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
        return f"Written: {path} ({content_bytes:,} bytes)"

    def _write_revised(self, path: Optional[str], content: Optional[str], encoding: str) -> str:
        """Write a revised copy next to an existing file without overwriting it."""
        if not path:
            return "Error: 'path' is required for write_revised operation."
        if content is None:
            return "Error: 'content' is required for write_revised operation."
        source_path = self._resolve_safe_path(path)
        self._check_extension(source_path)
        if not source_path.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        content_bytes = len(content.encode(encoding))
        if content_bytes > _MAX_WRITE_SIZE_BYTES:
            return f"Error: Content too large ({content_bytes:,} bytes, max {_MAX_WRITE_SIZE_BYTES:,})."
        revised_path = self._next_revised_path(source_path)
        revised_path.parent.mkdir(parents=True, exist_ok=True)
        revised_path.write_text(content, encoding=encoding)
        rel = revised_path.relative_to(self._sandbox)
        return f"Written revised copy: {rel.as_posix()} ({content_bytes:,} bytes)"

    @staticmethod
    def _next_revised_path(source_path: Path) -> Path:
        stem, suffix = source_path.stem, source_path.suffix
        candidate = source_path.with_name(f"{stem}.revised{suffix}")
        counter = 1
        while candidate.exists():
            candidate = source_path.with_name(f"{stem}.revised.{counter}{suffix}")
            counter += 1
        return candidate
