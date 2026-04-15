"""File operations tool — sandboxed read, write, list, and info operations."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class FileOpsInput(BaseModel):
    """Input for file operations."""

    operation: Literal["read", "write", "write_revised", "list", "info", "exists"] = Field(
        default="read",
        description="Operation: 'read' (file content), 'write' (create/overwrite), 'write_revised' (safe revised copy), 'list' (directory listing), 'info' (file metadata), 'exists' (check existence)",
    )
    path: Optional[str] = Field(
        default=None,
        description="File or directory path (relative to sandbox root)",
    )
    content: Optional[str] = Field(
        default=None,
        description="Content to write (for 'write' operation)",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding (default: utf-8)",
    )


class FileOpsTool(BaseTool):
    """Read, write, and list files within a sandboxed workspace directory."""

    name: str = "file_ops_tool"
    description: str = (
        "Read, write, and list files and directories within a sandboxed workspace. "
        "Use for file content queries, saving text, directory listings, or file metadata."
    )
    args_schema: type[BaseModel] = FileOpsInput

    # Config: injected by registry from tool.yaml / tools.yaml
    sandbox_dir: str = ""
    max_read_size_bytes: int = 1_048_576  # 1 MB
    max_write_size_bytes: int = 1_048_576  # 1 MB
    allowed_extensions: List[str] = [
        ".txt", ".md", ".json", ".yaml", ".yml", ".csv",
        ".xml", ".html", ".log", ".py", ".js", ".ts",
    ]

    def _get_sandbox_root(self) -> Path:
        """Return the resolved sandbox root directory."""
        if self.sandbox_dir:
            return Path(self.sandbox_dir).resolve()
        return Path.cwd().resolve()

    def _resolve_safe_path(self, user_path: str) -> Path:
        """Resolve a user-provided path safely within the sandbox.

        Raises ValueError if the resolved path escapes the sandbox.
        """
        sandbox = self._get_sandbox_root()
        resolved = (sandbox / user_path).resolve()

        # Prevent path traversal
        if not str(resolved).startswith(str(sandbox)):
            raise ValueError(f"Path escapes sandbox: {user_path}")

        return resolved

    def _check_extension(self, path: Path) -> None:
        """Verify the file extension is allowed."""
        if self.allowed_extensions and path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                f"Extension '{path.suffix}' not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            )

    def _run(
        self,
        operation: str = "read",
        path: Optional[str] = None,
        content: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> str:
        """Execute file operation."""
        try:
            if operation == "read":
                return self._read(path, encoding)
            elif operation == "write":
                return self._write(path, content, encoding)
            elif operation == "write_revised":
                return self._write_revised(path, content, encoding)
            elif operation == "list":
                return self._list(path)
            elif operation == "info":
                return self._info(path)
            elif operation == "exists":
                return self._exists(path)
            else:
                return f"Unknown operation '{operation}'. Available: read, write, write_revised, list, info, exists"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error("File operation failed", error=str(e), operation=operation, path=path)
            return f"Error: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Async execution (uses sync implementation)."""
        return self._run(**kwargs)

    # ── Read ──────────────────────────────────────────────────────────

    def _read(self, path: Optional[str], encoding: str) -> str:
        """Read a file's contents."""
        if not path:
            return "Error: 'path' is required for read operation."

        resolved = self._resolve_safe_path(path)
        self._check_extension(resolved)

        if not resolved.is_file():
            return f"Error: '{path}' is not a file or does not exist."

        size = resolved.stat().st_size
        if size > self.max_read_size_bytes:
            return f"Error: File too large ({size:,} bytes, max {self.max_read_size_bytes:,})."

        logger.info("Reading file", path=str(resolved), size=size)
        text = resolved.read_text(encoding=encoding)
        return f"File: {path} ({size:,} bytes)\n\n{text}"

    # ── Write ─────────────────────────────────────────────────────────

    def _write(self, path: Optional[str], content: Optional[str], encoding: str) -> str:
        """Write content to a file (creates parent dirs if needed)."""
        if not path:
            return "Error: 'path' is required for write operation."
        if content is None:
            return "Error: 'content' is required for write operation."

        resolved = self._resolve_safe_path(path)
        self._check_extension(resolved)

        content_bytes = len(content.encode(encoding))
        if content_bytes > self.max_write_size_bytes:
            return f"Error: Content too large ({content_bytes:,} bytes, max {self.max_write_size_bytes:,})."

        logger.info("Writing file", path=str(resolved), size=content_bytes)
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
        if content_bytes > self.max_write_size_bytes:
            return f"Error: Content too large ({content_bytes:,} bytes, max {self.max_write_size_bytes:,})."

        revised_path = self._next_revised_path(source_path)
        logger.info("Writing revised file", source=str(source_path), target=str(revised_path), size=content_bytes)
        revised_path.parent.mkdir(parents=True, exist_ok=True)
        revised_path.write_text(content, encoding=encoding)

        sandbox = self._get_sandbox_root()
        rel = revised_path.relative_to(sandbox)
        return f"Written revised copy: {rel.as_posix()} ({content_bytes:,} bytes)"

    def _next_revised_path(self, source_path: Path) -> Path:
        stem = source_path.stem
        suffix = source_path.suffix
        first_candidate = source_path.with_name(f"{stem}.revised{suffix}")
        if not first_candidate.exists():
            return first_candidate

        counter = 1
        while True:
            candidate = source_path.with_name(f"{stem}.revised.{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    # ── List ──────────────────────────────────────────────────────────

    def _list(self, path: Optional[str]) -> str:
        """List directory contents."""
        dir_path = path or "."
        resolved = self._resolve_safe_path(dir_path)

        if not resolved.is_dir():
            return f"Error: '{dir_path}' is not a directory or does not exist."

        logger.info("Listing directory", path=str(resolved))

        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return f"Directory '{dir_path}' is empty."

        lines = [f"Directory: {dir_path} ({len(entries)} items)\n"]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  📁 {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"  📄 {entry.name}  ({size:,} bytes)")

        return "\n".join(lines)

    # ── Info ──────────────────────────────────────────────────────────

    def _info(self, path: Optional[str]) -> str:
        """Return metadata about a file or directory."""
        if not path:
            return "Error: 'path' is required for info operation."

        resolved = self._resolve_safe_path(path)
        if not resolved.exists():
            return f"Error: '{path}' does not exist."

        stat = resolved.stat()
        kind = "Directory" if resolved.is_dir() else "File"
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()

        lines = [
            f"{kind}: {path}",
            f"• Size: {stat.st_size:,} bytes",
            f"• Modified: {modified}",
            f"• Created: {created}",
            f"• Permissions: {oct(stat.st_mode)}",
        ]
        if resolved.is_file():
            lines.append(f"• Extension: {resolved.suffix or '(none)'}")

        return "\n".join(lines)

    # ── Exists ────────────────────────────────────────────────────────

    def _exists(self, path: Optional[str]) -> str:
        """Check whether a path exists."""
        if not path:
            return "Error: 'path' is required for exists operation."

        resolved = self._resolve_safe_path(path)
        if resolved.exists():
            kind = "directory" if resolved.is_dir() else "file"
            return f"Yes: '{path}' exists (it is a {kind})."
        return f"No: '{path}' does not exist."


class WorkspaceFileOpsTool(FileOpsTool):
    """Workspace-scoped file tool intended for ephemeral chat workspaces only."""

    name: str = "workspace_file_ops_tool"
    description: str = (
        "Read/list/info/exists and write revised copies within a conversation-scoped workspace. "
        "Never overwrites originals when using write_revised."
    )
