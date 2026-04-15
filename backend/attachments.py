from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import shutil


_DEFAULT_ATTACHMENTS_ROOT = "/tmp/steuermann-ai/chat-workspaces"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class AttachmentValidationError(ValueError):
    """Raised when an attachment upload fails validation."""


class WorkspaceValidationError(ValueError):
    """Raised when a workspace operation fails validation."""


@dataclass(frozen=True)
class AttachmentManagerConfig:
    root_dir: Path
    max_file_bytes: int = 524_288
    retention_hours: int = 168
    allowed_extensions: tuple[str, ...] = (".txt", ".md")
    allowed_mime_types: tuple[str, ...] = (
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    )

    @classmethod
    def from_env(cls) -> "AttachmentManagerConfig":
        root_dir = Path(os.getenv("CHAT_ATTACHMENTS_ROOT", _DEFAULT_ATTACHMENTS_ROOT))
        max_file_bytes = int(os.getenv("CHAT_ATTACHMENTS_MAX_FILE_BYTES", "524288"))
        retention_hours = int(os.getenv("CHAT_ATTACHMENTS_RETENTION_HOURS", "168"))
        return cls(
            root_dir=root_dir,
            max_file_bytes=max_file_bytes,
            retention_hours=retention_hours,
        )


class ChatAttachmentManager:
    """Manage conversation-scoped temporary attachment files."""

    def __init__(self, config: Optional[AttachmentManagerConfig] = None) -> None:
        self.config = config or AttachmentManagerConfig.from_env()

    def ensure_root(self) -> Path:
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        return self.config.root_dir

    def get_conversation_root(self, conversation_id: str) -> Path:
        safe_conversation_id = self._validate_identifier(conversation_id, name="conversation_id")
        return self.ensure_root() / safe_conversation_id

    def get_attachments_dir(self, conversation_id: str) -> Path:
        attachments_dir = self.get_conversation_root(conversation_id) / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        return attachments_dir

    def sanitize_filename(self, filename: str) -> str:
        base_name = Path(filename or "attachment.txt").name.strip()
        if not base_name:
            base_name = "attachment.txt"

        suffix = Path(base_name).suffix.lower()
        stem = Path(base_name).stem.strip() or "attachment"
        stem = _SAFE_NAME_RE.sub("_", stem).strip("._-") or "attachment"
        safe_suffix = _SAFE_NAME_RE.sub("", suffix)
        return f"{stem}{safe_suffix}"

    def validate_upload(self, filename: str, content: bytes, content_type: Optional[str] = None) -> tuple[str, str]:
        safe_name = self.sanitize_filename(filename)
        suffix = Path(safe_name).suffix.lower()

        if suffix not in self.config.allowed_extensions:
            raise AttachmentValidationError(
                f"Extension '{suffix or '(none)'}' not allowed. Allowed: {', '.join(self.config.allowed_extensions)}"
            )

        size_bytes = len(content)
        if size_bytes > self.config.max_file_bytes:
            raise AttachmentValidationError(
                f"File too large ({size_bytes:,} bytes, max {self.config.max_file_bytes:,})."
            )

        normalized_content_type = (content_type or self._mime_type_for_suffix(suffix)).strip().lower()
        if normalized_content_type and normalized_content_type not in self.config.allowed_mime_types:
            raise AttachmentValidationError(
                f"Content type '{normalized_content_type}' not allowed. Allowed: {', '.join(self.config.allowed_mime_types)}"
            )

        if b"\x00" in content:
            raise AttachmentValidationError("Binary content is not allowed for text attachments.")

        return safe_name, normalized_content_type or self._mime_type_for_suffix(suffix)

    def extract_text(self, content: bytes) -> str:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AttachmentValidationError("Attachment must be valid UTF-8 text.") from exc

    def build_storage_path(self, conversation_id: str, attachment_id: str, filename: str) -> Path:
        safe_attachment_id = self._validate_identifier(attachment_id, name="attachment_id")
        safe_name = self.sanitize_filename(filename)
        return self.get_attachments_dir(conversation_id) / f"{safe_attachment_id}__{safe_name}"

    def store_attachment(
        self,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> dict[str, object]:
        safe_name, normalized_content_type = self.validate_upload(filename, content, content_type)
        extracted_text = self.extract_text(content)
        stored_path = self.build_storage_path(conversation_id, attachment_id, safe_name)
        stored_path.write_bytes(content)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.config.retention_hours)
        sha256 = hashlib.sha256(content).hexdigest()

        return {
            "original_name": Path(filename or safe_name).name or safe_name,
            "stored_path": str(stored_path),
            "mime_type": normalized_content_type,
            "size_bytes": len(content),
            "sha256": sha256,
            "extracted_text": extracted_text,
            "created_at": now,
            "expires_at": expires_at,
        }

    def delete_stored_file(self, stored_path: str) -> bool:
        path = Path(stored_path).resolve()
        root = self.ensure_root().resolve()

        if not str(path).startswith(str(root)):
            raise AttachmentValidationError("Stored path escapes attachments root.")

        if path.exists():
            path.unlink()
            return True
        return False

    def _validate_identifier(self, value: str, *, name: str) -> str:
        if not _SAFE_ID_RE.match(value or ""):
            raise AttachmentValidationError(f"Invalid {name}.")
        return value

    def _mime_type_for_suffix(self, suffix: str) -> str:
        if suffix == ".md":
            return "text/markdown"
        return "text/plain"


@dataclass(frozen=True)
class WorkspaceManagerConfig:
    root_dir: Path
    retention_hours: int = 24

    @classmethod
    def from_env(cls) -> "WorkspaceManagerConfig":
        attachments_root = Path(os.getenv("CHAT_ATTACHMENTS_ROOT", _DEFAULT_ATTACHMENTS_ROOT))
        configured_workspace_root = os.getenv("CHAT_WORKSPACE_ROOT", "").strip()
        root_dir = Path(configured_workspace_root) if configured_workspace_root else attachments_root
        retention_hours = int(os.getenv("CHAT_WORKSPACE_RETENTION_HOURS", "24"))
        return cls(root_dir=root_dir, retention_hours=retention_hours)


class ChatWorkspaceManager:
    """Manage conversation-scoped ephemeral workspace files."""

    def __init__(
        self,
        attachment_manager: Optional[ChatAttachmentManager] = None,
        config: Optional[WorkspaceManagerConfig] = None,
    ) -> None:
        self.attachment_manager = attachment_manager or ChatAttachmentManager()
        self.config = config or WorkspaceManagerConfig.from_env()

    def ensure_root(self) -> Path:
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        return self.config.root_dir

    def get_workspace_dir(self, conversation_id: str) -> Path:
        safe_conversation_id = self._validate_identifier(conversation_id, name="conversation_id")
        workspace_dir = self.ensure_root() / safe_conversation_id / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def resolve_workspace_path(self, conversation_id: str, relative_path: str) -> Path:
        if not relative_path or relative_path.strip() == "":
            raise WorkspaceValidationError("relative_path is required")
        workspace_dir = self.get_workspace_dir(conversation_id).resolve()
        candidate = (workspace_dir / relative_path).resolve()
        if not str(candidate).startswith(str(workspace_dir)):
            raise WorkspaceValidationError("Workspace path escapes conversation workspace root.")
        return candidate

    def materialize_attachment_copy(
        self,
        conversation_id: str,
        stored_attachment_path: str,
        target_name: Optional[str] = None,
    ) -> dict[str, Any]:
        source_path = Path(stored_attachment_path).resolve()
        if not source_path.exists() or not source_path.is_file():
            raise WorkspaceValidationError("Attachment source file does not exist.")

        requested_name = target_name or source_path.name.split("__", 1)[-1]
        safe_name = self.attachment_manager.sanitize_filename(requested_name)
        workspace_dir = self.get_workspace_dir(conversation_id)
        destination = self._next_available_path(workspace_dir / safe_name)

        shutil.copy2(source_path, destination)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.config.retention_hours)
        relative_path = str(destination.relative_to(workspace_dir))

        return {
            "workspace_root": str(workspace_dir),
            "path": str(destination),
            "relative_path": relative_path,
            "name": destination.name,
            "size_bytes": destination.stat().st_size,
            "created_at": now,
            "expires_at": expires_at,
        }

    def list_workspace_files(self, conversation_id: str) -> list[dict[str, Any]]:
        workspace_dir = self.get_workspace_dir(conversation_id)
        files: list[dict[str, Any]] = []
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(workspace_dir)),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        return files

    def delete_workspace_tree(
        self,
        conversation_id: Optional[str] = None,
        root_path: Optional[str] = None,
    ) -> int:
        if root_path:
            workspace_dir = Path(root_path).resolve()
        elif conversation_id:
            workspace_dir = self.get_workspace_dir(conversation_id).resolve()
        else:
            raise WorkspaceValidationError("conversation_id or root_path is required")

        root = self.ensure_root().resolve()
        if not str(workspace_dir).startswith(str(root)):
            raise WorkspaceValidationError("Workspace root escapes configured workspace base directory.")

        if not workspace_dir.exists():
            return 0

        deleted_count = sum(1 for _ in workspace_dir.rglob("*")) + 1
        shutil.rmtree(workspace_dir)

        conversation_root = workspace_dir.parent
        if conversation_root.exists() and conversation_root.is_dir():
            try:
                next(conversation_root.iterdir())
            except StopIteration:
                conversation_root.rmdir()
                deleted_count += 1

        return deleted_count

    def _next_available_path(self, desired_path: Path) -> Path:
        if not desired_path.exists():
            return desired_path

        suffix = desired_path.suffix
        stem = desired_path.stem
        counter = 1
        while True:
            candidate = desired_path.with_name(f"{stem}.{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _validate_identifier(self, value: str, *, name: str) -> str:
        if not _SAFE_ID_RE.match(value or ""):
            raise WorkspaceValidationError(f"Invalid {name}.")
        return value


class UserWorkspaceFileManager:
    """Manage persistent per-user workspace files with filesystem isolation.
    
    This manager provides:
    - Per-user workspace root directories
    - Filename sanitization and path confinement
    - Integration with WorkspaceDocumentStore for metadata
    """

    def __init__(
        self,
        attachment_manager: Optional[ChatAttachmentManager] = None,
        config: Optional[WorkspaceManagerConfig] = None,
    ) -> None:
        self.attachment_manager = attachment_manager or ChatAttachmentManager()
        self.config = config or WorkspaceManagerConfig.from_env()

    def ensure_root(self) -> Path:
        """Ensure base workspace root directory exists."""
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        return self.config.root_dir

    def get_user_workspace_root(self, user_id: str) -> Path:
        """Get or create per-user workspace root directory.
        
        Args:
            user_id: User identifier (validated as safe identifier)
        
        Returns:
            Path to user's workspace root directory
        
        Raises:
            WorkspaceValidationError: If user_id fails validation
        """
        safe_user_id = self._validate_identifier(user_id, name="user_id")
        workspace_root = self.ensure_root() / "user-workspaces" / safe_user_id
        workspace_root.mkdir(parents=True, exist_ok=True)
        return workspace_root

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to safe format.
        
        Uses same pattern as ChatAttachmentManager for consistency.
        """
        base_name = Path(filename or "document.txt").name.strip()
        if not base_name:
            base_name = "document.txt"

        suffix = Path(base_name).suffix.lower()
        stem = Path(base_name).stem.strip() or "document"
        stem = _SAFE_NAME_RE.sub("_", stem).strip("._-") or "document"
        safe_suffix = _SAFE_NAME_RE.sub("", suffix)
        return f"{stem}{safe_suffix}"

    def resolve_user_file_path(self, user_id: str, relative_path: str) -> Path:
        """Resolve a file path within user workspace with confinement check.
        
        Args:
            user_id: User identifier
            relative_path: Path relative to user workspace root
        
        Returns:
            Resolved Path object
        
        Raises:
            WorkspaceValidationError: If relative_path escapes workspace or is invalid
        """
        if not relative_path or relative_path.strip() == "":
            raise WorkspaceValidationError("relative_path is required")
        
        workspace_root = self.get_user_workspace_root(user_id).resolve()
        candidate = (workspace_root / relative_path).resolve()
        
        if not str(candidate).startswith(str(workspace_root)):
            raise WorkspaceValidationError("File path escapes user workspace root.")
        
        return candidate

    def store_document_file(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Store a new document file in user workspace.
        
        Args:
            user_id: User identifier
            document_id: Unique document identifier
            filename: Original filename (will be sanitized)
            content: File content as bytes
        
        Returns:
            Dictionary with storage metadata:
                - workspace_root: User workspace root path
                - stored_path: Full filesystem path to stored file
                - relative_path: Path relative to workspace root
                - filename: Sanitized filename
                - size_bytes: File size in bytes
                - sha256: SHA256 hash of content
        
        Raises:
            WorkspaceValidationError: On validation errors
        """
        # Validate user_id and document_id
        safe_user_id = self._validate_identifier(user_id, name="user_id")
        safe_document_id = self._validate_identifier(document_id, name="document_id")
        
        # Sanitize filename
        safe_name = self.sanitize_filename(filename)
        
        # Get workspace root and build storage path
        workspace_root = self.get_user_workspace_root(safe_user_id)
        stored_path = self._next_available_path(workspace_root / safe_name)
        
        # Write file to disk
        stored_path.write_bytes(content)
        
        # Calculate metadata
        size_bytes = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
        relative_path = str(stored_path.relative_to(workspace_root))
        
        return {
            "workspace_root": str(workspace_root),
            "stored_path": str(stored_path),
            "relative_path": relative_path,
            "filename": stored_path.name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }

    def read_document_file(self, user_id: str, document_id: str, stored_path: str) -> bytes:
        """Read content of a stored document file.
        
        Args:
            user_id: User identifier
            document_id: Document identifier (for validation context)
            stored_path: Full filesystem path to document file
        
        Returns:
            File content as bytes
        
        Raises:
            WorkspaceValidationError: If path escapes workspace or file doesn't exist
        """
        self._validate_identifier(user_id, name="user_id")
        self._validate_identifier(document_id, name="document_id")
        
        workspace_root = self.get_user_workspace_root(user_id).resolve()
        file_path = Path(stored_path).resolve()
        
        # Verify path confinement
        if not str(file_path).startswith(str(workspace_root)):
            raise WorkspaceValidationError("File path escapes user workspace root.")
        
        if not file_path.exists() or not file_path.is_file():
            raise WorkspaceValidationError("Document file does not exist.")
        
        return file_path.read_bytes()

    def update_document_file(
        self,
        user_id: str,
        document_id: str,
        stored_path: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Update content of a stored document file in-place.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            stored_path: Full filesystem path to document file
            content: New file content as bytes
        
        Returns:
            Dictionary with updated metadata:
                - stored_path: Full filesystem path
                - relative_path: Path relative to workspace root
                - size_bytes: Updated file size
                - sha256: New SHA256 hash
                - updated_at: ISO datetime of update
        
        Raises:
            WorkspaceValidationError: If path escapes workspace or file doesn't exist
        """
        self._validate_identifier(user_id, name="user_id")
        self._validate_identifier(document_id, name="document_id")
        
        workspace_root = self.get_user_workspace_root(user_id).resolve()
        file_path = Path(stored_path).resolve()
        
        # Verify path confinement
        if not str(file_path).startswith(str(workspace_root)):
            raise WorkspaceValidationError("File path escapes user workspace root.")

        # Recreate missing files in-place when DB metadata survived a container rebuild
        # but the workspace filesystem did not.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and not file_path.is_file():
            raise WorkspaceValidationError("Document file path is not a regular file.")
        
        # Write updated content
        file_path.write_bytes(content)
        
        # Calculate metadata
        size_bytes = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
        relative_path = str(file_path.relative_to(workspace_root))
        updated_at = datetime.now(timezone.utc)
        
        return {
            "stored_path": str(file_path),
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "updated_at": updated_at.isoformat(),
        }

    def delete_document_file(self, user_id: str, document_id: str, stored_path: str) -> bool:
        """Delete a stored document file (hard delete).
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            stored_path: Full filesystem path to document file
        
        Returns:
            True if file was deleted, False if it didn't exist
        
        Raises:
            WorkspaceValidationError: If path escapes workspace
        """
        self._validate_identifier(user_id, name="user_id")
        self._validate_identifier(document_id, name="document_id")
        
        workspace_root = self.get_user_workspace_root(user_id).resolve()
        file_path = Path(stored_path).resolve()
        
        # Verify path confinement
        if not str(file_path).startswith(str(workspace_root)):
            raise WorkspaceValidationError("File path escapes user workspace root.")
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_user_documents(self, user_id: str) -> list[dict[str, Any]]:
        """List all files in user workspace.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of document metadata dicts with:
                - relative_path: Path relative to workspace root
                - name: Filename only
                - size_bytes: File size
                - modified_at: ISO datetime of last modification
        """
        workspace_root = self.get_user_workspace_root(user_id)
        files: list[dict[str, Any]] = []
        
        for path in sorted(workspace_root.rglob("*")):
            if not path.is_file():
                continue
            
            stat = path.stat()
            files.append({
                "relative_path": str(path.relative_to(workspace_root)),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        
        return files

    def _next_available_path(self, desired_path: Path) -> Path:
        """Find next available filename if desired path exists.
        
        Appends numeric counter before extension if file exists.
        """
        if not desired_path.exists():
            return desired_path

        suffix = desired_path.suffix
        stem = desired_path.stem
        counter = 1
        
        while True:
            candidate = desired_path.with_name(f"{stem}.{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _validate_identifier(self, value: str, *, name: str) -> str:
        """Validate identifier format (user_id, document_id, etc.)."""
        if not _SAFE_ID_RE.match(value or ""):
            raise WorkspaceValidationError(f"Invalid {name}.")
        return value