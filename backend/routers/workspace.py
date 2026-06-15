"""User workspace document management API."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.attachments import AttachmentValidationError, UserWorkspaceFileManager, WorkspaceValidationError
from backend.db import WorkspaceDocumentStore, WorkspaceVersionConflictError
from backend.auth import CurrentUser, resolve_current_user
from backend.single_user import require_api_access

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspace",
    tags=["workspace"],
    dependencies=[Depends(require_api_access)],
)


# ── Request / Response Models ────────────────────────────────────────────


class WorkspaceDocumentResponse(BaseModel):
    """Workspace document metadata."""

    id: str
    user_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    version: int
    last_source: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WorkspaceDocumentWithContentResponse(BaseModel):
    """Workspace document with full content."""

    id: str
    user_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    content_text: str
    version: int
    last_source: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WorkspaceDocumentListResponse(BaseModel):
    """List of workspace documents."""

    documents: List[WorkspaceDocumentResponse]
    total: int


class WorkspaceDocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    document: WorkspaceDocumentResponse


class WorkspaceDocumentUpdateResponse(BaseModel):
    """Response after updating a document."""

    document: WorkspaceDocumentResponse


class WorkspaceDocumentPatch(BaseModel):
    """Partial update for a workspace document (currently: rename only)."""

    filename: Optional[str] = Field(default=None, min_length=1, max_length=255)


class WorkspaceDocumentVersionResponse(BaseModel):
    """A historical version snapshot (no content_text)."""

    id: str
    document_id: str
    version: int
    size_bytes: int
    sha256: str
    source: Optional[str] = None
    created_at: Optional[str] = None


class WorkspaceDocumentVersionWithContentResponse(BaseModel):
    """A historical version snapshot with full content."""

    id: str
    document_id: str
    version: int
    content_text: str
    size_bytes: int
    sha256: str
    source: Optional[str] = None
    created_at: Optional[str] = None


# ── Helper Functions ────────────────────────────────────────────────────


def _get_file_manager(request: Request) -> UserWorkspaceFileManager:
    """Extract UserWorkspaceFileManager from app state."""
    manager = getattr(request.app.state, "user_workspace_file_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="Workspace file manager not initialized")
    return manager


def _get_document_store(request: Request) -> WorkspaceDocumentStore:
    """Extract WorkspaceDocumentStore from app state."""
    store = getattr(request.app.state, "workspace_document_store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="Workspace document store not initialized")
    return store


_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown",
    ".json", ".yaml", ".yml",
    ".csv", ".html", ".xml",
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
})

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

_EXT_TO_MIME: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".csv": "text/csv",
    ".html": "text/html",
    ".xml": "text/xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Per-extension allowed MIME types — text/plain is accepted universally as a browser fallback
_EXT_ALLOWED_MIMES: dict[str, frozenset[str]] = {
    ".txt": frozenset({"text/plain"}),
    ".md": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    ".markdown": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    ".json": frozenset({"application/json", "text/plain"}),
    ".yaml": frozenset({"application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml", "text/plain"}),
    ".yml": frozenset({"application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml", "text/plain"}),
    ".csv": frozenset({"text/csv", "text/plain"}),
    ".html": frozenset({"text/html", "text/plain"}),
    ".xml": frozenset({"text/xml", "application/xml", "text/plain"}),
    ".jpg": frozenset({"image/jpeg"}),
    ".jpeg": frozenset({"image/jpeg"}),
    ".png": frozenset({"image/png"}),
    ".gif": frozenset({"image/gif"}),
    ".webp": frozenset({"image/webp"}),
}

_WORKSPACE_MAX_FILE_BYTES: int = int(os.getenv("WORKSPACE_MAX_FILE_BYTES", "10485760"))


def _validate_text_file(filename: str, content_type: Optional[str] = None) -> str:
    """Validate text file format (extension + mime type).

    Returns:
        Normalized MIME type

    Raises:
        AttachmentValidationError: If file format is invalid
    """
    ext = Path(filename).suffix.lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise AttachmentValidationError(
            f"File type '{ext or '(none)'}' not supported. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    if ext in _IMAGE_EXTENSIONS:
        raise AttachmentValidationError(
            f"Image files must be uploaded as images, not text documents. Use the image upload path for '{ext}' files."
        )

    normalized_content_type = (content_type or "").strip().lower().split(";")[0].strip()
    if not normalized_content_type:
        normalized_content_type = _EXT_TO_MIME.get(ext, "text/plain")
    elif normalized_content_type not in _EXT_ALLOWED_MIMES.get(ext, frozenset()):
        raise AttachmentValidationError(
            f"Content type '{normalized_content_type}' is not allowed for '{ext}' files."
        )

    return normalized_content_type


def _validate_utf8_text(content: bytes) -> str:
    """Validate and decode UTF-8 text content.

    Raises:
        AttachmentValidationError: If content is not valid UTF-8
    """
    if b"\x00" in content:
        raise AttachmentValidationError("Binary content is not allowed for text files.")

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AttachmentValidationError("File must be valid UTF-8 text.") from exc


def _validate_image_file(filename: str, content_type: Optional[str] = None) -> str:
    """Validate image file format (extension + mime type).

    Returns:
        Normalized MIME type

    Raises:
        AttachmentValidationError: If file format is invalid
    """
    ext = Path(filename).suffix.lower()

    if ext not in _IMAGE_EXTENSIONS:
        raise AttachmentValidationError(
            f"File type '{ext or '(none)'}' is not a supported image format. "
            f"Supported: {', '.join(sorted(_IMAGE_EXTENSIONS))}"
        )

    normalized_content_type = (content_type or "").strip().lower().split(";")[0].strip()
    if not normalized_content_type:
        normalized_content_type = _EXT_TO_MIME[ext]
    elif normalized_content_type not in _EXT_ALLOWED_MIMES.get(ext, frozenset()):
        # browsers occasionally send application/octet-stream for images — accept it
        if normalized_content_type != "application/octet-stream":
            raise AttachmentValidationError(
                f"Content type '{normalized_content_type}' is not allowed for '{ext}' files."
            )
        normalized_content_type = _EXT_TO_MIME[ext]

    return normalized_content_type


def _thumb_path(stored_path: str) -> Path:
    """Derive the cached thumbnail path for an image document."""
    return Path(str(stored_path) + ".thumb.jpg")


# ── API Endpoints ────────────────────────────────────────────────────────


@router.post("/documents", response_model=WorkspaceDocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Upload a new document or image to user workspace.

    Supports text/markdown files and images (jpg, jpeg, png, gif, webp).
    Stores both filesystem copy and database metadata.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)

    effective_user_id = current_user.user_id
    document_id = str(uuid.uuid4())
    filename = file.filename or "document.txt"

    try:
        content = await file.read()

        if len(content) > _WORKSPACE_MAX_FILE_BYTES:
            raise AttachmentValidationError(
                f"File too large ({len(content):,} bytes, max {_WORKSPACE_MAX_FILE_BYTES:,})."
            )

        ext = Path(filename).suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            normalized_mime_type = _validate_image_file(filename, file.content_type)
            content_text = ""
        else:
            normalized_mime_type = _validate_text_file(filename, file.content_type)
            content_text = _validate_utf8_text(content)

        stored_metadata = file_manager.store_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            filename=filename,
            content=content,
        )

        doc = document_store.create_document(
            document_id=document_id,
            user_id=effective_user_id,
            filename=filename,
            stored_path=str(stored_metadata["stored_path"]),
            mime_type=normalized_mime_type,
            size_bytes=int(stored_metadata["size_bytes"]),
            sha256=str(stored_metadata["sha256"]),
            content_text=content_text,
        )

        return {"document": doc}

    except AttachmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to upload document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents", response_model=WorkspaceDocumentListResponse)
async def list_documents(
    request: Request,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """List all documents in user workspace.
    
    Returns documents ordered by most recently updated first.
    """
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id
    
    try:
        documents = document_store.list_documents(
            user_id=effective_user_id,
            limit=limit,
            offset=offset,
        )
        return {
            "documents": documents,
            "total": document_store.count_documents(user_id=effective_user_id),
        }
    except Exception as exc:
        logger.error(f"Failed to list documents: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents/{document_id}", response_model=WorkspaceDocumentWithContentResponse)
async def get_document(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Fetch a document by ID with full content.
    
    Ownership is enforced: user can only access their own documents.
    """
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id
    
    try:
        doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/documents/{document_id}", response_model=WorkspaceDocumentUpdateResponse)
async def update_document(
    request: Request,
    document_id: str,
    file: UploadFile = File(...),
    expected_version: Optional[int] = Form(default=None),
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Overwrite a text document with new content.

    Updates the file on disk, increments version, and returns updated metadata.
    Images are immutable (cannot be re-uploaded as text). Pass `expected_version`
    to detect concurrent edits — a mismatch returns 409.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id

    try:
        # Verify document exists and belongs to user
        existing_doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Images are immutable — they cannot be meaningfully edited as text.
        if str(existing_doc.get("mime_type") or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Images cannot be edited")

        # Read and validate new content
        content = await file.read()
        normalized_mime_type = _validate_text_file(file.filename or existing_doc["filename"], file.content_type)
        content_text = _validate_utf8_text(content)

        # Compute metadata from content directly so we can do the DB write (with its
        # version check) BEFORE touching the filesystem. This prevents a conflict 409
        # from leaving the on-disk file ahead of the DB record.
        size_bytes = len(content)
        sha256 = hashlib.sha256(content).hexdigest()

        # Update database record first — acquires row lock + checks expected_version.
        # Only touches disk on success, so a stale-version reject leaves files intact.
        updated_doc = document_store.update_document_content(
            document_id=document_id,
            user_id=effective_user_id,
            content_text=content_text,
            size_bytes=size_bytes,
            sha256=sha256,
            expected_version=expected_version,
            source="user",
        )

        if not updated_doc:
            raise HTTPException(status_code=500, detail="Failed to update document")

        file_manager.update_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            stored_path=existing_doc["stored_path"],
            content=content,
        )

        return {"document": updated_doc}

    except HTTPException:
        raise
    except WorkspaceVersionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Document was modified elsewhere (now at version {exc.current_version}). Reload and retry.",
        ) from exc
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to update document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-. ]{1,255}$")


@router.patch("/documents/{document_id}", response_model=WorkspaceDocumentResponse)
async def patch_document(
    document_id: str,
    body: WorkspaceDocumentPatch,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> WorkspaceDocumentResponse:
    """Rename a workspace document."""
    effective_user_id = current_user.user_id
    document_store = _get_document_store(request)

    if not body.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    new_filename = body.filename.strip()
    if not _SAFE_FILENAME_RE.match(new_filename):
        raise HTTPException(status_code=400, detail="Invalid filename: only letters, digits, spaces, hyphens, underscores, and dots are allowed")

    ext = Path(new_filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext or '(none)'}' not supported. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    existing = document_store.get_document(document_id=document_id, user_id=effective_user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    # A rename must not change the document's type category (text ↔ image) — the
    # stored mime_type and bytes wouldn't match the new extension.
    old_ext = Path(existing["filename"]).suffix.lower()
    if (ext in _IMAGE_EXTENSIONS) != (old_ext in _IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Cannot change a document between text and image types via rename",
        )

    updated = document_store.rename_document(
        document_id=document_id,
        user_id=effective_user_id,
        filename=new_filename,
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to rename document")

    return WorkspaceDocumentResponse(**updated)


@router.delete("/documents/{document_id}", status_code=204, response_class=Response)
async def delete_document(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Response:
    """Delete a document (hard delete).
    
    Removes from filesystem and database.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id
    
    try:
        # Verify document exists and belongs to user
        existing_doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from filesystem
        file_manager.delete_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            stored_path=existing_doc["stored_path"],
        )
        _thumb_path(existing_doc["stored_path"]).unlink(missing_ok=True)

        # Delete from database
        deleted = document_store.delete_document(document_id=document_id, user_id=effective_user_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete document from database")

        return Response(status_code=204)
    
    except HTTPException:
        raise
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to delete document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents/{document_id}/download", response_class=FileResponse)
async def download_document(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> FileResponse:
    """Download current saved content of a document.
    
    Returns the exact file as stored, with appropriate Content-Disposition header.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id
    
    try:
        # Verify document exists and belongs to user
        existing_doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Read file content
        content = file_manager.read_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            stored_path=existing_doc["stored_path"],
        )
        
        # Return as file download with original filename
        return FileResponse(
            path=existing_doc["stored_path"],
            filename=existing_doc["filename"],
            media_type=existing_doc["mime_type"],
        )
    
    except HTTPException:
        raise
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to download document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents/{document_id}/thumbnail", response_class=FileResponse)
async def get_document_thumbnail(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> FileResponse:
    """Return a JPEG thumbnail (max 320×240) for an image workspace document.

    Thumbnails are generated lazily on first request and cached on disk alongside the
    original file as ``<stored_path>.thumb.jpg``.
    """
    from PIL import Image as PilImage  # noqa: PLC0415 — lazy import; Pillow is optional dep

    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id

    doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.get("mime_type", "").startswith("image/"):
        raise HTTPException(status_code=404, detail="Document is not an image")

    stored_path = doc["stored_path"]
    if not Path(stored_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    thumb = _thumb_path(stored_path)
    if not thumb.exists():
        try:
            img = PilImage.open(stored_path).convert("RGB")
            img.thumbnail((320, 240))
            img.save(str(thumb), format="JPEG", quality=85, optimize=True)
        except Exception as exc:
            logger.error(f"Thumbnail generation failed for {document_id}: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail") from exc

    return FileResponse(str(thumb), media_type="image/jpeg")


@router.delete("/documents", status_code=200)
async def delete_all_documents(
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Delete all workspace documents for the current user.

    DB records are removed first (within a single DELETE), then files are
    cleaned up on a best-effort basis so the database is always consistent.
    """
    document_store = _get_document_store(request)
    effective_user_id = current_user.user_id

    try:
        docs = document_store.list_documents(user_id=effective_user_id, limit=10000, offset=0)
        stored_paths = [d["stored_path"] for d in docs]

        count = document_store.delete_all_documents(user_id=effective_user_id)

        for sp in stored_paths:
            try:
                Path(sp).unlink(missing_ok=True)
                _thumb_path(sp).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(f"Could not remove file {sp}: {exc}")

        return {"deleted": count}
    except Exception as exc:
        logger.error(f"Failed to clear workspace documents: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Version History Endpoints ─────────────────────────────────────────────


@router.get(
    "/documents/{document_id}/versions",
    response_model=List[WorkspaceDocumentVersionResponse],
)
async def list_document_versions(
    document_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: CurrentUser = Depends(resolve_current_user),
) -> List[WorkspaceDocumentVersionResponse]:
    """List historical version snapshots for a document (no content_text)."""
    effective_user_id = current_user.user_id
    document_store = _get_document_store(request)

    existing = document_store.get_document(document_id=document_id, user_id=effective_user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    versions = document_store.list_document_versions(
        document_id=document_id, user_id=effective_user_id, limit=limit
    )
    return [WorkspaceDocumentVersionResponse(**v) for v in versions]


@router.get(
    "/documents/{document_id}/versions/{version}",
    response_model=WorkspaceDocumentVersionWithContentResponse,
)
async def get_document_version(
    document_id: str,
    version: int,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> WorkspaceDocumentVersionWithContentResponse:
    """Get a specific historical version with full content_text."""
    effective_user_id = current_user.user_id
    document_store = _get_document_store(request)

    row = document_store.get_document_version(
        document_id=document_id, user_id=effective_user_id, version=version
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    return WorkspaceDocumentVersionWithContentResponse(**row)


@router.post(
    "/documents/{document_id}/versions/{version}/restore",
    response_model=WorkspaceDocumentResponse,
)
async def restore_document_version(
    document_id: str,
    version: int,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> WorkspaceDocumentResponse:
    """Restore a historical version: saves current state as a snapshot, then applies old content as new version."""
    effective_user_id = current_user.user_id
    document_store = _get_document_store(request)
    file_manager = _get_file_manager(request)

    existing = document_store.get_document(document_id=document_id, user_id=effective_user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    # Images are immutable and never versioned, so there is nothing to restore.
    if str(existing.get("mime_type") or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Images cannot be edited")

    old_version = document_store.get_document_version(
        document_id=document_id, user_id=effective_user_id, version=version
    )
    if not old_version:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")

    content_bytes = old_version["content_text"].encode("utf-8")
    try:
        updated_metadata = file_manager.update_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            stored_path=existing["stored_path"],
            content=content_bytes,
        )
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    updated = document_store.update_document_content(
        document_id=document_id,
        user_id=effective_user_id,
        content_text=old_version["content_text"],
        size_bytes=int(updated_metadata["size_bytes"]),
        sha256=str(updated_metadata["sha256"]),
        source="restore",
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to restore version")

    return WorkspaceDocumentResponse(**{k: v for k, v in updated.items() if k in WorkspaceDocumentResponse.model_fields})
