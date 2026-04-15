"""User workspace document management API."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.attachments import AttachmentValidationError, UserWorkspaceFileManager, WorkspaceValidationError
from backend.db import WorkspaceDocumentStore
from backend.single_user import get_effective_user_id, require_api_access

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


def _validate_text_file(filename: str, content_type: Optional[str] = None) -> str:
    """Validate text file format (extension + mime type).
    
    Returns:
        Normalized MIME type
    
    Raises:
        AttachmentValidationError: If file format is invalid
    """
    allowed_extensions = (".txt", ".md", ".markdown")
    allowed_mime_types = ("text/plain", "text/markdown", "text/x-markdown")
    
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    
    if ext not in allowed_extensions:
        raise AttachmentValidationError(
            f"File type '{ext or '(none)'}' not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    normalized_content_type = (content_type or "").strip().lower()
    if not normalized_content_type:
        normalized_content_type = "text/markdown" if ext == ".md" else "text/plain"
    
    if normalized_content_type not in allowed_mime_types:
        raise AttachmentValidationError(
            f"Content type '{normalized_content_type}' not allowed. Allowed: {', '.join(allowed_mime_types)}"
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


# ── API Endpoints ────────────────────────────────────────────────────────


@router.post("/documents", response_model=WorkspaceDocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Upload a new document to user workspace.
    
    Supports plain text and markdown files (.txt, .md, .markdown).
    Stores both filesystem copy and database metadata.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    
    effective_user_id = get_effective_user_id()
    document_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file format
        normalized_mime_type = _validate_text_file(file.filename or "document.txt", file.content_type)
        
        # Validate UTF-8 text
        content_text = _validate_utf8_text(content)
        
        # Store file in filesystem
        stored_metadata = file_manager.store_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            filename=file.filename or "document.txt",
            content=content,
        )
        
        # Create database record
        doc = document_store.create_document(
            document_id=document_id,
            user_id=effective_user_id,
            filename=file.filename or "document.txt",
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
) -> Dict[str, Any]:
    """List all documents in user workspace.
    
    Returns documents ordered by most recently updated first.
    """
    document_store = _get_document_store(request)
    effective_user_id = get_effective_user_id()
    
    try:
        documents = document_store.list_documents(
            user_id=effective_user_id,
            limit=limit,
            offset=offset,
        )
        return {
            "documents": documents,
            "total": len(documents),  # Note: accurate within limit; full count requires separate query
        }
    except Exception as exc:
        logger.error(f"Failed to list documents: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents/{document_id}", response_model=WorkspaceDocumentWithContentResponse)
async def get_document(
    document_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Fetch a document by ID with full content.
    
    Ownership is enforced: user can only access their own documents.
    """
    document_store = _get_document_store(request)
    effective_user_id = get_effective_user_id()
    
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
) -> Dict[str, Any]:
    """Overwrite a document with new content.
    
    Updates the file on disk, increments version, and returns updated metadata.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = get_effective_user_id()
    
    try:
        # Verify document exists and belongs to user
        existing_doc = document_store.get_document(document_id=document_id, user_id=effective_user_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Read and validate new content
        content = await file.read()
        normalized_mime_type = _validate_text_file(file.filename or existing_doc["filename"], file.content_type)
        content_text = _validate_utf8_text(content)
        
        # Update file on disk
        updated_metadata = file_manager.update_document_file(
            user_id=effective_user_id,
            document_id=document_id,
            stored_path=existing_doc["stored_path"],
            content=content,
        )
        
        # Update database record (overwrite + version increment)
        new_sha256 = hashlib.sha256(content).hexdigest()
        updated_doc = document_store.update_document_content(
            document_id=document_id,
            user_id=effective_user_id,
            content_text=content_text,
            size_bytes=int(updated_metadata["size_bytes"]),
            sha256=new_sha256,
        )
        
        if not updated_doc:
            raise HTTPException(status_code=500, detail="Failed to update document")
        
        return {"document": updated_doc}
    
    except HTTPException:
        raise
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to update document: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/documents/{document_id}", status_code=204, response_class=Response)
async def delete_document(
    document_id: str,
    request: Request,
) -> Response:
    """Delete a document (hard delete).
    
    Removes from filesystem and database.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = get_effective_user_id()
    
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
) -> FileResponse:
    """Download current saved content of a document.
    
    Returns the exact file as stored, with appropriate Content-Disposition header.
    """
    file_manager = _get_file_manager(request)
    document_store = _get_document_store(request)
    effective_user_id = get_effective_user_id()
    
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
