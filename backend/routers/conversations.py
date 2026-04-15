"""Conversations REST API – CRUD for conversations and messages."""

from __future__ import annotations

import logging
import uuid
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from backend.attachments import AttachmentValidationError, WorkspaceValidationError
from backend.single_user import get_effective_user_id, require_api_access

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/conversations",
    tags=["conversations"],
    dependencies=[Depends(require_api_access)],
)


# ── Request / Response Models ────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    user_id: str = Field(default="anonymous")
    title: str = Field(default="New conversation")
    language: str = Field(default="en")
    fork_name: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None
    pinned: Optional[bool] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    response_time_ms: Optional[int] = None
    tools_used: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageFeedbackRequest(BaseModel):
    feedback: Optional[str] = Field(None, pattern="^(up|down)$")


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    language: str
    fork_name: Optional[str] = None
    archived: bool
    pinned: bool
    metadata: Dict[str, Any] = {}
    last_message: Optional[str] = None
    message_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    response_time_ms: Optional[int] = None
    tools_used: Optional[List[Dict[str, Any]]] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[str] = None


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int
    limit: int
    offset: int


class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]


class SearchResultItem(BaseModel):
    message_id: int
    conversation_id: str
    conversation_title: str
    role: str
    content: str
    created_at: str


class AttachmentResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    status: str
    created_at: Optional[str] = None
    expires_at: Optional[str] = None


class AttachmentUploadResponse(BaseModel):
    attachment: AttachmentResponse


class AttachmentListResponse(BaseModel):
    attachments: List[AttachmentResponse]


class WorkspaceFileResponse(BaseModel):
    name: str
    relative_path: str
    size_bytes: int
    modified_at: Optional[str] = None


class WorkspaceResponse(BaseModel):
    conversation_id: str
    user_id: str
    root_path: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    expires_at: Optional[str] = None
    files: List[WorkspaceFileResponse] = []


class WorkspaceMaterializeRequest(BaseModel):
    attachment_id: str
    target_name: Optional[str] = None


class WorkspaceMaterializeResponse(BaseModel):
    workspace: WorkspaceResponse
    file: WorkspaceFileResponse


# ── Helper ────────────────────────────────────────────────────────────


def _get_store(request: Request):
    return request.app.state.conversation_store


def _get_attachment_store(request: Request):
    return request.app.state.conversation_attachment_store


def _get_attachment_manager(request: Request):
    return request.app.state.chat_attachment_manager


def _get_workspace_store(request: Request):
    return request.app.state.conversation_workspace_store


def _get_workspace_manager(request: Request):
    return request.app.state.chat_workspace_manager


def _workspace_enabled() -> bool:
    return os.getenv("CHAT_WORKSPACE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(body: CreateConversationRequest, request: Request):
    """Create a new conversation."""
    store = _get_store(request)
    conversation_id = str(uuid.uuid4())
    effective_user_id = get_effective_user_id(body.user_id)
    try:
        conv = store.create_conversation(
            conversation_id=conversation_id,
            user_id=effective_user_id,
            title=body.title,
            language=body.language,
            fork_name=body.fork_name,
        )
        return conv
    except Exception as exc:
        logger.error(f"Failed to create conversation: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    user_id: str = Query(default="anonymous"),
    include_archived: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List conversations for a user (pinned first, then most-recently updated)."""
    store = _get_store(request)
    effective_user_id = get_effective_user_id(user_id)
    conversations, total = store.list_conversations(
        user_id=effective_user_id,
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )
    return {
        "conversations": conversations,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/search", response_model=List[SearchResultItem])
async def search_messages(
    request: Request,
    user_id: str = Query(default="anonymous"),
    q: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Full-text search across all messages for a user."""
    store = _get_store(request)
    effective_user_id = get_effective_user_id(user_id)
    return store.search_messages(user_id=effective_user_id, query=q, limit=limit)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: str, request: Request):
    """Get a conversation with its full message history."""
    store = _get_store(request)
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = store.get_messages(conversation_id)
    return {"conversation": conv, "messages": messages}


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str, body: UpdateConversationRequest, request: Request
):
    """Update conversation fields (title, archived, pinned, language, metadata)."""
    store = _get_store(request)
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        conv = store.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    conv = store.update_conversation(conversation_id, **updates)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request):
    """Delete a conversation and all its messages."""
    store = _get_store(request)
    deleted = store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


# ── Attachments ───────────────────────────────────────────────────────


@router.post(
    "/{conversation_id}/attachments",
    response_model=AttachmentUploadResponse,
    status_code=201,
)
async def upload_attachment(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None),
):
    """Upload a text attachment for a conversation.
    
    The uploaded file is persisted directly as a workspace document and the
    conversation stores only a reference to that canonical document.
    """
    store = _get_store(request)
    attachment_store = _get_attachment_store(request)
    attachment_manager = _get_attachment_manager(request)
    file_manager = getattr(request.app.state, "user_workspace_file_manager", None)
    document_store = getattr(request.app.state, "workspace_document_store", None)

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    resolved_user_id = user_id or conv["user_id"]
    resolved_user_id = get_effective_user_id(resolved_user_id)
    if resolved_user_id != conv["user_id"]:
        raise HTTPException(status_code=403, detail="Conversation does not belong to user")

    if file_manager is None or document_store is None:
        raise HTTPException(status_code=500, detail="Workspace document services are unavailable")

    try:
        content = await file.read()
        safe_name, mime_type = attachment_manager.validate_upload(
            file.filename or "attachment.txt",
            content,
            file.content_type,
        )
        extracted_text = attachment_manager.extract_text(content)

        attachment_id = str(uuid.uuid4())
        stored_doc = file_manager.store_document_file(
            user_id=resolved_user_id,
            document_id=attachment_id,
            filename=safe_name,
            content=content,
        )
        document_store.create_document(
            document_id=attachment_id,
            user_id=resolved_user_id,
            filename=str(stored_doc["filename"]),
            stored_path=str(stored_doc["stored_path"]),
            mime_type=mime_type,
            size_bytes=int(stored_doc["size_bytes"]),
            sha256=str(stored_doc["sha256"]),
            content_text=extracted_text,
        )
        attachment = attachment_store.create_attachment(
            attachment_id=attachment_id,
            conversation_id=conversation_id,
            user_id=resolved_user_id,
            original_name=str(stored_doc["filename"]),
            stored_path=str(stored_doc["stored_path"]),
            mime_type=mime_type,
            size_bytes=int(stored_doc["size_bytes"]),
            sha256=str(stored_doc["sha256"]),
            extracted_text=extracted_text,
            expires_at=None,
        )

        return {"attachment": attachment}
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to upload attachment: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{conversation_id}/attachments", response_model=AttachmentListResponse)
async def list_attachments(conversation_id: str, request: Request):
    """List active attachments for a conversation."""
    store = _get_store(request)
    attachment_store = _get_attachment_store(request)

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"attachments": attachment_store.list_attachments(conversation_id)}


@router.delete("/{conversation_id}/attachments/{attachment_id}", status_code=204)
async def delete_attachment(conversation_id: str, attachment_id: str, request: Request):
    """Remove a document reference from a conversation."""
    store = _get_store(request)
    attachment_store = _get_attachment_store(request)

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    attachment = attachment_store.get_attachment(attachment_id, conversation_id=conversation_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")

    try:
        attachment_store.mark_attachment_deleted(attachment_id, conversation_id=conversation_id)
        return Response(status_code=204)
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to delete attachment: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/{conversation_id}/workspace/materialize",
    response_model=WorkspaceMaterializeResponse,
)
async def materialize_workspace_file(
    conversation_id: str,
    body: WorkspaceMaterializeRequest,
    request: Request,
):
    """Copy a linked document into the conversation workspace for safe editing."""
    if not _workspace_enabled():
        raise HTTPException(status_code=403, detail="Chat workspace editing is disabled")

    store = _get_store(request)
    attachment_store = _get_attachment_store(request)
    workspace_store = _get_workspace_store(request)
    workspace_manager = _get_workspace_manager(request)

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    attachment = attachment_store.get_attachment(body.attachment_id, conversation_id=conversation_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")

    try:
        copied = workspace_manager.materialize_attachment_copy(
            conversation_id=conversation_id,
            stored_attachment_path=attachment["stored_path"],
            target_name=body.target_name or attachment["original_name"],
        )

        workspace = workspace_store.upsert_workspace(
            conversation_id=conversation_id,
            user_id=conv["user_id"],
            root_path=str(copied["workspace_root"]),
            expires_at=copied["expires_at"],
        )

        files = workspace_manager.list_workspace_files(conversation_id)
        workspace["files"] = files

        return {
            "workspace": workspace,
            "file": {
                "name": copied["name"],
                "relative_path": copied["relative_path"],
                "size_bytes": copied["size_bytes"],
                "modified_at": copied["created_at"].isoformat() if hasattr(copied["created_at"], "isoformat") else None,
            },
        }
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to materialize workspace file: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{conversation_id}/workspace", response_model=WorkspaceResponse)
async def get_workspace(conversation_id: str, request: Request):
    """Get workspace metadata and files for a conversation."""
    if not _workspace_enabled():
        raise HTTPException(status_code=403, detail="Chat workspace editing is disabled")

    store = _get_store(request)
    workspace_store = _get_workspace_store(request)
    workspace_manager = _get_workspace_manager(request)

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    workspace = workspace_store.get_workspace(conversation_id)
    if not workspace:
        workspace_root = str(workspace_manager.get_workspace_dir(conversation_id))
        workspace = workspace_store.upsert_workspace(
            conversation_id=conversation_id,
            user_id=conv["user_id"],
            root_path=workspace_root,
        )

    workspace["files"] = workspace_manager.list_workspace_files(conversation_id)
    return workspace


# ── Messages ──────────────────────────────────────────────────────────


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=201,
)
async def add_message(
    conversation_id: str, body: AddMessageRequest, request: Request
):
    """Add a message to an existing conversation."""
    store = _get_store(request)
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        msg = store.add_message(
            conversation_id=conversation_id,
            role=body.role,
            content=body.content,
            tokens_used=body.tokens_used,
            model_name=body.model_name,
            response_time_ms=body.response_time_ms,
            tools_used=body.tools_used,
            metadata=body.metadata,
        )
        return msg
    except Exception as exc:
        logger.error(f"Failed to add message: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.patch(
    "/{conversation_id}/messages/{message_id}/feedback",
    response_model=MessageResponse,
)
async def set_message_feedback(
    conversation_id: str,
    message_id: int,
    body: MessageFeedbackRequest,
    request: Request,
):
    """Set feedback (thumbs up/down) on a message."""
    store = _get_store(request)
    try:
        msg = store.update_message_feedback(message_id, body.feedback)
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")
        return msg
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Export ────────────────────────────────────────────────────────────


@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    request: Request,
    fmt: str = Query(default="json", pattern="^(json|markdown)$"),
):
    """Export a conversation as JSON or Markdown."""
    store = _get_store(request)
    result = store.export_conversation(conversation_id, fmt=fmt)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if fmt == "markdown":
        return Response(
            content=result,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="conversation-{conversation_id}.md"'
            },
        )
    return result
