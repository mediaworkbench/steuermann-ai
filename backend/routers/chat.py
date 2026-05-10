import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    config_from_env,
)
from backend.attachments import UserWorkspaceFileManager, WorkspaceValidationError
from backend.db import SettingsStore
from backend.rate_limit import limiter
from backend.single_user import get_effective_user_id, require_api_access
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.llm.provider_registry import normalize_model_id, parse_model_id
from universal_agentic_framework.monitoring.metrics import (
    track_workspace_intent_denied,
    track_profile_id_mismatch,
    track_workspace_created,
    track_workspace_revised_copy_created,
    track_workspace_write_allowed,
    track_workspace_write_denied,
    track_memory_retrieval_signal,
)
from universal_agentic_framework.tools.file_ops.tool import WorkspaceFileOpsTool

logger = logging.getLogger(__name__)
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8000")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
PROFILE_ID = os.getenv("PROFILE_ID", "starter")
ACTIVE_PROFILE_ID = os.getenv("ACTIVE_PROFILE_ID", PROFILE_ID)

# Simple in-memory cache for user settings (TTL: 5 minutes)
_settings_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
_SETTINGS_CACHE_TTL = 300  # seconds

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(require_api_access)])

LANGGRAPH_CIRCUIT_BREAKER = AsyncCircuitBreaker(
    name="langgraph_invoke",
    config=config_from_env(
        "LANGGRAPH_CIRCUIT",
        CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30.0, half_open_max_calls=1),
    ),
)

MODEL_VALIDATION_CIRCUIT_BREAKER = AsyncCircuitBreaker(
    name="llm_model_validation",
    config=config_from_env(
        "MODEL_VALIDATION_CIRCUIT",
        CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30.0, half_open_max_calls=1),
    ),
)


_USER_ID_RE = re.compile(r'^[a-zA-Z0-9_@.\-]{1,128}$')
_LANG_RE = re.compile(r'^[a-zA-Z]{2,5}$')
_WORKSPACE_DOC_ID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_WORKSPACE_DOC_FILENAME_QUOTED_RE = re.compile(r'["`\']([^"`\'\n]{1,255})["`\']')
_WORKSPACE_DOC_FILENAME_HINT_RE = re.compile(
    r"\bworkspace\s+document\s+[\"']?([^\"'\n]{1,255}?)\b(?:\s*\(|\s|$)",
    re.IGNORECASE,
)
_EXPLICIT_EDIT_INTENT_PATTERNS = [
    # English
    re.compile(r"\b(rewrite|revise|edit|redraft|refactor)\b", re.IGNORECASE),
    re.compile(r"\b(create|make)\s+(a\s+)?revised\s+(copy|version)\b", re.IGNORECASE),
    re.compile(r"\bupdate\b.*\b(file|document|attachment|draft|markdown|text)\b", re.IGNORECASE),
    # German
    re.compile(r"\b(uberarbeite|überarbeite|bearbeite|umschreiben|neu\s*fassen|aktualisiere)\b", re.IGNORECASE),
    re.compile(r"\berstelle\b.*\b(uberarbeitete|revidierte)\b.*\b(version|kopie)\b", re.IGNORECASE),
]
_WORKSPACE_SAVE_INTENT_PATTERNS = [
    re.compile(r"\b(save|write\s+back|overwrite|replace|persist|update)\b.*\b(workspace|document|file)\b", re.IGNORECASE),
    re.compile(r"\b(save|overwrite|replace)\s+it\s+back\b", re.IGNORECASE),
    re.compile(r"\b(im\s+workspace\s+speichern|im\s+workspace\s+ueberschreiben|im\s+workspace\s+überschreiben)\b", re.IGNORECASE),
    re.compile(r"\b(speicher|speichern|ueberschreibe|überschreibe|aktualisiere)\b.*\b(workspace|dokument|datei)\b", re.IGNORECASE),
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    user_id: str = Field(default="anonymous", max_length=128)
    language: str = Field(default="en", min_length=2, max_length=10)
    conversation_id: Optional[str] = Field(default=None, max_length=36)
    attachment_ids: List[str] = Field(default_factory=list, max_length=20)
    document_ids: List[str] = Field(default_factory=list, max_length=20)
    preferred_model: Optional[str] = Field(default=None, max_length=256)
    workspace_action: Optional["WorkspaceActionRequest"] = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not _USER_ID_RE.match(v):
            raise ValueError("user_id contains invalid characters")
        return v

    @field_validator("message", mode="before")
    @classmethod
    def sanitize_message(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if not _LANG_RE.match(v):
            raise ValueError("Invalid language code")
        return v.lower()


class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]


class WorkspaceActionRequest(BaseModel):
    operation: Literal["copy_to_workspace", "read_workspace_file", "write_workspace_file", "write_revised_copy"]
    attachment_id: Optional[str] = None
    path: Optional[str] = None
    target_name: Optional[str] = None
    content: Optional[str] = None


def _resolve_request_attachments(
    request: Request,
    request_body: ChatRequest,
    effective_user_id: str,
) -> list[dict[str, Any]]:
    attachment_ids = request_body.attachment_ids or []
    if not attachment_ids:
        return []

    if not request_body.conversation_id:
        raise HTTPException(
            status_code=400,
            detail="attachment_ids require conversation_id",
        )

    attachment_store = getattr(request.app.state, "conversation_attachment_store", None)
    if attachment_store is None:
        raise HTTPException(status_code=500, detail="Attachment store unavailable")

    attachments = attachment_store.get_attachments_by_ids(
        request_body.conversation_id,
        attachment_ids,
    )
    if len(attachments) != len(attachment_ids):
        found_ids = {attachment["id"] for attachment in attachments}
        missing_ids = [attachment_id for attachment_id in attachment_ids if attachment_id not in found_ids]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attachment_ids for conversation: {', '.join(missing_ids)}",
        )

    if any(attachment.get("user_id") != effective_user_id for attachment in attachments):
        raise HTTPException(status_code=403, detail="Attachment does not belong to user")

    return attachments


def _resolve_workspace_documents(
    request: Request,
    request_body: ChatRequest,
    effective_user_id: str,
    attachments: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Resolve workspace document IDs to full document content."""
    explicit_document_ids = list(dict.fromkeys(request_body.document_ids or []))
    document_store = getattr(request.app.state, "workspace_document_store", None)
    if document_store is None:
        if explicit_document_ids:
            raise HTTPException(status_code=500, detail="Workspace document store unavailable")
        return []

    documents_by_id: dict[str, dict[str, Any]] = {}

    if explicit_document_ids:
        explicit_documents = document_store.get_documents_by_ids(
            user_id=effective_user_id,
            document_ids=explicit_document_ids,
        )
        if len(explicit_documents) != len(explicit_document_ids):
            found_ids = {doc["id"] for doc in explicit_documents}
            missing_ids = [doc_id for doc_id in explicit_document_ids if doc_id not in found_ids]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document_ids: {', '.join(missing_ids)}",
            )
        documents_by_id.update({doc["id"]: doc for doc in explicit_documents})

    inferred_document_ids = _infer_workspace_document_ids_from_message(
        message=request_body.message,
        user_id=effective_user_id,
        document_store=document_store,
    )

    if not inferred_document_ids and attachments:
        attachment_names: list[str] = []
        for attachment in attachments:
            original_name = str(attachment.get("original_name") or "").strip().lower()
            if not original_name or "." not in original_name:
                continue
            if original_name not in attachment_names:
                attachment_names.append(original_name)

        if attachment_names:
            available_documents = document_store.list_documents(user_id=effective_user_id, limit=200, offset=0)
            docs_by_filename = {
                str(doc.get("filename") or "").strip().lower(): doc["id"]
                for doc in available_documents
                if doc.get("filename")
            }
            for attachment_name in attachment_names:
                matched_id = docs_by_filename.get(attachment_name)
                if matched_id and matched_id not in inferred_document_ids:
                    inferred_document_ids.append(matched_id)

    if not inferred_document_ids:
        inferred_document_ids = _infer_workspace_document_ids_from_recent_conversation_context(
            request=request,
            request_body=request_body,
            user_id=effective_user_id,
            document_store=document_store,
        )

    for inferred_id in inferred_document_ids:
        if inferred_id in documents_by_id:
            continue
        inferred_doc = document_store.get_document(document_id=inferred_id, user_id=effective_user_id)
        if inferred_doc:
            documents_by_id[inferred_id] = inferred_doc

    ordered_ids = explicit_document_ids + [doc_id for doc_id in inferred_document_ids if doc_id not in explicit_document_ids]
    if not ordered_ids:
        return []

    return [documents_by_id[doc_id] for doc_id in ordered_ids if doc_id in documents_by_id]


def _infer_workspace_document_ids_from_recent_conversation_context(
    request: Request,
    request_body: ChatRequest,
    user_id: str,
    document_store: Any,
) -> list[str]:
    """Fallback: reuse the most recently used workspace document in the conversation.

    This supports iterative follow-up edits like "elaborate that part and save again"
    when users omit both filename and attachment selection in subsequent turns.
    """
    conversation_id = (request_body.conversation_id or "").strip()
    if not conversation_id:
        return []

    conversation_store = getattr(request.app.state, "conversation_store", None)
    if conversation_store is None or not hasattr(conversation_store, "get_messages"):
        return []

    try:
        messages = conversation_store.get_messages(conversation_id, limit=50, offset=0)
    except Exception:
        return []

    for message in reversed(messages or []):
        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        writeback = metadata.get("workspace_document_writeback") or {}
        if isinstance(writeback, dict):
            writeback_doc_id = writeback.get("document_id")
            if isinstance(writeback_doc_id, str) and writeback_doc_id:
                doc = document_store.get_document(document_id=writeback_doc_id, user_id=user_id)
                if doc:
                    return [writeback_doc_id]

        documents_used = metadata.get("documents_used") or []
        if isinstance(documents_used, list):
            for doc_entry in documents_used:
                if not isinstance(doc_entry, dict):
                    continue
                doc_id = doc_entry.get("id")
                if not isinstance(doc_id, str) or not doc_id:
                    continue
                doc = document_store.get_document(document_id=doc_id, user_id=user_id)
                if doc:
                    return [doc_id]

    return []


def _infer_workspace_document_ids_from_message(
    message: str,
    user_id: str,
    document_store: Any,
) -> list[str]:
    text = (message or "").strip()
    if not text:
        return []

    documents = document_store.list_documents(user_id=user_id, limit=200, offset=0)
    if not documents:
        return []

    docs_by_id = {doc["id"].lower(): doc["id"] for doc in documents}
    docs_by_filename = {str(doc["filename"]).lower(): doc["id"] for doc in documents}

    inferred_ids: list[str] = []

    for raw_id in _WORKSPACE_DOC_ID_RE.findall(text):
        doc_id = docs_by_id.get(raw_id.lower())
        if doc_id and doc_id not in inferred_ids:
            inferred_ids.append(doc_id)

    quoted_candidates = [candidate.strip() for candidate in _WORKSPACE_DOC_FILENAME_QUOTED_RE.findall(text)]
    hinted_candidates = [candidate.strip() for candidate in _WORKSPACE_DOC_FILENAME_HINT_RE.findall(text)]
    for filename in quoted_candidates + hinted_candidates:
        if "." not in filename:
            continue
        doc_id = docs_by_filename.get(filename.lower())
        if doc_id and doc_id not in inferred_ids:
            inferred_ids.append(doc_id)

    text_folded = text.casefold()
    for filename, doc_id in docs_by_filename.items():
        if doc_id in inferred_ids:
            continue
        if "." not in filename:
            continue
        if re.search(rf"(?<!\w){re.escape(filename)}(?!\w)", text_folded, flags=re.IGNORECASE):
            inferred_ids.append(doc_id)

    return inferred_ids


def _workspace_enabled() -> bool:
    return os.getenv("CHAT_WORKSPACE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _workspace_retention_hours() -> int:
    raw = os.getenv("CHAT_WORKSPACE_RETENTION_HOURS", "24")
    try:
        return max(1, int(raw))
    except ValueError:
        return 24


def _workspace_expiration() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=_workspace_retention_hours())


def _serialize_loaded_memories(loaded_memory: Any) -> list[dict[str, Any]]:
    """Convert graph loaded_memory entries into frontend-safe metadata."""
    if not isinstance(loaded_memory, list):
        return []

    out: list[dict[str, Any]] = []
    for item in loaded_memory:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        memory_id = metadata.get("memory_id")
        if not memory_id:
            continue
        out.append(
            {
                "memory_id": str(memory_id),
                "text": str(item.get("text") or ""),
                "user_rating": metadata.get("user_rating"),
                "importance_score": metadata.get("importance_score"),
                "is_related": bool(metadata.get("is_related", False)),
            }
        )
    return out


def _workspace_error_to_http(detail: str) -> HTTPException:
    if "does not exist" in detail or "not found" in detail:
        return HTTPException(status_code=404, detail=detail)
    if "does not belong" in detail:
        return HTTPException(status_code=403, detail=detail)
    return HTTPException(status_code=400, detail=detail)


def _has_explicit_workspace_edit_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _EXPLICIT_EDIT_INTENT_PATTERNS)


def _has_workspace_save_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _WORKSPACE_SAVE_INTENT_PATTERNS)


def _normalize_workspace_writeback_content(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    fenced_match = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?\n?(.*?)\n?```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    return text


def _get_user_workspace_file_manager(request: Request) -> UserWorkspaceFileManager:
    manager = getattr(request.app.state, "user_workspace_file_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="Workspace file manager unavailable")
    return manager


def _write_response_back_to_workspace_document(
    request: Request,
    *,
    effective_user_id: str,
    document: dict[str, Any],
    content_text: str,
) -> dict[str, Any]:
    document_store = getattr(request.app.state, "workspace_document_store", None)
    if document_store is None:
        raise HTTPException(status_code=500, detail="Workspace document store unavailable")

    file_manager = _get_user_workspace_file_manager(request)
    normalized_content = _normalize_workspace_writeback_content(content_text)
    if not normalized_content:
        raise HTTPException(status_code=400, detail="Refusing to save empty workspace document content")

    try:
        updated_metadata = file_manager.update_document_file(
            user_id=effective_user_id,
            document_id=str(document["id"]),
            stored_path=str(document["stored_path"]),
            content=normalized_content.encode("utf-8"),
        )
        updated_document = document_store.update_document_content(
            document_id=str(document["id"]),
            user_id=effective_user_id,
            content_text=normalized_content,
            size_bytes=int(updated_metadata["size_bytes"]),
            sha256=str(updated_metadata["sha256"]),
        )
    except WorkspaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not updated_document:
        raise HTTPException(status_code=404, detail="Workspace document not found for writeback")

    return {
        "status": "saved",
        "document_id": updated_document["id"],
        "filename": updated_document["filename"],
        "version": updated_document["version"],
        "size_bytes": updated_document["size_bytes"],
    }


def _execute_workspace_action(
    request: Request,
    request_body: ChatRequest,
    effective_user_id: str,
) -> dict[str, Any]:
    action = request_body.workspace_action
    if action is None:
        raise HTTPException(status_code=400, detail="workspace_action is required")

    if not _workspace_enabled():
        raise HTTPException(status_code=403, detail="Chat workspace editing is disabled")

    if not request_body.conversation_id:
        raise HTTPException(status_code=400, detail="workspace_action requires conversation_id")

    conversation_id = request_body.conversation_id
    conversation_store = getattr(request.app.state, "conversation_store", None)
    attachment_store = getattr(request.app.state, "conversation_attachment_store", None)
    workspace_store = getattr(request.app.state, "conversation_workspace_store", None)
    workspace_manager = getattr(request.app.state, "chat_workspace_manager", None)

    if not conversation_store or not attachment_store or not workspace_store or not workspace_manager:
        raise HTTPException(status_code=500, detail="Workspace services unavailable")

    def _log_workspace_operation(
        operation: str,
        result: Literal["allowed", "denied", "failed"],
        *,
        attachment_id: Optional[str] = None,
        reason: Optional[str] = None,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ) -> None:
        try:
            workspace_store.log_workspace_operation(
                conversation_id=conversation_id,
                user_id=effective_user_id,
                attachment_id=attachment_id,
                operation=operation,
                result=result,
                reason=reason,
                source_path=source_path,
                target_path=target_path,
            )
        except Exception as exc:
            logger.warning("Failed to log workspace operation: %s", exc)

    conv = conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.get("user_id") != effective_user_id:
        raise HTTPException(status_code=403, detail="Conversation does not belong to user")

    if action.operation == "copy_to_workspace":
        if not action.attachment_id:
            raise HTTPException(status_code=400, detail="copy_to_workspace requires attachment_id")

        attachment = attachment_store.get_attachment(action.attachment_id, conversation_id=conversation_id)
        if not attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")
        if attachment.get("user_id") != effective_user_id:
            raise HTTPException(status_code=403, detail="Attachment does not belong to user")

        try:
            existing_workspace = workspace_store.get_workspace(conversation_id, include_inactive=True)
            copied = workspace_manager.materialize_attachment_copy(
                conversation_id=conversation_id,
                stored_attachment_path=attachment["stored_path"],
                target_name=action.target_name or attachment["original_name"],
            )
            workspace = workspace_store.upsert_workspace(
                conversation_id=conversation_id,
                user_id=effective_user_id,
                root_path=str(copied["workspace_root"]),
                expires_at=_workspace_expiration(),
            )
            workspace["files"] = workspace_manager.list_workspace_files(conversation_id)

            if existing_workspace is None:
                track_workspace_created(PROFILE_ID)

            _log_workspace_operation(
                "copy_to_workspace",
                "allowed",
                attachment_id=attachment.get("id"),
                source_path=attachment.get("stored_path"),
                target_path=str(copied["path"]),
            )
        except Exception as exc:
            _log_workspace_operation(
                "copy_to_workspace",
                "failed",
                attachment_id=attachment.get("id"),
                reason=str(exc),
                source_path=attachment.get("stored_path"),
            )
            raise _workspace_error_to_http(str(exc)) from exc

        return {
            "operation": action.operation,
            "workspace": workspace,
            "file": {
                "name": copied["name"],
                "relative_path": copied["relative_path"],
                "size_bytes": copied["size_bytes"],
                "modified_at": copied["created_at"].isoformat(),
            },
            "response": f"Copied attachment to workspace: {copied['relative_path']}",
        }

    if action.operation == "read_workspace_file":
        if not action.path:
            raise HTTPException(status_code=400, detail="read_workspace_file requires path")

        workspace_dir = workspace_manager.get_workspace_dir(conversation_id)
        tool = WorkspaceFileOpsTool(sandbox_dir=str(workspace_dir))
        try:
            output = tool._run(operation="read", path=action.path)
            if output.startswith("Error:"):
                raise _workspace_error_to_http(output.replace("Error:", "").strip())

            workspace = workspace_store.touch_workspace(
                conversation_id=conversation_id,
                expires_at=_workspace_expiration(),
            )
            if workspace is None:
                workspace = workspace_store.upsert_workspace(
                    conversation_id=conversation_id,
                    user_id=effective_user_id,
                    root_path=str(workspace_dir),
                    expires_at=_workspace_expiration(),
                )
                track_workspace_created(PROFILE_ID)

            workspace["files"] = workspace_manager.list_workspace_files(conversation_id)

            _log_workspace_operation(
                "read_workspace_file",
                "allowed",
                target_path=action.path,
            )
        except HTTPException as exc:
            _log_workspace_operation(
                "read_workspace_file",
                "failed",
                reason=str(exc.detail),
                target_path=action.path,
            )
            raise
        except Exception as exc:
            _log_workspace_operation(
                "read_workspace_file",
                "failed",
                reason=str(exc),
                target_path=action.path,
            )
            raise

        return {
            "operation": action.operation,
            "workspace": workspace,
            "path": action.path,
            "response": output,
        }

    if action.operation == "write_workspace_file":
        if not _has_explicit_workspace_edit_intent(request_body.message):
            track_workspace_write_denied(PROFILE_ID)
            track_workspace_intent_denied(
                PROFILE_ID,
                operation="write_workspace_file",
                reason="explicit_edit_intent_required",
            )
            _log_workspace_operation(
                "write_workspace_file",
                "denied",
                reason="explicit_edit_intent_required",
                target_path=action.path,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "write_workspace_file requires explicit edit intent in the message "
                    "(for example: 'rewrite this file' or 'edit this document')."
                ),
            )
        if not action.path:
            raise HTTPException(status_code=400, detail="write_workspace_file requires path")
        if action.content is None:
            raise HTTPException(status_code=400, detail="write_workspace_file requires content")

        workspace_dir = workspace_manager.get_workspace_dir(conversation_id)
        tool = WorkspaceFileOpsTool(sandbox_dir=str(workspace_dir))
        try:
            existing_workspace = workspace_store.get_workspace(conversation_id, include_inactive=True)

            output = tool._run(operation="write", path=action.path, content=action.content)
            if output.startswith("Error:"):
                raise _workspace_error_to_http(output.replace("Error:", "").strip())

            workspace = workspace_store.touch_workspace(
                conversation_id=conversation_id,
                expires_at=_workspace_expiration(),
            )
            if workspace is None:
                workspace = workspace_store.upsert_workspace(
                    conversation_id=conversation_id,
                    user_id=effective_user_id,
                    root_path=str(workspace_dir),
                    expires_at=_workspace_expiration(),
                )
            workspace["files"] = workspace_manager.list_workspace_files(conversation_id)

            if existing_workspace is None:
                track_workspace_created(PROFILE_ID)
            track_workspace_write_allowed(PROFILE_ID)

            _log_workspace_operation(
                "write_workspace_file",
                "allowed",
                source_path=action.path,
                target_path=action.path,
            )
        except HTTPException as exc:
            track_workspace_write_denied(PROFILE_ID)
            _log_workspace_operation(
                "write_workspace_file",
                "failed",
                reason=str(exc.detail),
                source_path=action.path,
            )
            raise
        except Exception as exc:
            track_workspace_write_denied(PROFILE_ID)
            _log_workspace_operation(
                "write_workspace_file",
                "failed",
                reason=str(exc),
                source_path=action.path,
            )
            raise

        return {
            "operation": action.operation,
            "workspace": workspace,
            "path": action.path,
            "response": output,
        }

    if action.operation == "write_revised_copy":
        if not _has_explicit_workspace_edit_intent(request_body.message):
            track_workspace_write_denied(PROFILE_ID)
            track_workspace_intent_denied(
                PROFILE_ID,
                operation="write_revised_copy",
                reason="explicit_edit_intent_required",
            )
            _log_workspace_operation(
                "write_revised_copy",
                "denied",
                reason="explicit_edit_intent_required",
                target_path=action.path,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "write_revised_copy requires explicit edit intent in the message "
                    "(for example: 'rewrite the uploaded markdown file' or 'create a revised copy')."
                ),
            )
        if not action.path:
            raise HTTPException(status_code=400, detail="write_revised_copy requires path")
        if action.content is None:
            raise HTTPException(status_code=400, detail="write_revised_copy requires content")

        workspace_dir = workspace_manager.get_workspace_dir(conversation_id)
        tool = WorkspaceFileOpsTool(sandbox_dir=str(workspace_dir))
        try:
            existing_workspace = workspace_store.get_workspace(conversation_id, include_inactive=True)

            output = tool._run(operation="write_revised", path=action.path, content=action.content)
            if output.startswith("Error:"):
                raise _workspace_error_to_http(output.replace("Error:", "").strip())

            revised_relative_path = None
            match = re.search(r"Written revised copy:\s+(.+)\s+\(\d[\d,]*\s+bytes\)$", output)
            if match:
                revised_relative_path = match.group(1).strip()

            workspace = workspace_store.touch_workspace(
                conversation_id=conversation_id,
                expires_at=_workspace_expiration(),
            )
            if workspace is None:
                workspace = workspace_store.upsert_workspace(
                    conversation_id=conversation_id,
                    user_id=effective_user_id,
                    root_path=str(workspace_dir),
                    expires_at=_workspace_expiration(),
                )
            workspace["files"] = workspace_manager.list_workspace_files(conversation_id)

            if existing_workspace is None:
                track_workspace_created(PROFILE_ID)
            track_workspace_write_allowed(PROFILE_ID)
            track_workspace_revised_copy_created(PROFILE_ID)

            _log_workspace_operation(
                "write_revised_copy",
                "allowed",
                source_path=action.path,
                target_path=revised_relative_path,
            )
        except HTTPException as exc:
            track_workspace_write_denied(PROFILE_ID)
            _log_workspace_operation(
                "write_revised_copy",
                "failed",
                reason=str(exc.detail),
                source_path=action.path,
            )
            raise
        except Exception as exc:
            track_workspace_write_denied(PROFILE_ID)
            _log_workspace_operation(
                "write_revised_copy",
                "failed",
                reason=str(exc),
                source_path=action.path,
            )
            raise

        return {
            "operation": action.operation,
            "workspace": workspace,
            "path": action.path,
            "revised_path": revised_relative_path,
            "response": output,
        }

    raise HTTPException(status_code=400, detail=f"Unsupported workspace operation: {action.operation}")


def _get_cached_settings(user_id: str, settings_store: SettingsStore) -> Dict[str, Any]:
    """Load user settings with cache (5min TTL)."""
    now = time.time()
    
    # Check cache
    if user_id in _settings_cache:
        settings, timestamp = _settings_cache[user_id]
        if now - timestamp < _SETTINGS_CACHE_TTL:
            return settings
    
    # Cache miss or expired - load from DB
    settings = settings_store.get_user_settings(user_id)
    if settings is None:
        # Return defaults
        settings = {
            "tool_toggles": {},
            "rag_config": {"collection": "", "top_k": 5},
            "preferred_model": None,
            "theme": "auto",
            "language": "en",
        }
    
    # Cache it
    _settings_cache[user_id] = (settings, now)
    return settings


def _get_latest_llm_capability_probes(request: Request) -> list[dict[str, Any]]:
    """Load latest persisted probe results grouped by provider.

    Returns most recent row per provider_id for the active profile.
    """
    probe_store = getattr(request.app.state, "llm_capability_probe_store", None)
    if probe_store is None:
        return []

    profile_id = ACTIVE_PROFILE_ID or PROFILE_ID
    try:
        rows = probe_store.list_probe_results(profile_id=profile_id, limit=100)
    except Exception as exc:
        logger.warning("Failed to load LLM capability probes", extra={"error": str(exc)})
        return []

    latest_by_provider: dict[str, dict[str, Any]] = {}
    for row in rows:
        provider_id = str(row.get("provider_id") or "").strip()
        if not provider_id or provider_id in latest_by_provider:
            continue
        latest_by_provider[provider_id] = {
            "provider_id": provider_id,
            "model_name": row.get("model_name"),
            "configured_tool_calling_mode": row.get("configured_tool_calling_mode"),
            "supports_bind_tools": row.get("supports_bind_tools"),
            "supports_tool_schema": row.get("supports_tool_schema"),
            "capability_mismatch": bool(row.get("capability_mismatch", False)),
            "status": row.get("status"),
            "error_message": row.get("error_message"),
            "probed_at": row.get("probed_at"),
        }

    return list(latest_by_provider.values())


async def _validate_preferred_model(model_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Validate preferred model and return canonical provider-prefixed id.

    Supports OpenAI-compatible /models endpoints and Ollama /api/tags.
    Returns (validated_model, warning).
    """
    if not model_name:
        return None, None

    def _default_provider_prefix() -> str:
        try:
            cfg = load_core_config()
            default_model = cfg.llm.providers.primary.models.en
            if default_model:
                return parse_model_id(str(default_model)).provider
        except Exception:
            pass
        return "openai"

    provider_prefix = _default_provider_prefix()

    known_prefixes = {
        "openai",
        "ollama",
        "lm_studio",
        "anthropic",
        "azure",
        "bedrock",
        "groq",
        "mistral",
        "vertex_ai",
    }

    def _canonical_from_endpoint_id(raw_name: str) -> str:
        raw_name = str(raw_name).strip()
        if not raw_name:
            return raw_name
        if "/" not in raw_name:
            return f"{provider_prefix}/{raw_name}"
        head = raw_name.split("/", 1)[0].strip().lower()
        if head in known_prefixes:
            return raw_name
        return f"{provider_prefix}/{raw_name}"

    try:
        normalized_request = normalize_model_id(model_name)
    except Exception:
        normalized_request = f"{provider_prefix}/{str(model_name).strip()}"

    def _canonical_model_aliases(available_models: list[str]) -> dict[str, str]:
        """Build alias -> canonical model map.

        Canonical form uses provider-prefixed names (for LiteLLM compatibility).
        """
        aliases: dict[str, str] = {}
        for available in available_models:
            if not available:
                continue

            canonical = _canonical_from_endpoint_id(available)

            aliases[available] = canonical
            if "/" in available:
                rest = available.split("/", 1)[1]
                aliases.setdefault(rest, canonical)
                aliases.setdefault(f"{provider_prefix}/{rest}", f"{provider_prefix}/{rest}")
            else:
                aliases.setdefault(f"{provider_prefix}/{available}", canonical)

        return aliases

    def _request_aliases(name: str) -> list[str]:
        candidates = [name]
        if "/" in name:
            rest = name.split("/", 1)[1]
            candidates.append(rest)
            candidates.append(f"{provider_prefix}/{rest}")
        else:
            candidates.append(f"{provider_prefix}/{name}")

        # Preserve insertion order while removing duplicates.
        return list(dict.fromkeys(candidates))

    async def _fetch_models() -> list[str]:
        async with httpx.AsyncClient(timeout=3.0) as client:
            base = str(LLM_ENDPOINT).rstrip("/")
            try:
                resp = await client.get(f"{base}/models")
                resp.raise_for_status()
                data = resp.json()
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
            except Exception:
                # Ollama native API fallback.
                if provider_prefix != "ollama":
                    raise
                resp = await client.get(f"{base}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")]

    try:
        available_models = await MODEL_VALIDATION_CIRCUIT_BREAKER.call(_fetch_models)
        alias_map = _canonical_model_aliases(available_models)
        for candidate in _request_aliases(normalized_request):
            if candidate in alias_map:
                return alias_map[candidate], None

        warning = f"Preferred model '{normalized_request}' not found at configured LLM endpoint. Using default."
        logger.warning(warning)
        return None, warning
    except CircuitBreakerOpenError as exc:
        logger.warning("Model validation circuit breaker open: %s", exc)
        # Fail open: do not block request flow when model endpoint is flaky.
        return normalized_request, "Model validation temporarily bypassed due to circuit breaker."
    except Exception as e:
        logger.error(f"Failed to validate model against OpenAI-compatible endpoint: {e}")
        # Fail open - assume model is valid if we can't reach endpoint
        return normalized_request, None


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, request_body: ChatRequest) -> ChatResponse:
    """Route a chat request to the LangGraph container."""
    start_time = time.time()
    effective_user_id = get_effective_user_id(request_body.user_id)

    if request_body.workspace_action is not None:
        workspace_result = _execute_workspace_action(request, request_body, effective_user_id)

        return ChatResponse(
            response=workspace_result.pop("response"),
            metadata={
                "workspace_action": workspace_result,
                "tools_executed": ["workspace_file_ops_tool"],
            },
        )

    # Load user settings (with cache)
    settings_store = getattr(request.app.state, "settings_store", None)
    user_settings = {}
    model_warning = None
    
    if settings_store:
        user_settings = _get_cached_settings(effective_user_id, settings_store)
        
        # Validate preferred model if set
        if user_settings.get("preferred_model"):
            validated_model, warning = await _validate_preferred_model(user_settings["preferred_model"])
            if warning:
                model_warning = warning
            # Update settings with validated model (or None if invalid)
            user_settings["preferred_model"] = validated_model
    
    # Language source of truth: user settings → API request fallback → profile default
    effective_language = user_settings.get("language") or request_body.language or "en"
    attachments = _resolve_request_attachments(request, request_body, effective_user_id)
    documents = _resolve_workspace_documents(request, request_body, effective_user_id, attachments=attachments)
    workspace_writeback_requested = _has_workspace_save_intent(request_body.message)
    llm_capability_probes = _get_latest_llm_capability_probes(request)
    
    state = {
        "messages": [{"role": "user", "content": request_body.message}],
        "user_id": effective_user_id,
        "language": effective_language,
        "user_settings": user_settings,  # Forward settings to LangGraph
        "llm_capability_probes": llm_capability_probes,
        "attachments": [
            {
                "id": attachment["id"],
                "original_name": attachment["original_name"],
                "mime_type": attachment["mime_type"],
                "size_bytes": attachment["size_bytes"],
                "extracted_text": attachment["extracted_text"],
            }
            for attachment in attachments
        ],
        "workspace_documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "stored_path": doc["stored_path"],
                "mime_type": doc["mime_type"],
                "size_bytes": doc["size_bytes"],
                "version": doc["version"],
                "content_text": doc["content_text"],
            }
            for doc in documents
        ],
        "workspace_writeback_requested": workspace_writeback_requested,
    }

    logger.info(f"Routing chat request to {LANGGRAPH_URL}/invoke", extra={
        "user_id": effective_user_id,
        "message_length": len(request_body.message),
        "language": effective_language,
        "has_preferred_model": bool(user_settings.get("preferred_model")),
        "probe_result_count": len(llm_capability_probes),
        "attachment_count": len(attachments),
        "document_count": len(documents),
    })

    result: dict[str, Any] = {}
    try:
        headers: Dict[str, str] = {}

        async def _invoke_langgraph() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=360.0) as client:
                response = await client.post(f"{LANGGRAPH_URL}/invoke", json=state, headers=headers)
                response.raise_for_status()
                return response.json()

        result = await LANGGRAPH_CIRCUIT_BREAKER.call(_invoke_langgraph)
    except CircuitBreakerOpenError as exc:
        logger.error("LangGraph circuit breaker open: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LangGraph temporarily unavailable (circuit breaker open). Please retry shortly.",
        ) from exc
    except httpx.RequestError as exc:
        logger.error(f"LangGraph HTTP request failed: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LangGraph request failed: {str(exc)}") from exc
    except httpx.HTTPStatusError as exc:
        logger.error(f"LangGraph HTTP error {exc.response.status_code}: {exc.response.text}", exc_info=True)
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"LangGraph error: {exc.response.text}",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected error calling LangGraph: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Unexpected error: {str(exc)}") from exc

    messages = result.get("messages", [])
    assistant_msg = _extract_assistant_message(messages)

    tokens_used = result.get("tokens_used", 0)
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    provider_used = result.get("provider_used", "unknown")
    model_used = result.get("model_used", "unknown")
    raw_profile_id = result.get("profile_id")
    profile_id = raw_profile_id or ACTIVE_PROFILE_ID
    if raw_profile_id and raw_profile_id != ACTIVE_PROFILE_ID:
        logger.warning(
            "Profile ID mismatch between adapter and LangGraph response",
            extra={
                "active_profile_id": ACTIVE_PROFILE_ID,
                "reported_profile_id": raw_profile_id,
                "fork_name": PROFILE_ID,
            },
        )
        track_profile_id_mismatch(
            fork_name=PROFILE_ID,
            active_profile_id=ACTIVE_PROFILE_ID,
            reported_profile_id=str(raw_profile_id),
        )
    tools_executed = list((result.get("tool_results") or {}).keys())

    # If knowledge base (RAG) returned results, track it as a tool invocation
    knowledge_context = result.get("knowledge_context") or []
    if knowledge_context:
        tools_executed.append("knowledge_base")

    sources = result.get("sources") or []
    attachments_used = [
        {
            "id": attachment["id"],
            "original_name": attachment["original_name"],
        }
        for attachment in attachments
    ]
    documents_used = [
        {
            "id": doc["id"],
            "filename": doc["filename"],
            "version": doc["version"],
        }
        for doc in documents
    ]
    memories_used = _serialize_loaded_memories(result.get("loaded_memory"))

    # Emit retrieval-quality feedback signals for each memory served in this response.
    for _mem in memories_used:
        try:
            track_memory_retrieval_signal(PROFILE_ID, _mem.get("user_rating"))
        except Exception:
            pass

    workspace_document_writeback = None

    if workspace_writeback_requested and len(documents) == 1:
        workspace_document_writeback = _write_response_back_to_workspace_document(
            request,
            effective_user_id=effective_user_id,
            document=documents[0],
            content_text=assistant_msg,
        )
        documents_used = [
            {
                "id": workspace_document_writeback["document_id"],
                "filename": workspace_document_writeback["filename"],
                "version": workspace_document_writeback["version"],
            }
        ]

    logger.info("Chat request completed successfully", extra={
        "tokens_used": tokens_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_used": model_used,
        "attachment_count_sent": len(attachments),
        "attachment_count_used": len(attachments_used),
        "attachment_delivery_success": len(attachments) > 0 and len(attachments_used) > 0,
        "document_count_sent": len(documents),
        "document_count_used": len(documents_used),
    })

    # Persist messages to conversation if conversation_id was provided
    conversation_id = request_body.conversation_id
    if conversation_id:
        try:
            conv_store = request.app.state.conversation_store
            # Save user message
            conv_store.add_message(
                conversation_id=conversation_id,
                role="user",
                content=request_body.message,
                metadata={
                    "attachment_ids": request_body.attachment_ids,
                    "document_ids": request_body.document_ids,
                },
            )
            # Save assistant message with metadata
            conv_store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_msg,
                tokens_used=tokens_used,
                tools_used=[{"name": t, "status": "success"} for t in tools_executed] if tools_executed else None,
                metadata={
                    "attachments_used": attachments_used,
                    "documents_used": documents_used,
                    "memories_used": memories_used,
                    "workspace_document_writeback": workspace_document_writeback,
                    "profile_id": profile_id,
                },
            )
        except Exception as exc:
            # Don't fail the chat request if persistence fails
            logger.warning(f"Failed to persist messages to conversation {conversation_id}: {exc}")

    # Log analytics event
    try:
        analytics_store = getattr(request.app.state, "analytics_store", None)
        if analytics_store:
            request_duration_seconds = time.time() - start_time
            analytics_store.log_event(
                user_id=effective_user_id,
                event_type="chat_request",
                model_name=model_used,
                tokens_used=tokens_used,
                request_duration_seconds=request_duration_seconds,
                status="success",
                fork_name=PROFILE_ID,
            )
    except Exception as exc:
        # Don't fail the request if analytics logging fails
        logger.warning(f"Failed to log analytics event: {exc}")

    return ChatResponse(
        response=assistant_msg,
        metadata={
            "tokens_used": tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider_used": provider_used,
            "model_used": model_used,
            "profile_id": profile_id,
            "tools_executed": tools_executed,
            "sources": sources,
            "attachments_used": attachments_used,
            "documents_used": documents_used,
            "memories_used": memories_used,
            "workspace_document_writeback": workspace_document_writeback,
            "model_warning": model_warning,
            "circuit_breakers": {
                "langgraph_invoke": LANGGRAPH_CIRCUIT_BREAKER.status(),
                "llm_model_validation": MODEL_VALIDATION_CIRCUIT_BREAKER.status(),
            },
        },
    )


def _extract_assistant_message(messages: list[dict[str, Any]]) -> str:
    """Extract the last assistant message content from a messages list."""
    def _sanitize_content(text: str) -> str:
        cleaned = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<\|[^|>]+\|>", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned or "Sorry, no readable response was generated."

    if not messages:
        return ""

    for message in reversed(messages):
        role = message.get("role")
        if role == "assistant":
            return _sanitize_content(str(message.get("content") or ""))

    last = messages[-1]
    return _sanitize_content(str(last.get("content") or ""))
