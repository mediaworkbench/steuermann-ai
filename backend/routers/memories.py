from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.single_user import get_effective_user_id, require_api_access
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.memory import MemoryDeleteBackend, MemoryRatingBackend
from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.monitoring.metrics import track_memory_rated_after_retrieval

router = APIRouter(
    prefix="/api/memories",
    tags=["memories"],
    dependencies=[Depends(require_api_access)],
)

PROFILE_ID = os.getenv("PROFILE_ID", "starter")
_RETRIEVAL_LOOKBACK_MESSAGES = 200  # messages to scan for recent retrievals


class MemoryRatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5 stars")


def _memory_was_recently_retrieved(request: Request, memory_id: str, user_id: str) -> bool:
    """Return True if *memory_id* appears in any recent conversation message's memories_used."""
    try:
        conversation_store = getattr(request.app.state, "conversation_store", None)
        if conversation_store is None or not hasattr(conversation_store, "search_messages"):
            return False

        # Scan recent messages across user conversations via search_messages with broad query.
        # We use a lightweight direct DB approach: list recent conversations then scan messages.
        if not hasattr(conversation_store, "list_conversations"):
            return False

        conversations = conversation_store.list_conversations(user_id=user_id, limit=20, offset=0)
        for conv in conversations or []:
            conv_id = conv.get("id")
            if not conv_id:
                continue
            try:
                messages = conversation_store.get_messages(
                    conv_id,
                    limit=_RETRIEVAL_LOOKBACK_MESSAGES,
                    offset=0,
                )
            except Exception:
                continue
            for msg in reversed(messages or []):
                metadata = msg.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                for mem_ref in metadata.get("memories_used") or []:
                    if isinstance(mem_ref, dict) and mem_ref.get("memory_id") == memory_id:
                        return True
    except Exception:
        pass
    return False


def _extract_owner_user_id(point: dict[str, Any]) -> Optional[str]:
    payload = point.get("payload") or {}
    return payload.get("user_id")


def _serialize_memory_point(point: dict[str, Any]) -> dict[str, Any]:
    payload = point.get("payload") or {}
    metadata = payload.get("metadata") or {}
    memory_id = str(metadata.get("memory_id") or point.get("point_id") or "")
    return {
        "memory_id": memory_id,
        "text": str(payload.get("text") or ""),
        "user_rating": metadata.get("user_rating"),
        "importance_score": metadata.get("importance_score"),
        "is_related": bool(metadata.get("is_related", False)),
        "metadata": metadata,
    }


@router.get("")
async def list_memories(
    limit: int = Query(default=50, ge=1, le=500),
    query: Optional[str] = Query(default=None),
    include_related: bool = Query(default=False),
    user_id: str = Depends(get_effective_user_id),
):
    cfg = load_core_config()
    backend = build_memory_backend(cfg)

    records = backend.load(
        user_id=user_id,
        query=query,
        top_k=limit,
        include_related=include_related,
    )

    memories = []
    for record in records:
        metadata = dict(record.metadata or {})
        memory_id = metadata.get("memory_id")
        if not memory_id:
            continue
        memories.append(
            {
                "memory_id": str(memory_id),
                "text": record.text,
                "user_rating": metadata.get("user_rating"),
                "importance_score": metadata.get("importance_score"),
                "is_related": bool(metadata.get("is_related", False)),
                "created_at": metadata.get("created_at"),
                "metadata": metadata,
            }
        )

    return {
        "items": memories,
        "count": len(memories),
        "limit": limit,
        "query": query,
        "include_related": include_related,
    }


@router.get("/stats")
async def memory_stats(
    sample_limit: int = Query(default=500, ge=1, le=5000),
    user_id: str = Depends(get_effective_user_id),
):
    cfg = load_core_config()
    backend = build_memory_backend(cfg)

    records = backend.load(user_id=user_id, query=None, top_k=sample_limit, include_related=False)
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=7)

    total = len(records)
    rated = 0
    related = 0
    importance_values: list[float] = []
    recent_7d = 0

    for record in records:
        metadata = record.metadata or {}
        if metadata.get("user_rating") is not None:
            rated += 1
        if metadata.get("is_related"):
            related += 1
        score = metadata.get("importance_score")
        if isinstance(score, (int, float)):
            importance_values.append(float(score))
        created_at = metadata.get("created_at")
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_dt >= recent_cutoff:
                    recent_7d += 1
            except ValueError:
                pass

    return {
        "sample_limit": sample_limit,
        "sample_count": total,
        "totals": {
            "memories": total,
            "rated": rated,
            "unrated": max(total - rated, 0),
            "related": related,
            "recent_7d": recent_7d,
        },
        "ratios": {
            "rated_coverage": (rated / total) if total else 0.0,
            "related_ratio": (related / total) if total else 0.0,
        },
        "quality": {
            "average_importance": (sum(importance_values) / len(importance_values))
            if importance_values
            else 0.0,
        },
    }


@router.get("/{memory_id}")
async def get_memory(
    memory_id: str,
    user_id: str = Depends(get_effective_user_id),
):
    cfg = load_core_config()
    backend = build_memory_backend(cfg)

    if not isinstance(backend, MemoryRatingBackend):
        raise HTTPException(status_code=503, detail="Memory backend does not support memory lookup")

    point = backend.find_memory_point(memory_id)
    if not point:
        raise HTTPException(status_code=404, detail="Memory not found")

    owner_user_id = _extract_owner_user_id(point)
    if owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="Memory does not belong to user")

    return _serialize_memory_point(point)


@router.post("/{memory_id}/rate")
async def rate_memory(
    memory_id: str,
    body: MemoryRatingRequest,
    request: Request,
    user_id: str = Depends(get_effective_user_id),
):
    """Rate a memory (1-5 stars) for importance scoring."""
    cfg = load_core_config()
    backend = build_memory_backend(cfg)

    if not isinstance(backend, MemoryRatingBackend):
        raise HTTPException(status_code=503, detail="Memory backend does not support rating")

    point = backend.find_memory_point(memory_id)
    if not point:
        raise HTTPException(status_code=404, detail="Memory not found")

    payload = point.get("payload") or {}
    owner_user_id = payload.get("user_id")
    if owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="Memory does not belong to user")

    metadata = payload.get("metadata") or {}
    backend.set_memory_user_rating(
        point_id=point.get("point_id"),
        metadata=metadata,
        rating=body.rating,
    )

    # Feedback-loop signal: was this memory recently retrieved in a conversation?
    try:
        if _memory_was_recently_retrieved(request, memory_id, user_id):
            track_memory_rated_after_retrieval(PROFILE_ID)
    except Exception:
        pass

    return {"status": "ok", "memory_id": memory_id, "rating": body.rating}


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = Depends(get_effective_user_id),
):
    cfg = load_core_config()
    backend = build_memory_backend(cfg)

    if not isinstance(backend, MemoryRatingBackend):
        raise HTTPException(status_code=503, detail="Memory backend does not support memory lookup")
    if not isinstance(backend, MemoryDeleteBackend):
        raise HTTPException(status_code=503, detail="Memory backend does not support delete")

    point = backend.find_memory_point(memory_id)
    if not point:
        raise HTTPException(status_code=404, detail="Memory not found")

    owner_user_id = _extract_owner_user_id(point)
    if owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="Memory does not belong to user")

    try:
        backend.delete_memory(memory_id=memory_id, user_id=user_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Memory does not belong to user")

    return {"status": "ok", "memory_id": memory_id, "deleted": True}
