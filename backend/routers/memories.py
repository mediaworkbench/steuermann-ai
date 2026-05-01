from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.single_user import get_effective_user_id, require_api_access
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.memory import MemoryRatingBackend
from universal_agentic_framework.memory.factory import build_memory_backend

router = APIRouter(
    prefix="/api/memories",
    tags=["memories"],
    dependencies=[Depends(require_api_access)],
)


class MemoryRatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5 stars")


@router.post("/{memory_id}/rate")
async def rate_memory(
    memory_id: str,
    body: MemoryRatingRequest,
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

    return {"status": "ok", "memory_id": memory_id, "rating": body.rating}
