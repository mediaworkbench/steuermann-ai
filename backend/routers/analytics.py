"""Analytics router for time-series data and trends."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.db import AnalyticsStore
from backend.single_user import get_effective_user_id, require_api_access


router = APIRouter(prefix="/api", tags=["analytics"], dependencies=[Depends(require_api_access)])


def _get_analytics_store(request: Request) -> AnalyticsStore:
    """Get the analytics store from app state."""
    store = getattr(request.app.state, "analytics_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store unavailable")
    return store


@router.get("/analytics/usage-trends")
async def get_usage_trends(
    days: int = Query(default=30, ge=1, le=365),
    request: Request = None,
) -> Dict[str, Any]:
    """Get usage trends for the past N days."""
    store = _get_analytics_store(request)
    trends = store.get_usage_trends(days=days)
    return {
        "period_days": days,
        "trends": trends,
        "total_requests": sum(t["requests"] for t in trends),
        "unique_users": sum(t["users"] for t in trends),
    }


@router.get("/analytics/token-consumption")
async def get_token_consumption(
    days: int = Query(default=30, ge=1, le=365),
    request: Request = None,
) -> Dict[str, Any]:
    """Get token consumption trends for the past N days."""
    store = _get_analytics_store(request)
    consumption = store.get_token_consumption(days=days)
    
    total_tokens = sum(c["total_tokens"] for c in consumption)
    avg_tokens_per_request = (
        total_tokens / sum(c["requests"] for c in consumption)
        if sum(c["requests"] for c in consumption) > 0
        else 0
    )
    
    return {
        "period_days": days,
        "consumption": consumption,
        "total_tokens": total_tokens,
        "avg_tokens_per_request": avg_tokens_per_request,
    }


@router.get("/analytics/latency-analysis")
async def get_latency_analysis(
    days: int = Query(default=30, ge=1, le=365),
    request: Request = None,
) -> Dict[str, Any]:
    """Get latency analysis for the past N days."""
    store = _get_analytics_store(request)
    latency_data = store.get_latency_analysis(days=days)
    
    # Calculate overall stats
    avg_latencies = [d["avg_latency_ms"] for d in latency_data if d["avg_latency_ms"] > 0]
    overall_avg = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0
    overall_min = min((d["min_latency_ms"] for d in latency_data), default=0)
    overall_max = max((d["max_latency_ms"] for d in latency_data), default=0)
    
    return {
        "period_days": days,
        "latency_data": latency_data,
        "overall_avg_ms": round(overall_avg, 2),
        "overall_min_ms": round(overall_min, 2),
        "overall_max_ms": round(overall_max, 2),
        "total_requests": sum(d["requests"] for d in latency_data),
    }


@router.get("/analytics/cost-projection")
async def get_cost_projection(
    days: int = Query(default=30, ge=1, le=365),
    cost_per_token: float = Query(default=0.00002, ge=0),
    request: Request = None,
) -> Dict[str, Any]:
    """Get cost projection based on historical token usage."""
    store = _get_analytics_store(request)
    projection = store.get_cost_projection(cost_per_token=cost_per_token, days=days)
    return projection


@router.post("/analytics/log-event")
async def log_event(
    user_id: str | None = Query(default=None),
    event_type: str = Query(...),
    model_name: str | None = Query(default=None),
    tokens_used: int | None = Query(default=None),
    request_duration_seconds: float | None = Query(default=None),
    status: str = Query(default="success"),
    fork_name: str | None = Query(default=None),
    request: Request = None,
) -> Dict[str, str]:
    """Log an analytics event."""
    store = _get_analytics_store(request)
    effective_user_id = get_effective_user_id(user_id)
    success = store.log_event(
        user_id=effective_user_id,
        event_type=event_type,
        model_name=model_name,
        tokens_used=tokens_used,
        request_duration_seconds=request_duration_seconds,
        status=status,
        fork_name=fork_name,
    )
    
    if success:
        return {"status": "logged"}
    else:
        raise HTTPException(status_code=500, detail="Failed to log event")


@router.get("/analytics/message-quality")
async def get_message_quality(
    days: int = Query(default=30, ge=1, le=365),
    request: Request = None,
) -> Dict[str, Any]:
    """Get daily message quality (thumbs up/down) trends for the past N days.

    Each entry contains the date, up_count, down_count, total_feedback,
    total_assistant_messages, and net_score (up - down) for that day.
    """
    store = _get_analytics_store(request)
    quality_data = store.get_message_quality(days=days)

    total_up = sum(d["up_count"] for d in quality_data)
    total_down = sum(d["down_count"] for d in quality_data)
    total_feedback = sum(d["total_feedback"] for d in quality_data)
    total_messages = sum(d["total_assistant_messages"] for d in quality_data)
    feedback_rate = round(total_feedback / total_messages, 4) if total_messages > 0 else 0.0

    return {
        "period_days": days,
        "quality_data": quality_data,
        "total_up": total_up,
        "total_down": total_down,
        "total_feedback": total_feedback,
        "total_assistant_messages": total_messages,
        "net_score": total_up - total_down,
        "feedback_rate": feedback_rate,
    }
