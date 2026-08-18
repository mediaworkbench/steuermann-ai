"""Human-in-the-loop API for the cognitive memory Dreaming Engine (Phase 6).

Two audiences, strictly separated by privacy posture:

- **User-scoped** (`current_user_id`): the signed-in user resolves their own
  dissonance conflicts, approves/rejects proposed procedural rules, and undoes
  recent engine actions within the 7-day window. ``user_id`` always comes from the
  authenticated identity — **never** from the request body — and every store call
  filters on it, so user A can never see or mutate user B's data.
- **Admin-aggregated** (`require_admin`): `GET /api/admin/dreaming-metrics` calls
  only the stores' ``aggregate_metrics`` / ``aggregate_task_stats`` (COUNT/GROUP BY,
  no text/user_id/payload columns) → structurally PII-safe (concept §5A).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from backend.auth import current_user_id, require_admin
from backend.single_user import require_api_access
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.orchestration.procedural_node import invalidate_procedural_cache

router = APIRouter(
    prefix="/api",
    tags=["dreaming"],
    dependencies=[Depends(require_api_access)],
)

DREAMING_TASK_NAME = "dreaming"

# Lazily-built memory backend for the Qdrant mutations of resolve/undo. Built once
# per process (connecting per request would be wasteful for these rare actions).
_backend: Any = None


def _get_backend() -> Any:
    global _backend
    if _backend is None:
        _backend = build_memory_backend(load_core_config())
    return _backend


def _store(request: Request, name: str) -> Any:
    store = getattr(request.app.state, name, None)
    if store is None:
        raise HTTPException(status_code=503, detail=f"{name} unavailable")
    return store


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ResolveConflictRequest(BaseModel):
    choice: Literal["keep_old", "accept_new", "depends"]


# --------------------------------------------------------------------------- #
# Dissonance conflicts
# --------------------------------------------------------------------------- #
@router.get("/memory/dreaming/conflicts")
def list_conflicts(request: Request, user_id: str = Depends(current_user_id)) -> Dict[str, Any]:
    rows = _store(request, "memory_conflict_store").list_open(user_id)
    conflicts = [
        {
            "conflict_id": r["conflict_id"],
            "semantic_text": r.get("semantic_text", ""),
            "contradiction_text": r.get("contradiction_text", ""),
            "prior_confidence": r.get("prior_confidence"),
            "new_confidence": r.get("new_confidence"),
            "choices": r.get("choices") or [],
            "created_at": r.get("created_at"),
        }
        for r in rows
    ]
    return {"conflicts": conflicts}


@router.post("/memory/dreaming/conflicts/{conflict_id}/resolve")
def resolve_conflict(
    conflict_id: str,
    body: ResolveConflictRequest,
    request: Request,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    conflict_store = _store(request, "memory_conflict_store")
    audit_store = _store(request, "memory_audit_store")

    # resolve() is user-scoped (WHERE conflict_id AND user_id) and returns the row,
    # so a non-owner / unknown id yields None → 404 (no cross-user resolution).
    resolved = conflict_store.resolve(
        conflict_id=conflict_id,
        user_id=user_id,
        resolution={"choice": body.choice},
        status="resolved",
    )
    if resolved is None:
        raise HTTPException(status_code=404, detail="Conflict not found")

    semantic_id = resolved.get("semantic_memory_id")
    prior_conf = resolved.get("prior_confidence")
    backend = _get_backend()
    try:
        if body.choice == "keep_old" and semantic_id and prior_conf is not None:
            # The established memory wins → restore its pre-drift confidence.
            backend.update_metadata(str(semantic_id), {"confidence": float(prior_conf)})
        elif body.choice == "accept_new" and semantic_id:
            # The new info wins → the semantic was wrong; drop it.
            backend.delete_memory(memory_id=str(semantic_id), user_id=user_id)
        # "depends" → leave the semantic as-is; the resolution row records the review.
    except Exception:  # noqa: BLE001 — the conflict is resolved regardless of Qdrant
        pass

    audit_store.record(
        user_id=user_id,
        cycle="drift",
        action="resolve_conflict",
        target_kind="semantic",
        target_id=str(semantic_id) if semantic_id else None,
        after_state={"choice": body.choice},
    )
    return {"status": "resolved", "choice": body.choice}


# --------------------------------------------------------------------------- #
# Procedural approvals
# --------------------------------------------------------------------------- #
@router.get("/memory/dreaming/procedural")
def list_procedural(request: Request, user_id: str = Depends(current_user_id)) -> Dict[str, Any]:
    rows = _store(request, "procedural_store").list_for_user(user_id)
    rules = [
        {
            "rule_key": r["rule_key"],
            "rule_text": r.get("rule_text", ""),
            "tier": r.get("tier"),
            "status": r.get("status"),
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence") or {},
            "updated_at": r.get("updated_at"),
        }
        for r in rows
    ]
    return {"rules": rules}


def _set_procedural_status(
    request: Request, user_id: str, rule_key: str, new_status: str
) -> Dict[str, Any]:
    store = _store(request, "procedural_store")
    existing = store.get(user_id, rule_key)
    if existing is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    # Only a surfaced (proposed) rule can be approved/rejected — never an immature
    # observing candidate or an already-decided rule.
    if existing.get("status") != "proposed":
        raise HTTPException(status_code=409, detail="Rule is not awaiting approval")
    updated = store.set_status(user_id=user_id, rule_key=rule_key, status=new_status)
    invalidate_procedural_cache(user_id)  # active set changed → drop the prompt cache
    return {"status": (updated or {}).get("status", new_status), "rule_key": rule_key}


@router.post("/memory/dreaming/procedural/{rule_key}/approve")
def approve_procedural(
    rule_key: str, request: Request, user_id: str = Depends(current_user_id)
) -> Dict[str, Any]:
    return _set_procedural_status(request, user_id, rule_key, "active")


@router.post("/memory/dreaming/procedural/{rule_key}/reject")
def reject_procedural(
    rule_key: str, request: Request, user_id: str = Depends(current_user_id)
) -> Dict[str, Any]:
    return _set_procedural_status(request, user_id, rule_key, "rejected")


# --------------------------------------------------------------------------- #
# Undo feed
# --------------------------------------------------------------------------- #
@router.get("/memory/dreaming/audit")
def list_audit(request: Request, user_id: str = Depends(current_user_id)) -> Dict[str, Any]:
    rows = _store(request, "memory_audit_store").list_reversible(user_id=user_id)
    items = [
        {
            "audit_id": r["audit_id"],
            "cycle": r.get("cycle"),
            "action": r.get("action"),
            "target_kind": r.get("target_kind"),
            "target_id": r.get("target_id"),
            "before_state": r.get("before_state"),  # the user's own data — safe to show
            "after_state": r.get("after_state"),
            "created_at": r.get("created_at"),
            "reversible_until": r.get("reversible_until"),
        }
        for r in rows
    ]
    return {"reversible": items}


@router.post("/memory/dreaming/audit/{audit_id}/undo")
def undo_action(
    audit_id: str, request: Request, user_id: str = Depends(current_user_id)
) -> Dict[str, Any]:
    audit_store = _store(request, "memory_audit_store")
    row = audit_store.get(audit_id=audit_id, user_id=user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Audit entry not found")

    reversible_until = row.get("reversible_until")
    if reversible_until is None:
        raise HTTPException(status_code=409, detail="Action is not reversible")
    if isinstance(reversible_until, datetime):
        ru = reversible_until if reversible_until.tzinfo else reversible_until.replace(tzinfo=timezone.utc)
        if ru <= datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Undo window has expired")

    backend = _get_backend()
    before = row.get("before_state") or {}
    after = row.get("after_state") or {}
    action = row.get("action")
    target_id = row.get("target_id")

    try:
        if action == "delete":
            # Re-insert the forgotten memory verbatim from its snapshot.
            text = str(before.get("text") or "")
            if text:
                backend.restore_memory(user_id, text=text, metadata=before.get("metadata") or {})
        elif action == "lower_confidence" and target_id:
            backend.update_metadata(str(target_id), {"confidence": float(before.get("confidence", 1.0))})
        elif action == "promote" and target_id:
            backend.delete_memory(memory_id=str(target_id), user_id=user_id)
            for src in after.get("source_episodic_ids") or []:
                backend.update_metadata(str(src), {"epiphany_contributor": False})
        elif action == "propose" and target_id:
            # Revert a matured procedural rule.
            procedural = getattr(request.app.state, "procedural_store", None)
            if procedural is not None:
                procedural.set_status(user_id=user_id, rule_key=str(target_id), status="reverted")
                invalidate_procedural_cache(user_id)
    except Exception:  # noqa: BLE001 — best-effort reversal; still mark undone below
        pass

    audit_store.mark_undone(audit_id=audit_id, user_id=user_id)
    audit_store.record(
        user_id=user_id,
        cycle=str(row.get("cycle") or "drift"),
        action="undo",
        target_kind=row.get("target_kind"),
        target_id=target_id,
        before_state={"undid_audit_id": audit_id, "undid_action": action},
    )
    return {"status": "undone", "audit_id": audit_id}


# --------------------------------------------------------------------------- #
# Admin-aggregated metrics (PII-safe)
# --------------------------------------------------------------------------- #
@router.get("/admin/dreaming-metrics")
def admin_dreaming_metrics(request: Request, _admin=Depends(require_admin)) -> Dict[str, Any]:
    conflict_store = _store(request, "memory_conflict_store")
    procedural_store = _store(request, "procedural_store")
    audit_store = _store(request, "memory_audit_store")
    run_store = _store(request, "heartbeat_run_store")

    run_stats = run_store.aggregate_task_stats(DREAMING_TASK_NAME)
    conflict_metrics = conflict_store.aggregate_metrics()
    procedural_metrics = procedural_store.aggregate_metrics()
    audit_metrics = audit_store.aggregate_metrics()

    cycles_run = int(run_stats.get("count", 0))
    total_tokens = int(run_stats.get("total_tokens", 0))
    open_conflicts = int(conflict_metrics.get("open", 0))
    proposed_procedural = int(procedural_metrics.get("proposed_pending", 0))
    by_action = audit_metrics.get("by_action", {})

    try:
        vector_count = _get_backend().count_points()
    except Exception:  # noqa: BLE001
        vector_count = 0

    return {
        "cycles_run": cycles_run,
        "vector_count": vector_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_cycle": round(total_tokens / cycles_run, 1) if cycles_run else 0,
        "pending_resolutions": {
            "open_conflicts": open_conflicts,
            "proposed_procedural": proposed_procedural,
            "total": open_conflicts + proposed_procedural,
        },
        "deletion_count": int(by_action.get("delete", 0)),
        "promotion_count": int(by_action.get("promote", 0)),
        "run_status": run_stats.get("by_status", {}),
    }
