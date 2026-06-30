"""Dreaming Engine — per-user background memory cognition (plan-memory.md Phase 3).

One ``DreamingEngineTask`` (scope ``per_user``, 24h cooldown) runs the cheap
cognitive cycles for a single user at a time:

- **Cycle C — Forgetting (GC, no LLM, every tick):** delete episodic memories that
  are old, never-retrieved, and not epiphany contributors. Cheap and always-on.
- **Cycle B — Drift (capped auxiliary-LLM, daily):** for episodics created since
  the last drift run, find the nearest semantic memory and ask the auxiliary model
  whether the episodic contradicts it; if so lower the semantic's confidence, and
  if it falls below the floor open a ``memory_conflict`` for the user to resolve.

Design guarantees (concept §3 + Resilience):
- **One-by-one isolation:** an ``asyncio.Semaphore(dreaming_max_concurrency)``
  (default 1) wraps observe→reason→act so only one user dreams at a time even with
  multiple heartbeat workers — the engine never holds two users' data at once.
- **LLM-then-write separation:** ``reason()`` does ALL model work and writes
  nothing; ``act()`` does ALL writes and calls no model. A provider death in
  ``reason()`` therefore can't half-apply a mutation.
- **Degrade by cost:** a provider-reachability gate at the top of the LLM phase
  defers drift (status ``partial``, reason ``provider_offline``) while forgetting
  keeps running — and a partial run does NOT advance ``last_cycle_run``, so the
  cycle stays due and resumes next beat (cadence is the retry).
- **Per-user circuit breaker:** observe + each LLM call run through
  ``_breaker_for(user_id)`` so one user's flaky provider trips only their breaker.
- **Undo-safe ordering:** ``act`` writes the audit ``before_state`` BEFORE any
  destructive op, with a ``reversible_until`` = now + the undo window.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

import httpx

from backend.circuit_breaker import CircuitBreakerOpenError
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.orchestration.helpers.text_processing import extract_json_object

from ..task import HeartbeatTask, TickContext

logger = get_logger(__name__)


# Templated dissonance MCQ (concept §5B) — NOT model-generated.
CONFLICT_CHOICES: List[Dict[str, str]] = [
    {"id": "keep_old", "label": "Keep the existing memory"},
    {"id": "accept_new", "label": "Accept the new information"},
    {"id": "depends", "label": "It depends / both can be true"},
]

_DRIFT_PROMPT = (
    "You compare two statements about the same user and decide whether the NEW one "
    "CONTRADICTS the ESTABLISHED one.\n"
    "ESTABLISHED: {semantic}\n"
    "NEW: {episodic}\n"
    'Reply ONLY with a JSON object: {{"contradicts": true}} or {{"contradicts": false}}. '
    "Answer true only when the two statements cannot both be true."
)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EpisodicMemory:
    memory_id: str
    text: str
    created_at: Optional[datetime]
    access_count: int
    epiphany_contributor: bool
    raw: Dict[str, Any]  # full normalized record (audit before_state for undo)


@dataclass(frozen=True)
class SemanticMemory:
    memory_id: str
    text: str
    confidence: float


@dataclass(frozen=True)
class DriftCandidate:
    episodic: EpisodicMemory
    semantic: SemanticMemory


@dataclass
class DreamSnapshot:
    episodics: List[EpisodicMemory]
    drift_candidates: List[DriftCandidate]
    due_cycles: set[str]
    promotion_last: Optional[datetime]
    now: datetime


@dataclass
class ForgetAction:
    memory_id: str
    before_state: Dict[str, Any]


@dataclass
class DriftAction:
    semantic_id: str
    episodic_id: str
    prior_confidence: float
    new_confidence: float
    open_conflict: bool
    semantic_text: str
    episodic_text: str


@dataclass
class DreamPlan:
    forget: List[ForgetAction] = field(default_factory=list)
    drift: List[DriftAction] = field(default_factory=list)
    completed_cycles: set[str] = field(default_factory=set)
    deferred: List[str] = field(default_factory=list)
    tokens: int = 0
    status: str = "ok"  # "ok" | "partial"
    reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Read-side adapter protocol (a fake is injected in tests)
# --------------------------------------------------------------------------- #
class DreamMemoryReader(Protocol):
    def fetch_all(self, user_id: str) -> List[Dict[str, Any]]: ...
    def nearest_semantic(self, user_id: str, text: str) -> Optional[Dict[str, Any]]: ...
    def delete_memory(self, user_id: str, memory_id: str) -> None: ...
    def set_confidence(self, user_id: str, memory_id: str, confidence: float) -> None: ...


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #
class DreamingEngineTask(HeartbeatTask):
    def __init__(
        self,
        *,
        name: str,
        cooldown_seconds: int,
        scope: str,
        run_store: Any,
        audit_store: Any,
        conflict_store: Any,
        reader: DreamMemoryReader,
        adjudicator: Callable[[str, str], Awaitable[Dict[str, Any]]],
        health_gate: Callable[[], Awaitable[bool]],
        opt_out_checker: Callable[[str], bool],
        settings: Any,  # CognitiveMemorySettings
        semaphore: Optional[asyncio.Semaphore] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        super().__init__(
            name=name, cooldown_seconds=cooldown_seconds, run_store=run_store, scope=scope
        )
        self._audit = audit_store
        self._conflicts = conflict_store
        self._reader = reader
        self._adjudicator = adjudicator
        self._health_gate = health_gate
        self._opt_out = opt_out_checker
        self._s = settings
        max_conc = max(1, int(getattr(settings, "dreaming_max_concurrency", 1)))
        self._semaphore = semaphore or asyncio.Semaphore(max_conc)
        self._clock = clock
        self.active_user_recency_days = int(getattr(settings, "dreaming_active_user_days", 30))

    # --- Tick override: opt-out skip + one-by-one semaphore + rich status ----
    async def tick(self, ctx: Optional[TickContext] = None) -> Dict[str, Any]:
        if ctx is None:
            ctx = TickContext()
        user_id = ctx.user_id

        if self.cooldown_seconds > 0 and await self._within_cooldown(ctx):
            await self._record(ctx, "skipped", 0, {"reason": "cooldown"})
            return {"status": "skipped", "reason": "cooldown"}

        if user_id is not None and await self._is_opted_out(user_id):
            await self._record(ctx, "skipped", 0, {"reason": "opt_out"})
            return {"status": "skipped", "reason": "opt_out"}

        # One user dreams at a time (privacy isolation) even across workers.
        async with self._semaphore:
            start = time.monotonic()
            try:
                snapshot = await self._breaker_for(user_id).call(self.observe, ctx)
            except CircuitBreakerOpenError as exc:
                return await self._fail(ctx, start, "observe", exc, breaker_open=True)
            except Exception as exc:  # noqa: BLE001 — recorded, never propagated
                return await self._fail(ctx, start, "observe", exc)

            try:
                plan = await self.reason(ctx, snapshot)
                await self.act(ctx, plan)
            except Exception as exc:  # noqa: BLE001
                return await self._fail(ctx, start, "reason_act", exc)

            duration_ms = int((time.monotonic() - start) * 1000)
            detail: Dict[str, Any] = {
                "tokens": plan.tokens,
                "forgot": len(plan.forget),
                "drift": len(plan.drift),
            }
            if plan.deferred:
                detail["deferred"] = plan.deferred
            if plan.reason:
                detail["reason"] = plan.reason
            await self._record(ctx, plan.status, duration_ms, detail)
            logger.info(
                "dreaming_tick",
                user_id=user_id,
                status=plan.status,
                forgot=len(plan.forget),
                drift=len(plan.drift),
                tokens=plan.tokens,
            )
            return {"status": plan.status, "duration_ms": duration_ms}

    async def _is_opted_out(self, user_id: str) -> bool:
        try:
            return bool(await asyncio.to_thread(self._opt_out, user_id))
        except Exception:  # noqa: BLE001 — opt-out check best-effort → assume opted-in
            return False

    # --- observe (reads only) -----------------------------------------------
    async def observe(self, ctx: TickContext) -> DreamSnapshot:
        user_id = ctx.user_id
        now = self._clock()

        raw_items = await asyncio.to_thread(self._reader.fetch_all, user_id)
        episodics: List[EpisodicMemory] = []
        for item in raw_items:
            meta = item.get("metadata") or {}
            if str(meta.get("cognitive_tier", "episodic")) != "episodic":
                continue
            episodics.append(
                EpisodicMemory(
                    memory_id=str(item.get("memory_id") or meta.get("memory_id") or ""),
                    text=str(item.get("text") or ""),
                    created_at=_parse_dt(meta.get("created_at")),
                    access_count=int(meta.get("access_count", 0) or 0),
                    epiphany_contributor=bool(meta.get("epiphany_contributor", False)),
                    raw=item,
                )
            )

        drift_last = await asyncio.to_thread(
            self._audit.last_cycle_run, user_id=user_id, cycle="drift"
        )
        promotion_last = await asyncio.to_thread(
            self._audit.last_cycle_run, user_id=user_id, cycle="promotion"
        )

        due: set[str] = {"forgetting"}  # GC runs every tick
        if drift_last is None or (now - drift_last) >= timedelta(days=1):
            due.add("drift")

        drift_candidates: List[DriftCandidate] = []
        if "drift" in due:
            recent = [
                e for e in episodics
                if e.created_at is not None and (drift_last is None or e.created_at > drift_last)
            ]
            recent = recent[: max(0, int(getattr(self._s, "max_drift_checks_per_user", 10)))]
            for ep in recent:
                hit = await asyncio.to_thread(self._reader.nearest_semantic, user_id, ep.text)
                if not hit:
                    continue
                drift_candidates.append(
                    DriftCandidate(
                        episodic=ep,
                        semantic=SemanticMemory(
                            memory_id=str(hit.get("memory_id") or ""),
                            text=str(hit.get("text") or ""),
                            confidence=float(hit.get("confidence", 1.0)),
                        ),
                    )
                )

        return DreamSnapshot(
            episodics=episodics,
            drift_candidates=drift_candidates,
            due_cycles=due,
            promotion_last=promotion_last,
            now=now,
        )

    # --- reason (all LLM, no writes) ----------------------------------------
    async def reason(self, ctx: TickContext, snapshot: DreamSnapshot) -> DreamPlan:
        plan = DreamPlan()

        # Cycle C — Forgetting: cheap, no LLM, always completes when due.
        if "forgetting" in snapshot.due_cycles:
            plan.forget = self._plan_forgetting(snapshot)
            plan.completed_cycles.add("forgetting")

        # Cycle B — Drift: gated by provider reachability (degrade-by-cost).
        if "drift" in snapshot.due_cycles and snapshot.drift_candidates:
            reachable = await self._health_gate()
            if not reachable:
                plan.deferred.append("drift")
                plan.status = "partial"
                plan.reason = "provider_offline"
            else:
                drift_actions, tokens, broke = await self._plan_drift(ctx, snapshot)
                plan.drift = drift_actions
                plan.tokens += tokens
                if broke:
                    # Partial: keep what succeeded, defer the rest (don't advance cadence).
                    plan.status = "partial"
                    plan.reason = plan.reason or "drift_partial"
                    plan.deferred.append("drift")
                else:
                    plan.completed_cycles.add("drift")
        elif "drift" in snapshot.due_cycles:
            # Due but nothing to check → the cycle still ran to completion.
            plan.completed_cycles.add("drift")

        return plan

    def _plan_forgetting(self, snapshot: DreamSnapshot) -> List[ForgetAction]:
        cutoff = snapshot.now - timedelta(days=int(self._s.forget_age_days))
        cap = int(getattr(self._s, "max_forgets_per_run", 50))
        out: List[ForgetAction] = []
        for ep in snapshot.episodics:
            if ep.epiphany_contributor or ep.access_count != 0 or ep.created_at is None:
                continue
            if ep.created_at >= cutoff:
                continue
            # Guard: never GC something a long-deferred promotion might still claim.
            if snapshot.promotion_last is not None and ep.created_at >= snapshot.promotion_last:
                continue
            out.append(ForgetAction(memory_id=ep.memory_id, before_state=dict(ep.raw)))
            if len(out) >= cap:
                break
        return out

    async def _plan_drift(
        self, ctx: TickContext, snapshot: DreamSnapshot
    ) -> tuple[List[DriftAction], int, bool]:
        actions: List[DriftAction] = []
        tokens = 0
        broke = False
        breaker = self._breaker_for(ctx.user_id)
        floor = float(self._s.drift_confidence_floor)
        decrement = float(self._s.drift_decrement)
        for cand in snapshot.drift_candidates:
            try:
                verdict = await breaker.call(
                    self._adjudicator, cand.semantic.text, cand.episodic.text
                )
            except CircuitBreakerOpenError:
                broke = True
                break
            except Exception:  # noqa: BLE001 — provider failure → defer the rest
                broke = True
                break
            tokens += int((verdict or {}).get("tokens", 0) or 0)
            # Reject garbage: only act on an explicit True (None/ambiguous → no action).
            if not (verdict and verdict.get("contradicts") is True):
                continue
            prior = cand.semantic.confidence
            new_conf = max(0.0, prior - decrement)
            actions.append(
                DriftAction(
                    semantic_id=cand.semantic.memory_id,
                    episodic_id=cand.episodic.memory_id,
                    prior_confidence=prior,
                    new_confidence=new_conf,
                    open_conflict=new_conf < floor,
                    semantic_text=cand.semantic.text,
                    episodic_text=cand.episodic.text,
                )
            )
        return actions, tokens, broke

    # --- act (all writes, no LLM) -------------------------------------------
    async def act(self, ctx: TickContext, plan: DreamPlan) -> None:
        user_id = ctx.user_id
        now = self._clock()
        reversible_until = now + timedelta(days=int(self._s.undo_window_days))

        # Cycle C — Forgetting. Audit before_state BEFORE deleting (undo-safe).
        for action in plan.forget:
            await asyncio.to_thread(
                self._audit.record,
                user_id=user_id,
                cycle="forgetting",
                action="delete",
                target_kind="episodic",
                target_id=action.memory_id,
                before_state=action.before_state,
                reversible_until=reversible_until,
            )
            await asyncio.to_thread(self._reader.delete_memory, user_id, action.memory_id)

        # Cycle B — Drift. Lower confidence; open a conflict if below the floor.
        for d in plan.drift:
            await asyncio.to_thread(
                self._reader.set_confidence, user_id, d.semantic_id, d.new_confidence
            )
            await asyncio.to_thread(
                self._audit.record,
                user_id=user_id,
                cycle="drift",
                action="lower_confidence",
                target_kind="semantic",
                target_id=d.semantic_id,
                before_state={"confidence": d.prior_confidence},
                after_state={"confidence": d.new_confidence},
                reversible_until=reversible_until,
            )
            if d.open_conflict:
                await asyncio.to_thread(
                    self._conflicts.open_conflict,
                    user_id=user_id,
                    semantic_memory_id=d.semantic_id,
                    episodic_memory_id=d.episodic_id,
                    semantic_text=d.semantic_text,
                    contradiction_text=d.episodic_text,
                    prior_confidence=d.prior_confidence,
                    new_confidence=d.new_confidence,
                    choices=CONFLICT_CHOICES,
                )

        # Advance last_cycle_run ONLY for fully-completed cycles (a marker row).
        # A partial run skips its cycle here, so due-ness keeps it for next beat.
        for cycle in sorted(plan.completed_cycles):
            await asyncio.to_thread(
                self._audit.record,
                user_id=user_id,
                cycle=cycle,
                action="cycle_run",
                target_kind="cycle",
                target_id=cycle,
            )


# --------------------------------------------------------------------------- #
# Real-dependency adapters + factory (wired by the scheduler)
# --------------------------------------------------------------------------- #
class Mem0DreamReader:
    """DreamMemoryReader over the Mem0 backend (reads Qdrant via Mem0/direct search)."""

    def __init__(self, backend: Any) -> None:
        self._b = backend

    def fetch_all(self, user_id: str) -> List[Dict[str, Any]]:
        return self._b.get_all_for_dreaming(user_id, limit=2000)

    def nearest_semantic(self, user_id: str, text: str) -> Optional[Dict[str, Any]]:
        hits = self._b.find_nearest_semantic(user_id, text, top_k=1)
        return hits[0] if hits else None

    def delete_memory(self, user_id: str, memory_id: str) -> None:
        self._b.delete_memory(memory_id=memory_id, user_id=user_id)

    def set_confidence(self, user_id: str, memory_id: str, confidence: float) -> None:
        self._b.update_metadata(memory_id, {"confidence": confidence})


def build_auxiliary_drift_adjudicator(config: Any) -> Callable[[str, str], Awaitable[Dict[str, Any]]]:
    """Async drift adjudicator using the auxiliary provider via direct httpx
    (the proven path that keeps api_base in async; mirrors chat.py)."""

    async def _adjudicate(semantic_text: str, episodic_text: str) -> Dict[str, Any]:
        provider = config.llm.get_role_provider("auxiliary")
        api_base = str(getattr(provider, "api_base", "") or "").rstrip("/")
        if not api_base:
            raise RuntimeError("auxiliary api_base not configured")
        model_name = config.llm.get_role_model_name("auxiliary", "en")
        bare = model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
        prompt = _DRIFT_PROMPT.format(semantic=semantic_text[:1000], episodic=episodic_text[:1000])
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": bare,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
                headers={"Authorization": f"Bearer {getattr(provider, 'api_key', None) or 'no-key'}"},
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = int((data.get("usage") or {}).get("total_tokens", 0) or 0)
        parsed = extract_json_object(content)
        contradicts: Optional[bool] = None
        if parsed is not None and "contradicts" in parsed:
            v = parsed["contradicts"]
            contradicts = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
        return {"contradicts": contradicts, "tokens": tokens}

    return _adjudicate


def build_provider_reachability_gate(config: Any) -> Callable[[], Awaitable[bool]]:
    """One-shot auxiliary-provider reachability probe (degrade-by-cost gate)."""

    async def _reachable() -> bool:
        try:
            provider = config.llm.get_role_provider("auxiliary")
            api_base = str(getattr(provider, "api_base", "") or "").rstrip("/")
            if not api_base:
                return False
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"{api_base}/models",
                    headers={"Authorization": f"Bearer {getattr(provider, 'api_key', None) or 'no-key'}"},
                )
            return resp.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    return _reachable


def make_opt_out_checker(settings_store: Any) -> Callable[[str], bool]:
    """User opted out when their persisted analytics_preferences.memory_enabled is False."""

    def _opted_out(user_id: str) -> bool:
        try:
            settings = settings_store.get_user_settings(user_id)
        except Exception:  # noqa: BLE001
            return False
        if not settings:
            return False
        prefs = settings.get("analytics_preferences") or {}
        return prefs.get("memory_enabled") is False

    return _opted_out


def build_dreaming_task(
    *,
    name: str,
    cooldown_seconds: int,
    scope: str,
    run_store: Any,
    audit_store: Any,
    conflict_store: Any,
) -> DreamingEngineTask:
    """Construct a fully-wired DreamingEngineTask from runtime config.

    Builds the Mem0-backed reader, the auxiliary-LLM adjudicator + reachability
    gate, and the per-user opt-out checker. Raises on hard config/backend errors
    so the scheduler can skip the task and keep the heartbeat beating.
    """
    from universal_agentic_framework.config import load_core_config
    from universal_agentic_framework.memory.factory import build_memory_backend
    from backend.db import SettingsStore, init_db_pool

    config = load_core_config()
    cognitive = config.memory.cognitive
    backend = build_memory_backend(config)
    settings_store = SettingsStore(init_db_pool())

    return DreamingEngineTask(
        name=name,
        cooldown_seconds=cooldown_seconds,
        scope=scope,
        run_store=run_store,
        audit_store=audit_store,
        conflict_store=conflict_store,
        reader=Mem0DreamReader(backend),
        adjudicator=build_auxiliary_drift_adjudicator(config),
        health_gate=build_provider_reachability_gate(config),
        opt_out_checker=make_opt_out_checker(settings_store),
        settings=cognitive,
    )
