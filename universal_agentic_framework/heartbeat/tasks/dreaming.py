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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

import httpx
import numpy as np

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

_EPIPHANY_PROMPT = (
    "Below are several related things observed about the same user over time. "
    "Synthesize ONE concise, general statement (an enduring fact or preference) that "
    "captures what they have in common. Write a single sentence in the third person, "
    "no preamble.\n\nOBSERVATIONS:\n{observations}"
)


def _strip_reasoning(text: str) -> str:
    """Drop a ``<think>…</think>`` reasoning block some local models inline into the
    content (and an unterminated/truncated one), so only the final answer remains.
    Mirrors the conversation-title sanitizer — reasoning models need a generous
    token budget AND defensive stripping or the answer is lost."""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"(?is)<think>.*$", "", cleaned)
    return cleaned.strip()


def _classify_tier(rule_key: str) -> Optional[int]:
    """Tier from the rule_key namespace (concept §4): format.*→1, style.*→2,
    logic.*/safety.*→3 (locked). Unknown namespaces → None (ignored)."""
    key = (rule_key or "").strip().lower()
    if key.startswith("format."):
        return 1
    if key.startswith("style."):
        return 2
    if key.startswith("logic.") or key.startswith("safety."):
        return 3
    return None


def greedy_cosine_clusters(
    items: List[Dict[str, Any]], *, threshold: float, min_size: int
) -> List[List[Dict[str, Any]]]:
    """Greedy single-pass cosine clustering over item ``vector``s.

    Each unclustered item seeds a cluster and absorbs every other unclustered item
    within ``threshold`` cosine similarity. Only clusters of ≥ ``min_size`` are
    returned. Deterministic (input order) and side-effect free.
    """
    rows: List[tuple[Dict[str, Any], Any]] = []
    for it in items:
        vec = it.get("vector")
        if not vec:
            continue
        arr = np.asarray(vec, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            continue
        rows.append((it, arr / norm))

    used = [False] * len(rows)
    clusters: List[List[Dict[str, Any]]] = []
    for i in range(len(rows)):
        if used[i]:
            continue
        it_i, v_i = rows[i]
        members = [it_i]
        used[i] = True
        for j in range(i + 1, len(rows)):
            if used[j]:
                continue
            if float(v_i @ rows[j][1]) >= threshold:
                members.append(rows[j][0])
                used[j] = True
        if len(members) >= min_size:
            clusters.append(members)
    return clusters


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
    # Promotion (Cycle A) inputs — populated only when promotion is due.
    episodic_vectors: List[Dict[str, Any]] = field(default_factory=list)
    existing_sources: List[List[str]] = field(default_factory=list)
    # Procedural (Cycle D) inputs — recent behavioural texts, when procedural is due.
    procedural_observations: List[str] = field(default_factory=list)


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
class PromotionAction:
    text: str
    confidence: float
    source_episodic_ids: List[str]


@dataclass
class ProceduralCandidate:
    rule_key: str
    rule_text: str
    tier: int          # 1 (format.*) or 2 (style.*)
    samples: int       # behavioural instances supporting it this window
    example: str


@dataclass
class Tier3Suggestion:
    rule_key: str
    rule_text: str


@dataclass
class DreamPlan:
    forget: List[ForgetAction] = field(default_factory=list)
    drift: List[DriftAction] = field(default_factory=list)
    promote: List[PromotionAction] = field(default_factory=list)
    procedural: List[ProceduralCandidate] = field(default_factory=list)
    tier3_suggestions: List[Tier3Suggestion] = field(default_factory=list)
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
    # Promotion (Cycle A)
    def fetch_episodic_vectors(self, user_id: str) -> List[Dict[str, Any]]: ...
    def existing_semantic_sources(self, user_id: str) -> List[List[str]]: ...
    def write_semantic(
        self, user_id: str, text: str, confidence: float, source_episodic_ids: List[str]
    ) -> str: ...
    def flag_contributor(self, user_id: str, memory_id: str) -> None: ...


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
        synthesizer: Optional[Callable[[List[str]], Awaitable[Dict[str, Any]]]] = None,
        procedural_store: Any = None,
        proposer: Optional[Callable[[List[str]], Awaitable[Dict[str, Any]]]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        super().__init__(
            name=name, cooldown_seconds=cooldown_seconds, run_store=run_store, scope=scope
        )
        self._audit = audit_store
        self._conflicts = conflict_store
        self._procedural = procedural_store
        self._reader = reader
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer
        self._proposer = proposer
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
                "promoted": len(plan.promote),
                "procedural": len(plan.procedural),
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
                promoted=len(plan.promote),
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
        # Promotion needs the LLM synthesizer; without it the cycle can't run.
        promotion_interval = int(getattr(self._s, "promotion_interval_days", 7))
        if self._synthesizer is not None and (
            promotion_last is None or (now - promotion_last) >= timedelta(days=promotion_interval)
        ):
            due.add("promotion")

        # Procedural (Cycle D) runs daily over 24h windows; needs the proposer + store.
        procedural_last = await asyncio.to_thread(
            self._audit.last_cycle_run, user_id=user_id, cycle="procedural"
        )
        if (
            self._proposer is not None
            and self._procedural is not None
            and (procedural_last is None or (now - procedural_last) >= timedelta(days=1))
        ):
            due.add("procedural")

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

        episodic_vectors: List[Dict[str, Any]] = []
        existing_sources: List[List[str]] = []
        if "promotion" in due:
            episodic_vectors = await asyncio.to_thread(
                self._reader.fetch_episodic_vectors, user_id
            )
            existing_sources = await asyncio.to_thread(
                self._reader.existing_semantic_sources, user_id
            )

        procedural_observations: List[str] = []
        if "procedural" in due:
            window = int(getattr(self._s, "procedural_tier2_window_days", 5))
            recent_cutoff = now - timedelta(days=window)
            procedural_observations = [
                e.text
                for e in episodics
                if e.text and e.created_at is not None and e.created_at >= recent_cutoff
            ][:40]

        return DreamSnapshot(
            episodics=episodics,
            drift_candidates=drift_candidates,
            due_cycles=due,
            promotion_last=promotion_last,
            now=now,
            episodic_vectors=episodic_vectors,
            existing_sources=existing_sources,
            procedural_observations=procedural_observations,
        )

    # --- reason (all LLM, no writes) ----------------------------------------
    async def reason(self, ctx: TickContext, snapshot: DreamSnapshot) -> DreamPlan:
        plan = DreamPlan()

        # Cycle C — Forgetting: cheap, no LLM, always completes when due.
        if "forgetting" in snapshot.due_cycles:
            plan.forget = self._plan_forgetting(snapshot)
            plan.completed_cycles.add("forgetting")

        # The two LLM cycles share one reachability probe (don't hammer a dead
        # provider with N calls). Resolved lazily, at most once per tick.
        self._reachable_cache: Optional[bool] = None

        # Cycle B — Drift.
        if "drift" in snapshot.due_cycles and snapshot.drift_candidates:
            if not await self._reachable(plan):
                self._defer(plan, "drift")
            else:
                drift_actions, tokens, broke = await self._plan_drift(ctx, snapshot)
                plan.drift = drift_actions
                plan.tokens += tokens
                if broke:
                    self._defer(plan, "drift", reason="drift_partial")
                else:
                    plan.completed_cycles.add("drift")
        elif "drift" in snapshot.due_cycles:
            plan.completed_cycles.add("drift")  # due but nothing to check

        # Cycle A — Promotion / Epiphany.
        if "promotion" in snapshot.due_cycles and snapshot.episodic_vectors:
            if not await self._reachable(plan):
                self._defer(plan, "promotion")
            else:
                promotions, tokens, broke = await self._plan_promotions(ctx, snapshot)
                plan.promote = promotions
                plan.tokens += tokens
                if broke:
                    self._defer(plan, "promotion", reason="promotion_partial")
                else:
                    plan.completed_cycles.add("promotion")
        elif "promotion" in snapshot.due_cycles:
            plan.completed_cycles.add("promotion")  # due but nothing to cluster

        # Cycle D — Procedural learning.
        if "procedural" in snapshot.due_cycles and snapshot.procedural_observations:
            if not await self._reachable(plan):
                self._defer(plan, "procedural")
            else:
                candidates, suggestions, tokens, broke = await self._plan_procedural(ctx, snapshot)
                plan.procedural = candidates
                plan.tier3_suggestions = suggestions
                plan.tokens += tokens
                if broke:
                    self._defer(plan, "procedural", reason="procedural_partial")
                else:
                    plan.completed_cycles.add("procedural")
        elif "procedural" in snapshot.due_cycles:
            plan.completed_cycles.add("procedural")  # due but no behaviour to analyse

        # Same-tick de-confliction: never forget an episodic that this tick just
        # promoted (its epiphany_contributor flag isn't written until act()).
        if plan.promote and plan.forget:
            promoted_ids = {sid for p in plan.promote for sid in p.source_episodic_ids}
            plan.forget = [f for f in plan.forget if f.memory_id not in promoted_ids]

        return plan

    async def _reachable(self, plan: DreamPlan) -> bool:
        """Provider reachability, probed at most once per tick (degrade-by-cost)."""
        if self._reachable_cache is None:
            self._reachable_cache = await self._health_gate()
        return self._reachable_cache

    @staticmethod
    def _defer(plan: DreamPlan, cycle: str, *, reason: str = "provider_offline") -> None:
        plan.deferred.append(cycle)
        plan.status = "partial"
        plan.reason = plan.reason or reason

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

    async def _plan_promotions(
        self, ctx: TickContext, snapshot: DreamSnapshot
    ) -> tuple[List[PromotionAction], int, bool]:
        min_size = int(self._s.promotion_min_cluster_size)
        clusters = greedy_cosine_clusters(
            snapshot.episodic_vectors,
            threshold=float(self._s.promotion_similarity_threshold),
            min_size=min_size,
        )
        existing = [set(s) for s in snapshot.existing_sources]
        confidence = float(getattr(self._s, "promotion_start_confidence", 0.6))
        cap = int(self._s.max_promotions_per_run)
        breaker = self._breaker_for(ctx.user_id)

        actions: List[PromotionAction] = []
        tokens = 0
        broke = False
        for cluster in clusters:
            if len(actions) >= cap:
                break
            ids = [str(m.get("memory_id")) for m in cluster if m.get("memory_id")]
            id_set = set(ids)
            # Skip clusters already covered by an existing semantic (idempotency).
            if any(len(id_set & cov) >= min_size for cov in existing):
                continue
            texts = [str(m.get("text") or "") for m in cluster if m.get("text")]
            if not texts:
                continue
            try:
                result = await breaker.call(self._synthesizer, texts)
            except CircuitBreakerOpenError:
                broke = True
                break
            except Exception:  # noqa: BLE001 — provider failure → defer the rest
                broke = True
                break
            tokens += int((result or {}).get("tokens", 0) or 0)
            text = str((result or {}).get("text") or "").strip()
            if not text:  # reject garbage: an unparseable synthesis is skipped
                continue
            actions.append(
                PromotionAction(text=text, confidence=confidence, source_episodic_ids=ids)
            )
            # Treat the just-synthesized cluster as covered so an overlapping later
            # cluster in the same run isn't double-promoted.
            existing.append(id_set)
        return actions, tokens, broke

    async def _plan_procedural(
        self, ctx: TickContext, snapshot: DreamSnapshot
    ) -> tuple[List[ProceduralCandidate], List[Tier3Suggestion], int, bool]:
        candidates: List[ProceduralCandidate] = []
        suggestions: List[Tier3Suggestion] = []
        tokens = 0
        broke = False
        try:
            result = await self._breaker_for(ctx.user_id).call(
                self._proposer, snapshot.procedural_observations
            )
        except CircuitBreakerOpenError:
            return candidates, suggestions, 0, True
        except Exception:  # noqa: BLE001 — provider failure → defer
            return candidates, suggestions, 0, True

        tokens += int((result or {}).get("tokens", 0) or 0)
        for rule in (result or {}).get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            rule_key = str(rule.get("rule_key") or "").strip()
            rule_text = str(rule.get("rule_text") or "").strip()
            tier = _classify_tier(rule_key)
            if tier is None or not rule_text:  # unknown namespace / empty → reject
                continue
            if tier == 3:  # locked: never written, only a read-only suggestion
                suggestions.append(Tier3Suggestion(rule_key=rule_key, rule_text=rule_text))
                continue
            candidates.append(
                ProceduralCandidate(
                    rule_key=rule_key,
                    rule_text=rule_text,
                    tier=tier,
                    samples=max(1, int(rule.get("samples", 1) or 1)),
                    example=str(rule.get("example") or rule_text)[:300],
                )
            )
        return candidates, suggestions, tokens, broke

    def _tier_thresholds(self, tier: int) -> tuple[int, int]:
        if tier == 1:
            return (
                int(self._s.procedural_tier1_window_days),
                int(self._s.procedural_tier1_min_samples),
            )
        return (
            int(self._s.procedural_tier2_window_days),
            int(self._s.procedural_tier2_min_samples),
        )

    def _merge_evidence(
        self, existing: Optional[Dict[str, Any]], candidate: ProceduralCandidate, now: datetime
    ) -> Dict[str, Any]:
        prior = dict((existing or {}).get("evidence") or {})
        day = now.date().isoformat()
        days = set(prior.get("observation_days") or [])
        days.add(day)
        examples = list(prior.get("examples") or [])
        if candidate.example and candidate.example not in examples:
            examples.append(candidate.example)
        return {
            "observation_days": sorted(days),
            "sample_count": int(prior.get("sample_count", 0) or 0) + candidate.samples,
            "first_seen": prior.get("first_seen") or now.isoformat(),
            "examples": examples[:5],
        }

    @staticmethod
    def _is_mature(evidence: Dict[str, Any], window_days: int, min_samples: int) -> bool:
        return (
            len(evidence.get("observation_days") or []) >= window_days
            and int(evidence.get("sample_count", 0) or 0) >= min_samples
        )

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

        # Cycle A — Promotion. Write the synthesized semantic, flag each source
        # episodic as an epiphany contributor (so forgetting spares it), audit.
        for p in plan.promote:
            new_id = await asyncio.to_thread(
                self._reader.write_semantic, user_id, p.text, p.confidence, p.source_episodic_ids
            )
            for episodic_id in p.source_episodic_ids:
                await asyncio.to_thread(self._reader.flag_contributor, user_id, episodic_id)
            await asyncio.to_thread(
                self._audit.record,
                user_id=user_id,
                cycle="promotion",
                action="promote",
                target_kind="semantic",
                target_id=new_id,
                after_state={
                    "text": p.text,
                    "confidence": p.confidence,
                    "source_episodic_ids": p.source_episodic_ids,
                },
                reversible_until=reversible_until,
            )

        # Cycle D — Procedural. Accumulate evidence (observing), mature → proposed
        # (never auto-active; user approval is the gate to the prompt). Tier-3 is
        # locked: the engine logs a read-only suggestion and writes no rule.
        for candidate in plan.procedural:
            existing = await asyncio.to_thread(
                self._procedural.get, user_id, candidate.rule_key
            )
            evidence = self._merge_evidence(existing, candidate, now)
            await asyncio.to_thread(
                self._procedural.upsert_observation,
                user_id=user_id,
                rule_key=candidate.rule_key,
                rule_text=candidate.rule_text,
                tier=candidate.tier,
                confidence=0.0,
                evidence=evidence,
            )
            window_days, min_samples = self._tier_thresholds(candidate.tier)
            if self._is_mature(evidence, window_days, min_samples):
                promoted = await asyncio.to_thread(
                    self._procedural.promote_to_proposed,
                    user_id=user_id,
                    rule_key=candidate.rule_key,
                )
                if promoted is not None:  # only audit the actual observing→proposed flip
                    await asyncio.to_thread(
                        self._audit.record,
                        user_id=user_id,
                        cycle="procedural",
                        action="propose",
                        target_kind="procedural_rule",
                        target_id=candidate.rule_key,
                        after_state={"rule_text": candidate.rule_text, "tier": candidate.tier},
                    )

        for suggestion in plan.tier3_suggestions:
            await asyncio.to_thread(
                self._audit.record,
                user_id=user_id,
                cycle="procedural",
                action="tier3_suggestion",
                target_kind="procedural_rule",
                target_id=suggestion.rule_key,
                after_state={"rule_text": suggestion.rule_text, "tier": 3},
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

    def fetch_episodic_vectors(self, user_id: str) -> List[Dict[str, Any]]:
        return self._b.get_episodic_points_with_vectors(user_id, limit=2000)

    def existing_semantic_sources(self, user_id: str) -> List[List[str]]:
        return self._b.get_semantic_source_sets(user_id, limit=2000)

    def write_semantic(
        self, user_id: str, text: str, confidence: float, source_episodic_ids: List[str]
    ) -> str:
        record = self._b.add_semantic(
            user_id, text, confidence=confidence, source_episodic_ids=source_episodic_ids
        )
        return str(record.metadata.get("memory_id"))

    def flag_contributor(self, user_id: str, memory_id: str) -> None:
        self._b.update_metadata(memory_id, {"epiphany_contributor": True})


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


def build_auxiliary_epiphany_synthesizer(
    config: Any,
) -> Callable[[List[str]], Awaitable[Dict[str, Any]]]:
    """Async epiphany synthesizer: condense a cluster of episodic texts into one
    enduring semantic statement via the auxiliary provider (direct httpx)."""

    async def _synthesize(observations: List[str]) -> Dict[str, Any]:
        provider = config.llm.get_role_provider("auxiliary")
        api_base = str(getattr(provider, "api_base", "") or "").rstrip("/")
        if not api_base:
            raise RuntimeError("auxiliary api_base not configured")
        model_name = config.llm.get_role_model_name("auxiliary", "en")
        bare = model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
        joined = "\n".join(f"- {o[:300]}" for o in observations[:12])
        prompt = _EPIPHANY_PROMPT.format(observations=joined)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": bare,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    # Generous cap: reasoning auxiliary models (e.g. gemma-4-e2b)
                    # spend budget "thinking" first; a tight cap truncates before the
                    # answer and returns empty content.
                    "max_tokens": 512,
                },
                headers={"Authorization": f"Bearer {getattr(provider, 'api_key', None) or 'no-key'}"},
            )
            resp.raise_for_status()
            data = resp.json()
        content = _strip_reasoning(str(data["choices"][0]["message"]["content"] or ""))
        tokens = int((data.get("usage") or {}).get("total_tokens", 0) or 0)
        return {"text": content, "tokens": tokens}

    return _synthesize


_PROCEDURAL_PROMPT = (
    "From the observations below about one user's interactions, infer up to 3 stable "
    "BEHAVIOURAL preferences (how they like answers formatted or styled). Use a "
    "namespaced rule_key: 'format.*' for formatting/UI, 'style.*' for tone/length. "
    "Do NOT propose core-logic or safety rules.\n"
    'Reply ONLY with JSON: {{"rules":[{{"rule_key":"format.bullets","rule_text":"...",'
    '"samples":1}}]}}. Empty list if nothing is consistent.\n\nOBSERVATIONS:\n{observations}'
)


def build_auxiliary_procedural_proposer(
    config: Any,
) -> Callable[[List[str]], Awaitable[Dict[str, Any]]]:
    """Async procedural-rule proposer over recent behaviour (auxiliary provider).
    Returns ``{"rules": [{rule_key, rule_text, samples}], "tokens": int}``; the
    engine classifies tier from the rule_key namespace (never trusts the model)."""

    async def _propose(observations: List[str]) -> Dict[str, Any]:
        provider = config.llm.get_role_provider("auxiliary")
        api_base = str(getattr(provider, "api_base", "") or "").rstrip("/")
        if not api_base:
            raise RuntimeError("auxiliary api_base not configured")
        model_name = config.llm.get_role_model_name("auxiliary", "en")
        bare = model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
        joined = "\n".join(f"- {o[:200]}" for o in observations[:40])
        prompt = _PROCEDURAL_PROMPT.format(observations=joined)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": bare,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    # Generous cap so a reasoning model reaches the JSON after thinking.
                    "max_tokens": 512,
                },
                headers={"Authorization": f"Bearer {getattr(provider, 'api_key', None) or 'no-key'}"},
            )
            resp.raise_for_status()
            data = resp.json()
        content = str(data["choices"][0]["message"]["content"] or "")
        tokens = int((data.get("usage") or {}).get("total_tokens", 0) or 0)
        parsed = extract_json_object(content) or {}
        rules = parsed.get("rules") if isinstance(parsed.get("rules"), list) else []
        return {"rules": rules, "tokens": tokens}

    return _propose


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
    procedural_store: Any = None,
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
        synthesizer=build_auxiliary_epiphany_synthesizer(config),
        proposer=build_auxiliary_procedural_proposer(config),
        procedural_store=procedural_store,
        health_gate=build_provider_reachability_gate(config),
        opt_out_checker=make_opt_out_checker(settings_store),
        settings=cognitive,
    )
