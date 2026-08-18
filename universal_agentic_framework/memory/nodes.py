"""Memory lifecycle nodes: load and update operations for graph execution.

SOURCE OF TRUTH OWNERSHIP:
- Digest chain: LangGraph orchestration (GraphState.digest_context)
  - Created by: performance_nodes.conversation_compression_node()
  - Propagated by: graph_builder.node_update_memory() → update_memory_node()
  - Persisted in: Mem0 memory record metadata
  
- Long-memory records: Mem0 + Qdrant vector store (MemoryBackend)
  - Accessed via: load_memory_node(), update_memory_node()
    - Metadata owned by: Mem0MemoryBackend (cache + canonical Mem0 metadata fields)
  - Ratings stored in: PostgreSQL (via backend) + memory record metadata
  
- Co-occurrence links: PostgreSQL durable + Mem0 metadata projection (planned Phase 3)
  - Tracked by: MemoryCoOccurrenceTracker
  - Currently: in-memory only (non-persistent)
  - Planned: hybrid persistence with decay model

READ PATTERN:
1. load_memory_node() queries memory backend
2. Backend searches Mem0/Qdrant + retrieves related memories
3. Digest context extracted from loaded_memory (sorted by timestamp)
4. Related memories attached to memory_analytics

WRITE PATTERN:
1. update_memory_node() receives (user_id, text, digest_chain)
2. Passes digest_chain to backend.upsert() for embedding in metadata
3. Backend stores to Mem0 + updates metadata consistency state
4. Co-occurrence tracker records memory IDs for knowledge graph

VALIDATION GATES:
- Digest metadata must persist end-to-end (checkpoint #7 in plan)
- Rating signals must correctly emit after retrieval (checkpoint #8)
- Related memories must be includable in retrieval context (feature flag)

See: docs/technical_architecture.md (Memory Architecture) for full architecture
"""

from __future__ import annotations

import os
import structlog
import inspect
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from . import MemoryBackend
from .factory import build_memory_backend
from universal_agentic_framework.config import load_core_config, load_features_config
from universal_agentic_framework.monitoring import metrics

logger = structlog.get_logger(__name__)


# --- Retrieval-touch persistence (plan-memory.md Phase 2) -------------------
# After assembling loaded_memory we best-effort persist access_count/last_accessed
# for the injected memories so the forgetting cycle can trust "never retrieved".
# Debounced (≥60s per memory) so a chatty user can't storm Qdrant; Redis-backed
# with an in-process fallback so it works (degraded) when Redis is down.
_TOUCH_DEBOUNCE_SECONDS = 60
_TOUCH_KEY_PREFIX = "mem:touch:"
_local_touch_cache: Dict[str, float] = {}
_touch_redis_client: Any = None
_touch_redis_checked = False


def _get_touch_redis() -> Any:
    """Lazily build a sync Redis client for touch debouncing (best-effort)."""
    global _touch_redis_client, _touch_redis_checked
    if _touch_redis_checked:
        return _touch_redis_client
    _touch_redis_checked = True
    try:
        import redis as _redis

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = _redis.from_url(
            url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        client.ping()  # one probe; fall back to in-process cache on failure
        _touch_redis_client = client
    except Exception:
        _touch_redis_client = None
    return _touch_redis_client


def _should_touch(memory_id: str) -> bool:
    """Return True if this memory hasn't been touched within the debounce window
    (and atomically mark it touched). Uses Redis SET NX EX, else a local cache."""
    client = _get_touch_redis()
    if client is not None:
        try:
            # SET NX returns truthy only when the key was absent → first touch in window.
            return bool(client.set(f"{_TOUCH_KEY_PREFIX}{memory_id}", "1", nx=True, ex=_TOUCH_DEBOUNCE_SECONDS))
        except Exception:
            pass  # fall through to in-process debounce
    now = time.monotonic()
    last = _local_touch_cache.get(memory_id)
    if last is not None and (now - last) < _TOUCH_DEBOUNCE_SECONDS:
        return False
    _local_touch_cache[memory_id] = now
    return True


def _persist_retrieval_touch(store: Any, injected: List[Dict[str, Any]]) -> None:
    """Persist access_count/last_accessed for the injected memories (top_k).

    Best-effort and debounced: never raises, never blocks the turn. Requires the
    backend to expose ``update_metadata`` (text-preserving). The loaded metadata's
    ``access_count`` already reflects the in-memory access bump when importance
    scoring is enabled, so we persist that value plus a fresh ``last_accessed``.
    """
    if not hasattr(store, "update_metadata"):
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    for entry in injected:
        metadata = entry.get("metadata") or {}
        memory_id = metadata.get("memory_id")
        if not memory_id:
            continue
        if not _should_touch(str(memory_id)):
            continue
        try:
            store.update_metadata(
                str(memory_id),
                {
                    "access_count": int(metadata.get("access_count", 0)),
                    "last_accessed": now_iso,
                },
            )
        except Exception:
            logger.debug("retrieval_touch_failed", memory_id=memory_id)


def _is_digest_memory(entry: Dict[str, Any]) -> bool:
    metadata = entry.get("metadata") or {}
    return bool(metadata.get("digest_id") or metadata.get("type") == "summary")


def _parse_sort_timestamp(value: Any) -> float:
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return 0.0
    return 0.0


def _extract_digest_context(
    loaded_memory: list[Dict[str, Any]],
    *,
    max_items: int = 2,
) -> list[Dict[str, Any]]:
    digests = [m for m in loaded_memory if _is_digest_memory(m)]
    for item in digests:
        metadata = item.setdefault("metadata", {})
        metadata["is_digest"] = True
    digests.sort(
        key=lambda item: _parse_sort_timestamp(
            (item.get("metadata") or {}).get("created_at")
            or (item.get("metadata") or {}).get("timestamp")
        ),
        reverse=True,
    )
    return digests[:max_items]


def load_memory_node(
    state: Dict[str, Any],
    backend: Optional[MemoryBackend] = None,
    top_k: int = 5,
    include_related: bool = False,
    cognitive_enabled: bool = False,
    semantic_top_k: int = 3,
    episodic_top_k: int = 5,
) -> Dict[str, Any]:
    """Load relevant memory for a user based on latest message.

    Expects state to include keys: "user_id" and "messages" (list of dicts with a "content").

    Args:
        state: Current graph state with user_id, messages, and optional session_id
        backend: Optional memory backend (uses config if None)
        top_k: Number of primary memories to retrieve (default: 5)
        include_related: If True, fetch related memories via co-occurrence graph (default: False)
        cognitive_enabled: If True (plan-memory.md Phase 2), retrieve via load_blended
            (semantic prepended before episodic) and persist a retrieval-touch on the
            injected memories. Off → byte-identical to the legacy single store.load().
        semantic_top_k / episodic_top_k: per-tier caps for the blended retrieval.

    Returns:
        Updated state with loaded_memory, semantic_memory and memory_analytics fields

    KILL SWITCH: Set features.memory_load_enabled=false to disable this operation during emergency rollback.
    """
    # EMERGENCY ROLLBACK: Check memory_load_enabled feature flag
    try:
        features = load_features_config()
        if not features.memory_load_enabled:
            logger.info("memory_load_disabled_by_feature_flag")
            # Return empty loaded_memory to bypass retrieval
            state["loaded_memory"] = []
            state["semantic_memory"] = []
            state["digest_context"] = []
            state["memory_analytics"] = {
                "primary_count": 0,
                "related_count": 0,
                "digest_count": 0,
                "total_count": 0,
                "include_related_enabled": False,
            }
            return state
    except Exception as e:
        logger.warning("failed_to_load_features_for_kill_switch", error=str(e))
        # Continue normally if features config unavailable

    user_id = state.get("user_id")
    messages = state.get("messages", [])
    query = messages[-1]["content"] if messages else None
    session_id = state.get("session_id", user_id)  # Fallback to user_id for session tracking
    profile_name = state.get("profile_name", "default")

    if backend is None:
        cfg = load_core_config()
        store = build_memory_backend(cfg)
    else:
        store = backend
    
    # Load memories. Cognitive blend (flag on) splits one search into
    # semantic-prepended-before-episodic; otherwise the legacy single load
    # (with optional co-occurrence related-memory expansion).
    use_blended = bool(cognitive_enabled) and hasattr(store, "load_blended")
    try:
        if use_blended:
            results = store.load_blended(
                user_id=user_id,
                query=query,
                semantic_top_k=semantic_top_k,
                episodic_top_k=episodic_top_k,
            )
        elif hasattr(store, 'load'):
            load_kwargs = {
                "user_id": user_id,
                "query": query,
                "top_k": top_k,
            }
            # Only add include_related if backend supports it
            import inspect
            sig = inspect.signature(store.load)
            if "include_related" in sig.parameters:
                load_kwargs["include_related"] = include_related
                load_kwargs["session_id"] = session_id

            results = store.load(**load_kwargs)
        else:
            results = store.load(user_id=user_id, query=query, top_k=top_k)
    except Exception as e:
        logger.error("memory_load_failed", error=str(e), user_id=user_id, exc_info=True)
        results = []

    # Extract memory data and analytics
    loaded_memory = [{"text": r.text, "metadata": r.metadata} for r in results]

    # Semantic-tier subset (graph-produced; populated only under the cognitive blend).
    semantic_memory = (
        [m for m in loaded_memory if (m.get("metadata") or {}).get("cognitive_tier") == "semantic"]
        if use_blended
        else []
    )

    # Digest context is retrieval-first-class: always attempt to include recent
    # digest summaries, even when the query itself matches only non-summary items.
    digest_source = list(loaded_memory)
    try:
        recent_results = store.load(user_id=user_id, query=None, top_k=max(top_k * 2, 10))
        digest_source.extend(
            {"text": r.text, "metadata": r.metadata}
            for r in recent_results
        )
    except Exception:
        pass

    digest_context = _extract_digest_context(digest_source, max_items=2)
    
    # Track memory analytics
    memory_count = len(loaded_memory)
    related_count = sum(1 for m in loaded_memory if m.get("metadata", {}).get("is_related", False))
    digest_count = len(digest_context)
    
    # Update Prometheus metrics
    if loaded_memory:
        avg_importance = sum(
            m.get("metadata", {}).get("importance_score", 0.0) 
            for m in loaded_memory
        ) / len(loaded_memory)
        metrics.track_memory_quality(profile_name, avg_importance)
    
    if related_count > 0:
        metrics.track_related_memories(profile_name, related_count)
    
    state["loaded_memory"] = loaded_memory
    state["semantic_memory"] = semantic_memory
    state["digest_context"] = digest_context
    state["memory_analytics"] = {
        "primary_count": memory_count - related_count,
        "related_count": related_count,
        "digest_count": digest_count,
        "total_count": memory_count,
        "include_related_enabled": include_related,
        "semantic_count": len(semantic_memory),
    }

    # Persist a retrieval-touch on the injected memories so the forgetting cycle
    # can trust access_count/last_accessed. Cognitive-blend only; best-effort.
    if use_blended and loaded_memory:
        _persist_retrieval_touch(store, loaded_memory)

    logger.info(
        "memory_loaded",
        user_id=user_id,
        primary_count=memory_count - related_count,
        related_count=related_count,
        digest_count=digest_count,
        total_count=memory_count,
        semantic_count=len(semantic_memory),
    )

    return state


def update_memory_node(
    state: Dict[str, Any],
    text: str,
    metadata: Optional[dict] = None,
    backend: Optional[MemoryBackend] = None,
    messages: Optional[list] = None,
    digest_chain: Optional[list[dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write a distilled memory entry for the user.

    Expects state to include key: "user_id".
    
    KILL SWITCH: Set features.memory_update_enabled=false to disable this operation during emergency rollback.
    """
    # EMERGENCY ROLLBACK: Check memory_update_enabled feature flag
    try:
        features = load_features_config()
        if not features.memory_update_enabled:
            logger.info("memory_update_disabled_by_feature_flag")
            # Skip the update operation
            return state
    except Exception as e:
        logger.warning("failed_to_load_features_for_kill_switch", error=str(e))
        # Continue normally if features config unavailable
    
    user_id = state.get("user_id")
    logger.info("update_memory_node called", user_id=user_id, text_length=len(text) if text else 0)
    
    if backend is None:
        logger.info("No backend provided, building from config", user_id=user_id)
        cfg = load_core_config()
        store = build_memory_backend(cfg)
        logger.info("Memory backend built from config", backend_type=type(store).__name__)
    else:
        logger.info("Using provided backend", backend_type=type(backend).__name__)
        store = backend
    
    logger.info("Attempting upsert", user_id=user_id, text_length=len(text) if text else 0, backend_type=type(store).__name__)
    try:
        upsert_kwargs = {
            "user_id": user_id,
            "text": text,
            "metadata": metadata,
            "messages": messages,
        }
        try:
            sig = inspect.signature(store.upsert)
            if "digest_chain" in sig.parameters and digest_chain:
                upsert_kwargs["digest_chain"] = digest_chain
        except Exception:
            # If signature inspection fails, use backward-compatible call path.
            pass

        result = store.upsert(**upsert_kwargs)
        logger.info("Upsert successful", user_id=user_id, memory_id=result.metadata.get("memory_id") if hasattr(result, "metadata") else "unknown")
        return state
    except Exception as e:
        logger.error("Upsert failed", error=str(e), user_id=user_id, exc_info=True)
        raise
