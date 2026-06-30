"""Procedural-overrides load node (plan-memory.md Phase 5).

Loads the user's *active* learned behavioural rules (approved procedural
overrides) so the respond node can merge them onto the static YAML persona as a
``=== LEARNED USER PREFERENCES ===`` block. Only ``active`` rules are loaded —
``observing``/``proposed`` rules never reach the prompt until the user approves
them. Gated by the ``procedural_overrides_enabled`` feature flag (off → empty,
so the merge is a no-op). Best-effort + Redis-cached (``procedural:{user_id}``);
never raises (a failure just yields no overrides for the turn).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import structlog

from universal_agentic_framework.config import load_features_config

logger = structlog.get_logger(__name__)

_PROCEDURAL_CACHE_PREFIX = "procedural:"
_PROCEDURAL_CACHE_TTL = 300  # seconds

_procedural_store: Any = None
_procedural_store_built = False
_redis_client: Any = None
_redis_checked = False


def _get_store() -> Any:
    """Lazily build a single ProceduralOverrideStore (best-effort)."""
    global _procedural_store, _procedural_store_built
    if _procedural_store_built:
        return _procedural_store
    _procedural_store_built = True
    try:
        from backend.db import ProceduralOverrideStore, init_db_pool

        _procedural_store = ProceduralOverrideStore(init_db_pool())
    except Exception as exc:  # noqa: BLE001 — degrade to no overrides
        logger.warning("procedural_store_unavailable", error=str(exc))
        _procedural_store = None
    return _procedural_store


def _get_redis() -> Any:
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis as _redis

        client = _redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
    except Exception:  # noqa: BLE001 — cache is optional
        _redis_client = None
    return _redis_client


def invalidate_procedural_cache(user_id: str) -> None:
    """Drop a user's cached active-rules list (called on approve/reject in HITL)."""
    client = _get_redis()
    if client is None:
        return
    try:
        client.delete(f"{_PROCEDURAL_CACHE_PREFIX}{user_id}")
    except Exception:  # noqa: BLE001
        pass


def load_active_procedural(user_id: str, *, store: Any = None) -> List[Dict[str, Any]]:
    """Return the user's active procedural rules ``[{rule_key, rule_text, tier}]``,
    Redis-cached. ``store`` is injectable for tests."""
    cache_key = f"{_PROCEDURAL_CACHE_PREFIX}{user_id}"
    client = _get_redis()
    if client is not None:
        try:
            cached = client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:  # noqa: BLE001
            pass

    rules: List[Dict[str, Any]] = []
    resolved_store = store if store is not None else _get_store()
    if resolved_store is not None:
        try:
            rows = resolved_store.list_active(user_id)
            rules = [
                {
                    "rule_key": r.get("rule_key"),
                    "rule_text": r.get("rule_text"),
                    "tier": r.get("tier"),
                }
                for r in rows
                if r.get("rule_text")
            ]
        except Exception as exc:  # noqa: BLE001 — never block the turn
            logger.warning("procedural_load_failed", user_id=user_id, error=str(exc))

    if client is not None:
        try:
            client.set(cache_key, json.dumps(rules), ex=_PROCEDURAL_CACHE_TTL)
        except Exception:  # noqa: BLE001
            pass
    return rules


def node_load_procedural(state: Dict[str, Any], *, store: Any = None) -> Dict[str, Any]:
    """Graph load-phase node: populate ``state["procedural_overrides"]`` with the
    user's active rules when ``procedural_overrides_enabled`` is on, else []."""
    try:
        features = load_features_config()
        enabled = getattr(features, "procedural_overrides_enabled", False)
    except Exception:  # noqa: BLE001
        enabled = False

    user_id = state.get("user_id")
    if not enabled or not user_id:
        state["procedural_overrides"] = []
        return state

    state["procedural_overrides"] = load_active_procedural(user_id, store=store)
    return state
