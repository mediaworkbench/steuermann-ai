from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.db import (
    SettingsStore,
    LLMCapabilityProbeStore,
    RoleToolPermissionStore,
    GlobalSettingsStore,
    HeartbeatRunStore,
    HEARTBEAT_RATE_SETTING_KEY,
    HEARTBEAT_COOLDOWNS_SETTING_KEY,
)
from backend.llm_capability_probe import LLMCapabilityProbeRunner
from backend.auth import (
    ADMIN_ROLE,
    RESEARCHER_ROLE,
    USER_ROLE,
    CurrentUser,
    current_user_id,
    require_admin,
    resolve_current_user,
)
from backend.single_user import require_api_access
from backend.version import get_framework_version
from universal_agentic_framework.cli.ingest import resolve_runtime_ingestion_defaults
from universal_agentic_framework.config import (
    get_active_profile_id,
    load_core_config,
    load_profile_metadata,
    load_profile_ui_config,
    load_tools_config,
)
from universal_agentic_framework.llm.provider_registry import normalize_model_id, parse_model_id

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api", tags=["settings"], dependencies=[Depends(require_api_access)])

KNOWN_PROVIDER_PREFIXES = {
    "openai",
    "ollama",
    "lm_studio",
    "openrouter",
    "anthropic",
    "azure",
    "bedrock",
    "groq",
    "mistral",
    "vertex_ai",
}


def _resolve_primary_provider_endpoint() -> str:
    """Resolve primary chat provider endpoint strictly from active config."""
    try:
        core = load_core_config()
        primary = core.llm.get_role_provider("chat")
        if primary and getattr(primary, "api_base", None):
            return str(primary.api_base).rstrip("/")
    except Exception as exc:
        raise ValueError(f"Failed to resolve primary provider endpoint from config: {exc}") from exc

    raise ValueError("No api_base configured for primary chat provider")


class ProfileConfigResponse(BaseModel):
    id: str
    display_name: str
    role_label: str
    description: Optional[str] = None
    app_name: Optional[str] = None
    theme: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class UserSettings(BaseModel):
    tool_toggles: Dict[str, bool] = Field(default_factory=dict)
    rag_config: Dict[str, Any] = Field(default_factory=lambda: {"top_k": 5, "enabled": True})
    analytics_preferences: Dict[str, Any] = Field(default_factory=dict)
    preferred_model: str | None = None
    preferred_models: Dict[str, Optional[str]] = Field(default_factory=dict)
    theme: str = Field(default="auto", description="light, dark, or auto")
    language: str = Field(default="en")


class UserSettingsResponse(BaseModel):
    user_id: str
    tool_toggles: Dict[str, bool]
    rag_config: Dict[str, Any]
    analytics_preferences: Dict[str, Any]
    preferred_model: Optional[str]
    preferred_models: Dict[str, Optional[str]]
    theme: str
    language: str
    updated_at: Optional[str]
    # Server-computed allowlist of tool ids for this user's role (admin ⇒ all).
    # Read-only: never accepted on the writable `UserSettings` input model.
    allowed_tools: List[str] = Field(default_factory=list)


# Roles whose tool access an admin can configure (administrators are always unrestricted).
CONFIGURABLE_ROLES = (USER_ROLE, RESEARCHER_ROLE)


class RoleToolsResponse(BaseModel):
    tools: List[Dict[str, str]]  # full catalog: {id, label, group}
    roles: Dict[str, List[str]]  # role_name -> allowed tool ids


class RoleToolsUpdate(BaseModel):
    allowed_tools: List[str]


class SystemConfigResponse(BaseModel):
    available_tools: List[Dict[str, str]]
    rag_defaults: Dict[str, Any]
    default_model: str
    framework_version: str
    profile: ProfileConfigResponse
    supported_languages: List[str] = Field(default_factory=list)
    model_roles: List[Dict[str, Any]] = Field(default_factory=list)


class ReingestAllResponse(BaseModel):
    status: str
    source: str
    collection: str
    language: str
    processed: int
    skipped: int
    errors: int
    total_chunks: int
    output_tail: str


def _get_settings_store(request: Request) -> SettingsStore:
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Settings store unavailable")
    return store


def _get_role_tool_store(request: Request) -> RoleToolPermissionStore:
    store = getattr(request.app.state, "role_tool_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Role tool store unavailable")
    return store


def _resolve_user_allowed_tools(request: Request, current_user: CurrentUser) -> List[str]:
    """Tool ids the user may use: admin ⇒ all; else the role allowlist (``[]`` if unconfigured).

    Defensive: when no role-tool store is wired (some dev/test setups) nothing is
    hidden — the full catalog is returned.
    """
    if current_user.role == ADMIN_ROLE:
        return _catalog_tool_ids()
    store = getattr(request.app.state, "role_tool_store", None)
    if store is None:
        return _catalog_tool_ids()
    allowed = store.get_allowed_tools(current_user.role)
    return allowed if allowed is not None else []


def _role_tools_snapshot(store: RoleToolPermissionStore) -> Dict[str, Any]:
    """Catalog + per-configurable-role allowlists for the admin UI."""
    roles: Dict[str, List[str]] = {}
    for role in CONFIGURABLE_ROLES:
        stored = store.get_allowed_tools(role)
        roles[role] = stored if stored is not None else []
    return {"tools": _load_tool_catalog(), "roles": roles}


def _get_probe_store(request: Request) -> Optional[LLMCapabilityProbeStore]:
    """Get probe store from app state; returns None if not available."""
    return getattr(request.app.state, "llm_capability_probe_store", None)


def _trigger_reprobe_on_model_change(request: Request, new_model: str) -> None:
    """Trigger curated reprobe for the specific model the user switched to."""
    probe_store = _get_probe_store(request)
    if not probe_store:
        logger.debug("Probe store unavailable; skipping reprobe trigger")
        return

    try:
        profile_id = get_active_profile_id()
        runner = LLMCapabilityProbeRunner(profile_id=profile_id)
        results = runner.reprobe_for_model(new_model)

        if not results:
            logger.info(
                "Reprobe skipped: model not in current config",
                extra={"profile_id": profile_id, "new_model": new_model},
            )
            return

        for result in results:
            probe_store.upsert_probe_result(result)

        logger.info(
            "Curated reprobe completed on preferred model change",
            extra={
                "profile_id": profile_id,
                "new_model": new_model,
                "reprobe_count": len(results),
            },
        )
    except Exception as exc:
        logger.warning(
            "Reprobe failed on model change",
            extra={
                "new_model": new_model,
                "error": str(exc),
            },
        )


def _normalize_preferred_model_value(raw_model: Optional[str]) -> Optional[str]:
    if not raw_model:
        return None
    preferred_model = str(raw_model).strip()
    if not preferred_model:
        return None
    if "/" in preferred_model:
        try:
            provider = parse_model_id(preferred_model).provider
            if provider in KNOWN_PROVIDER_PREFIXES:
                preferred_model = normalize_model_id(preferred_model)
        except Exception:
            # Unknown/non-provider slash patterns (e.g., org/model) are preserved.
            pass
    return preferred_model


async def _validate_chat_preference(model_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not model_name:
        return None, None

    from backend.routers.chat import _validate_preferred_model

    return await _validate_preferred_model(model_name)


async def _fetch_provider_models(
    provider: Any, preferred_language: str = "en"
) -> Tuple[List[str], Dict[str, int]]:
    """Return (canonical_model_names, context_window_by_canonical_name).

    context_window_by_canonical_name maps canonical model IDs to max_context_length
    as reported by the provider's /models endpoint (e.g. LM Studio).
    Providers that do not report this field (Ollama) return an empty dict.
    """
    provider_models = [m for m in provider.models.model_dump().values() if m]
    default_model = (
        getattr(provider.models, preferred_language, None)
        or getattr(provider.models, "en", None)
        or (provider_models[0] if provider_models else None)
    )
    provider_prefix = "openai"
    if default_model:
        try:
            provider_prefix = parse_model_id(str(default_model)).provider
        except Exception:
            pass

    def _canonical_model_name(raw_name: str) -> str:
        raw_name = str(raw_name).strip()
        if not raw_name:
            return raw_name
        if "/" not in raw_name:
            return f"{provider_prefix}/{raw_name}"
        head = raw_name.split("/", 1)[0].strip().lower()
        if head in KNOWN_PROVIDER_PREFIXES:
            return raw_name
        return f"{provider_prefix}/{raw_name}"

    async with httpx.AsyncClient(timeout=5.0) as client:
        base = str(getattr(provider, "api_base", "") or "").rstrip("/")
        if not base:
            return [], {}
        try:
            resp = await client.get(f"{base}/models")
            resp.raise_for_status()
            data = resp.json()
            raw_models = [m for m in data.get("data", []) if m.get("id")]
            model_names = [m.get("id") for m in raw_models]
        except Exception:
            if provider_prefix != "ollama":
                raise
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            model_names = [m.get("name") for m in data.get("models", []) if m.get("name")]
            raw_models = []

    canonical_names = [_canonical_model_name(n) for n in model_names]
    canonical_names = [
        n for n in canonical_names
        if not any(skip in n.lower() for skip in ("bge-", "nomic-embed", "embed"))
    ]

    # Build context window map from whatever the provider reports.
    # Prefer context_length (the value LM Studio is actually loaded with) over
    # max_context_length (the model's theoretical upper bound).
    context_windows: Dict[str, int] = {}
    for m in raw_models:
        raw_id = m.get("id")
        ctx = m.get("context_length") or m.get("max_context_length")
        if raw_id and ctx:
            try:
                context_windows[_canonical_model_name(raw_id)] = int(ctx)
            except (TypeError, ValueError):
                pass

    return sorted(set(canonical_names)), context_windows


def _compute_effective_mode(desired_mode: str, probe_row: Optional[Dict[str, Any]]) -> tuple[str, str]:
    if desired_mode != "native":
        return desired_mode, "model_config_non_native_mode"

    if not probe_row:
        return "structured", "probe_missing_forced_structured"

    ttl_seconds = int(os.getenv("LLM_CAPABILITY_PROBE_TTL_SECONDS", "3600"))
    probed_at_raw = str(probe_row.get("probed_at") or "").strip()
    if not probed_at_raw:
        return "structured", "probe_missing_timestamp_forced_structured"
    try:
        dt = datetime.fromisoformat(probed_at_raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
        if age_seconds > ttl_seconds:
            return "structured", "probe_stale_forced_structured"
    except Exception:
        return "structured", "probe_invalid_timestamp_forced_structured"

    mismatch = bool(probe_row.get("capability_mismatch", False))
    bind_failed = probe_row.get("supports_bind_tools") is False
    schema_failed = probe_row.get("supports_tool_schema") is False
    status = str(probe_row.get("status") or "").lower()

    if mismatch or bind_failed or schema_failed or status in {"warning", "error"}:
        return "structured", "probe_capability_mismatch_downgrade"

    return "native", "probe_confirmed_native"


@router.get("/settings/me", response_model=UserSettingsResponse)
def get_user_settings(
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Retrieve the authenticated user's persisted settings or defaults."""
    effective_user_id = current_user.user_id
    store = _get_settings_store(request)
    allowed_tools = _resolve_user_allowed_tools(request, current_user)
    record = store.get_user_settings(effective_user_id)
    if record is None:
        return {
            "user_id": effective_user_id,
            "tool_toggles": {},
            "rag_config": {"top_k": 5, "enabled": True},
            "analytics_preferences": {},
            "preferred_model": None,
            "preferred_models": {},
            "theme": "auto",
            "language": "en",
            "updated_at": None,
            "allowed_tools": allowed_tools,
        }
    record["allowed_tools"] = allowed_tools
    return record


@router.post("/settings/me", response_model=UserSettingsResponse)
async def update_user_settings(
    settings: UserSettings,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> Dict[str, Any]:
    """Persist settings for the authenticated user."""
    effective_user_id = current_user.user_id
    store = _get_settings_store(request)
    
    # Get old settings to detect model changes
    old_record = store.get_user_settings(effective_user_id)
    old_model = old_record.get("preferred_model") if old_record else None

    fields_set = set(getattr(settings, "model_fields_set", set()))

    def _resolved_value(field_name: str, fallback: Any) -> Any:
        if field_name in fields_set:
            return getattr(settings, field_name)
        if old_record is not None:
            return old_record.get(field_name, fallback)
        return fallback
    
    tool_toggles = _resolved_value("tool_toggles", {})
    rag_config = _resolved_value("rag_config", {"top_k": 5, "enabled": True})
    analytics_preferences = _resolved_value("analytics_preferences", {})
    theme = _resolved_value("theme", "auto")
    language = _resolved_value("language", "en")

    has_preferred_model_update = "preferred_model" in fields_set

    preferred_model = _normalize_preferred_model_value(_resolved_value("preferred_model", None))

    raw_preferred_models = _resolved_value("preferred_models", {}) or {}
    if "preferred_models" not in fields_set and has_preferred_model_update:
        raw_preferred_models = {
            role: model_name
            for role, model_name in raw_preferred_models.items()
            if str(role) != "chat"
        }

    preferred_models = {
        str(role): _normalize_preferred_model_value(model_name)
        for role, model_name in raw_preferred_models.items()
    }
    preferred_models = {role: model_name for role, model_name in preferred_models.items() if model_name}

    # Keep legacy chat setting in sync with role-based preference map.
    chat_preferred_model = preferred_models.get("chat")
    if chat_preferred_model:
        preferred_model = chat_preferred_model
    elif preferred_model:
        preferred_models["chat"] = preferred_model

    if preferred_model:
        validated_model, validation_warning = await _validate_chat_preference(preferred_model)
        if (
            validation_warning
            and validated_model is None
            and "not found at configured llm endpoint" in validation_warning.lower()
        ):
            logger.warning(
                "Dropping invalid preferred model during settings update",
                extra={
                    "user_id": effective_user_id,
                    "preferred_model": preferred_model,
                    "warning": validation_warning,
                },
            )
            preferred_model = None
            preferred_models.pop("chat", None)

    record = store.upsert_user_settings(
        user_id=effective_user_id,
        tool_toggles=tool_toggles,
        rag_config=rag_config,
        analytics_preferences=analytics_preferences,
        preferred_model=preferred_model,
        preferred_models=preferred_models,
        theme=theme,
        language=language,
    )
    
    # Clear settings cache for this user to force reload on next chat request
    from backend.routers.chat import _settings_cache
    if effective_user_id in _settings_cache:
        del _settings_cache[effective_user_id]
    
    # Trigger curated reprobe if preferred_model changed
    if preferred_model and preferred_model != old_model:
        _trigger_reprobe_on_model_change(request, preferred_model)

    # allowed_tools is server-computed and read-only — surface the current value.
    record["allowed_tools"] = _resolve_user_allowed_tools(request, current_user)
    return record


@router.get("/admin/role-tools", response_model=RoleToolsResponse)
def get_role_tools(
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin view: full tool catalog + each configurable role's allowed tool ids."""
    return _role_tools_snapshot(_get_role_tool_store(request))


@router.put("/admin/role-tools/{role_name}", response_model=RoleToolsResponse)
def update_role_tools(
    role_name: str,
    payload: RoleToolsUpdate,
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin: set the explicit tool allowlist for a configurable role.

    Administrators are always unrestricted, so they cannot be configured here.
    Submitted ids are validated against the catalog and stored in catalog order.
    """
    if role_name not in CONFIGURABLE_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Role '{role_name}' is not configurable. Configurable roles: {list(CONFIGURABLE_ROLES)}",
        )
    catalog_ids = _catalog_tool_ids()
    catalog_set = set(catalog_ids)
    requested = set(payload.allowed_tools)
    unknown = sorted(requested - catalog_set)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown tool ids: {unknown}")
    # Store deduplicated, in catalog order, always as an explicit list.
    cleaned = [tid for tid in catalog_ids if tid in requested]
    store = _get_role_tool_store(request)
    store.set_allowed_tools(role_name, cleaned)
    return _role_tools_snapshot(store)


# --- Heartbeat rate (admin) --------------------------------------------------

MIN_HEARTBEAT_RATE_MINUTES = 1
MAX_HEARTBEAT_RATE_MINUTES = 1440  # 24h


class HeartbeatRateResponse(BaseModel):
    heartbeat_rate_minutes: int
    default_rate_minutes: int
    enabled: bool
    source: str  # "override" (admin-set) | "default" (from profile core.yaml)
    last_run: Optional[Dict[str, Any]] = None


class HeartbeatRateUpdate(BaseModel):
    heartbeat_rate_minutes: int = Field(
        ..., ge=MIN_HEARTBEAT_RATE_MINUTES, le=MAX_HEARTBEAT_RATE_MINUTES
    )


def _get_global_settings_store(request: Request) -> GlobalSettingsStore:
    store = getattr(request.app.state, "global_settings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Global settings store unavailable")
    return store


def _heartbeat_config() -> Tuple[int, bool, Optional[str]]:
    """Return (default_rate_minutes, enabled, first_task_name) from the profile."""
    hb = getattr(load_core_config(), "heartbeat", None)
    if hb is None:
        return 5, False, None
    first_task = hb.tasks[0].name if hb.tasks else None
    return int(hb.default_rate_minutes), bool(hb.enabled), first_task


def _heartbeat_last_run(request: Request, task_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not task_name:
        return None
    db_pool = getattr(request.app.state, "db_pool", None)
    if db_pool is None:
        return None
    try:
        rows = HeartbeatRunStore(db_pool).recent_runs(task_name, 1)
    except Exception:  # pragma: no cover - best-effort observability
        return None
    if not rows:
        return None
    row = rows[0]
    fired_at = row.get("fired_at")
    return {
        "task_name": row.get("task_name"),
        "status": row.get("status"),
        "duration_ms": row.get("duration_ms"),
        "fired_at": fired_at.isoformat() if hasattr(fired_at, "isoformat") else fired_at,
    }


def _heartbeat_rate_snapshot(request: Request) -> Dict[str, Any]:
    default_rate, enabled, first_task = _heartbeat_config()
    store = _get_global_settings_store(request)
    override = store.get_setting(HEARTBEAT_RATE_SETTING_KEY)
    if override is None:
        rate, source = default_rate, "default"
    else:
        try:
            rate = int(override)
        except (TypeError, ValueError):
            rate = default_rate
        rate = max(MIN_HEARTBEAT_RATE_MINUTES, min(MAX_HEARTBEAT_RATE_MINUTES, rate))
        source = "override"
    return {
        "heartbeat_rate_minutes": rate,
        "default_rate_minutes": default_rate,
        "enabled": enabled,
        "source": source,
        "last_run": _heartbeat_last_run(request, first_task),
    }


@router.get("/admin/settings/heartbeat-rate", response_model=HeartbeatRateResponse)
def get_heartbeat_rate(
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin view: the effective heartbeat beat rate (minutes) and its source."""
    return _heartbeat_rate_snapshot(request)


@router.put("/admin/settings/heartbeat-rate", response_model=HeartbeatRateResponse)
def update_heartbeat_rate(
    payload: HeartbeatRateUpdate,
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin: set the global heartbeat beat rate (minutes).

    The LangGraph-embedded scheduler picks up the change within ~30s (control poll).
    """
    store = _get_global_settings_store(request)
    store.set_setting(HEARTBEAT_RATE_SETTING_KEY, int(payload.heartbeat_rate_minutes))
    return _heartbeat_rate_snapshot(request)


# --- Heartbeat inspector (admin) ---------------------------------------------

MAX_HEARTBEAT_RUNS = 200


MIN_HEARTBEAT_COOLDOWN_SECONDS = 0
MAX_HEARTBEAT_COOLDOWN_SECONDS = 604800  # 7 days


class HeartbeatTaskInfo(BaseModel):
    name: str
    type: str
    scope: str  # "global" | "per_user"
    cooldown_seconds: int          # effective (override if set, else config default)
    cooldown_default: int          # configured default (core.yaml)
    cooldown_source: str           # "override" | "default"
    enabled: bool
    last_run: Optional[Dict[str, Any]] = None


class HeartbeatCooldownUpdate(BaseModel):
    cooldown_seconds: int = Field(
        ..., ge=MIN_HEARTBEAT_COOLDOWN_SECONDS, le=MAX_HEARTBEAT_COOLDOWN_SECONDS
    )


class HeartbeatTasksResponse(BaseModel):
    tasks: List[HeartbeatTaskInfo]


class HeartbeatRun(BaseModel):
    task_name: str
    user_id: Optional[str] = None
    status: str
    duration_ms: int
    fired_at: Optional[str] = None
    detail: Dict[str, Any] = {}


class HeartbeatRunsResponse(BaseModel):
    runs: List[HeartbeatRun]


def _get_heartbeat_run_store(request: Request) -> Optional[HeartbeatRunStore]:
    db_pool = getattr(request.app.state, "db_pool", None)
    if db_pool is None:
        return None
    try:
        return HeartbeatRunStore(db_pool)
    except Exception:  # pragma: no cover - best-effort observability
        return None


def _serialize_run(row: Dict[str, Any]) -> Dict[str, Any]:
    fired_at = row.get("fired_at")
    return {
        "task_name": row.get("task_name"),
        "user_id": row.get("user_id"),
        "status": row.get("status"),
        "duration_ms": int(row.get("duration_ms") or 0),
        "fired_at": fired_at.isoformat() if hasattr(fired_at, "isoformat") else fired_at,
        "detail": row.get("detail") or {},
    }


@router.get("/admin/heartbeat/tasks", response_model=HeartbeatTasksResponse)
def list_heartbeat_tasks(
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin: the configured heartbeat tasks and each task's most recent run.

    ``cooldown_seconds`` is the *effective* value (an admin override from
    ``global_settings`` wins over the configured default).
    """
    hb = getattr(load_core_config(), "heartbeat", None)
    configured = list(hb.tasks) if hb is not None else []
    store = _get_heartbeat_run_store(request)
    overrides = _heartbeat_cooldown_overrides(request)
    tasks: List[Dict[str, Any]] = []
    for tcfg in configured:
        last_run = None
        if store is not None:
            try:
                rows = store.recent_runs_all(1, task_name=tcfg.name)
                last_run = _serialize_run(rows[0]) if rows else None
            except Exception:  # pragma: no cover - best-effort
                last_run = None
        default_cd = int(tcfg.cooldown_seconds)
        override_cd = overrides.get(tcfg.name)
        tasks.append(
            {
                "name": tcfg.name,
                "type": tcfg.type,
                "scope": tcfg.scope,
                "cooldown_seconds": override_cd if override_cd is not None else default_cd,
                "cooldown_default": default_cd,
                "cooldown_source": "override" if override_cd is not None else "default",
                "enabled": bool(tcfg.enabled),
                "last_run": last_run,
            }
        )
    return {"tasks": tasks}


def _heartbeat_cooldown_overrides(request: Request) -> Dict[str, int]:
    """Read + clamp the admin per-task cooldown override map from global_settings."""
    store = _get_global_settings_store(request)
    raw = store.get_setting(HEARTBEAT_COOLDOWNS_SETTING_KEY)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for name, value in raw.items():
        try:
            out[str(name)] = max(
                MIN_HEARTBEAT_COOLDOWN_SECONDS, min(MAX_HEARTBEAT_COOLDOWN_SECONDS, int(value))
            )
        except (TypeError, ValueError):
            continue
    return out


@router.put("/admin/heartbeat/tasks/{task_name}/cooldown", response_model=HeartbeatTasksResponse)
def update_heartbeat_cooldown(
    task_name: str,
    payload: HeartbeatCooldownUpdate,
    request: Request,
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin: override a heartbeat task's per-user cooldown (seconds), applied live
    by the scheduler within ~30s (control poll) — no rebuild. Returns the full task
    list so the UI can refresh in one round-trip."""
    hb = getattr(load_core_config(), "heartbeat", None)
    config_by_name = {t.name: int(t.cooldown_seconds) for t in (hb.tasks if hb is not None else [])}
    if task_name not in config_by_name:
        raise HTTPException(status_code=404, detail=f"Unknown heartbeat task '{task_name}'")

    store = _get_global_settings_store(request)
    overrides = _heartbeat_cooldown_overrides(request)
    # Setting a task back to its configured default clears the override (→ "default").
    if int(payload.cooldown_seconds) == config_by_name[task_name]:
        overrides.pop(task_name, None)
    else:
        overrides[task_name] = int(payload.cooldown_seconds)
    store.set_setting(HEARTBEAT_COOLDOWNS_SETTING_KEY, overrides)
    return list_heartbeat_tasks(request, _admin)


@router.get("/admin/heartbeat/runs", response_model=HeartbeatRunsResponse)
def list_heartbeat_runs(
    request: Request,
    task: Optional[str] = Query(default=None),
    user: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=MAX_HEARTBEAT_RUNS),
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin: the heartbeat run log, newest first, optionally filtered by task/user."""
    store = _get_heartbeat_run_store(request)
    if store is None:
        return {"runs": []}
    try:
        rows = store.recent_runs_all(int(limit), task_name=task, user_id=user)
    except Exception:  # pragma: no cover - best-effort observability
        return {"runs": []}
    return {"runs": [_serialize_run(row) for row in rows]}


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """Fetch available LLM models and return canonical provider-prefixed IDs."""
    try:
        core = load_core_config()
        models, _ = await _fetch_provider_models(core.llm.get_role_provider("chat"), preferred_language=core.profile.language)
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.get("/system-config", response_model=SystemConfigResponse)
async def get_system_config(request: Request) -> Dict[str, Any]:
    """Fetch system configuration: available tools, RAG defaults, default models."""

    try:
        profile_id = get_active_profile_id()
        core_config = load_core_config()
        profile_metadata = load_profile_metadata(profile_id=profile_id)
        profile_ui = load_profile_ui_config(profile_id=profile_id)

        available_tools = _load_tool_catalog()

        rag_cfg = core_config.rag
        collection_name = _resolve_env_placeholder(
            rag_cfg.collection_name if rag_cfg else None,
            default="",
        )
        top_k = rag_cfg.top_k if rag_cfg else 5

        chat_provider = core_config.llm.get_role_provider("chat")
        try:
            default_model = core_config.llm.get_role_model_name("chat", core_config.profile.language)
        except Exception:
            default_model = getattr(chat_provider.models, core_config.profile.language, None) or chat_provider.models.en or ""

        # Load probe results once — used to overlay context_window_tokens (probe fires at
        # startup and on model change, so it captures the value the server is loaded with).
        probe_store = _get_probe_store(request)
        probe_ctx_by_model: Dict[str, int] = {}
        if probe_store:
            try:
                for row in probe_store.list_probe_results(profile_id=profile_id, limit=200):
                    mn = str(row.get("model_name") or "")
                    ctx = (row.get("metadata") or {}).get("context_window_tokens")
                    if mn and ctx:
                        try:
                            probe_ctx_by_model[normalize_model_id(mn)] = int(ctx)
                        except Exception:
                            pass
            except Exception:
                pass

        model_roles: List[Dict[str, Any]] = []

        language = core_config.profile.language
        for role_name in ("chat", "embedding", "vision", "auxiliary"):
            try:
                role_chain = core_config.llm.get_role_provider_chain_with_models(role_name, language)
            except Exception:
                continue

            for provider_id, provider, role_default_model in role_chain:
                available_models: List[str] = []
                context_windows: Dict[str, int] = {}
                model_load_error: Optional[str] = None
                try:
                    available_models, context_windows = await _fetch_provider_models(provider, preferred_language=language)
                except Exception as role_exc:
                    model_load_error = str(role_exc)

                role_settings = getattr(core_config.llm.roles, role_name, None)
                # Prefer probe store (captures the value the server is actually loaded with,
                # updated on startup and model change) over the /models endpoint value.
                probe_ctx = None
                if role_default_model:
                    try:
                        probe_ctx = probe_ctx_by_model.get(normalize_model_id(str(role_default_model)))
                    except Exception:
                        pass
                # Explicit profile override wins over runtime auto-detection (probe / /models),
                # so the context-window indicator stays correct even when the model isn't loaded
                # at probe time or the provider doesn't report context length.
                config_ctx = getattr(role_settings, "context_window_tokens", None)
                context_window_tokens = config_ctx or probe_ctx or context_windows.get(str(role_default_model)) or None
                # Per-model context windows so the UI ring's denominator tracks the
                # model the user actually selected (not just the role default). Start
                # from the provider /models map, then overlay probe values (captured at
                # the size the server loaded the model with).
                per_model_windows: Dict[str, int] = {}
                for mname, mctx in context_windows.items():
                    try:
                        per_model_windows[str(mname)] = int(mctx)
                    except Exception:
                        continue
                for mname in available_models:
                    try:
                        pctx = probe_ctx_by_model.get(normalize_model_id(str(mname)))
                    except Exception:
                        pctx = None
                    if pctx:
                        per_model_windows[str(mname)] = int(pctx)
                # The explicit profile override applies to the role default model.
                if config_ctx and role_default_model:
                    per_model_windows[str(role_default_model)] = int(config_ctx)
                model_roles.append(
                    {
                        "role": role_name,
                        "provider_id": str(provider_id),
                        "default_model": role_default_model,
                        "available_models": available_models,
                        "model_load_error": model_load_error,
                        "max_tokens": getattr(role_settings, "max_tokens", None),
                        "context_window_tokens": context_window_tokens,
                        "context_windows": per_model_windows,
                    }
                )
        display_name = profile_metadata.display_name if profile_metadata else "Base Profile"
        role_label = profile_ui.branding.role_label or display_name or os.getenv("NEXT_PUBLIC_SINGLE_USER_ROLE_LABEL", "Local Profile")
        app_name = profile_ui.branding.app_name or os.getenv("NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME", "Single User")

        supported_languages = core_config.profile.supported_languages
        if not supported_languages:
            # Derive from prompt files if not explicitly configured
            prompts_cfg = core_config.prompts
            if prompts_cfg and prompts_cfg.languages:
                supported_languages = sorted(prompts_cfg.languages.keys())
            else:
                supported_languages = [core_config.profile.language]

        return {
            "available_tools": available_tools,
            "rag_defaults": {
                "collection_name": collection_name,
                "top_k": int(top_k) if isinstance(top_k, (int, float, str)) else 5,
            },
            "default_model": default_model,
            "framework_version": get_framework_version(),
            "supported_languages": supported_languages,
            "model_roles": model_roles,
            "profile": {
                "id": profile_id,
                "display_name": display_name,
                "role_label": role_label,
                "description": profile_metadata.description if profile_metadata else None,
                "app_name": app_name,
                "theme": {
                    "colors": profile_ui.theme.colors,
                    "fonts": profile_ui.theme.fonts,
                    "radius": profile_ui.theme.radius,
                    "custom_css_vars": profile_ui.theme.custom_css_vars,
                },
            },
        }
    except Exception as e:
        # Fallback defaults if config loading fails
        return {
            "available_tools": [
                {"id": "web_search_mcp", "label": "Web Search"},
                {"id": "extract_webpage_mcp", "label": "Extract Webpage"},
                {"id": "analyze_image_tool", "label": "Analyze Image"},
                {"id": "ocr_tool", "label": "OCR"},
                {"id": "analyze_document_tool", "label": "Analyze Document"},
                {"id": "analyze_chart_tool", "label": "Analyze Chart"},
                {"id": "image_metadata_tool", "label": "Image Metadata"},
                {"id": "read_barcodes_tool", "label": "Read Barcodes"},
                {"id": "datetime_tool", "label": "Datetime"},
                {"id": "calculator_tool", "label": "Calculator"},
                {"id": "map_tool", "label": "Map"},
                {"id": "weather_tool", "label": "Weather"},
                {"id": "file_ops_tool", "label": "File Operations"},
            ],
            "rag_defaults": {"collection_name": "framework", "top_k": 5},
            "default_model": "",
            "framework_version": get_framework_version(),
            "supported_languages": ["en"],
            "model_roles": [],
            "profile": {
                "id": "base",
                "display_name": "Base Profile",
                "role_label": os.getenv("NEXT_PUBLIC_SINGLE_USER_ROLE_LABEL", "Local Profile"),
                "description": None,
                "app_name": os.getenv("NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME", "Single User"),
                "theme": {"colors": {}, "fonts": {}, "radius": {}, "custom_css_vars": {}},
            },
            "error": str(e),
        }


@router.post("/llm/reprobe")
def trigger_llm_reprobe(request: Request) -> Dict[str, Any]:
    """Manually trigger LLM capability reprobe for all configured providers."""
    probe_store = _get_probe_store(request)
    if not probe_store:
        raise HTTPException(status_code=503, detail="Probe store unavailable")
    
    try:
        profile_id = get_active_profile_id()
        runner = LLMCapabilityProbeRunner(profile_id=profile_id)
        
        # Run full reprobe
        results = runner.run()
        
        # Persist all reprobe results
        for result in results:
            probe_store.upsert_probe_result(result)
        
        logger.info(
            "Manual LLM reprobe triggered",
            extra={"profile_id": profile_id, "reprobe_count": len(results)},
        )
        
        return {
            "status": "ok",
            "profile_id": profile_id,
            "reprobe_count": len(results),
            "results": [
                {
                    "provider_id": r.provider_id,
                    "model_name": r.model_name,
                    "status": r.status,
                    "capability_mismatch": r.capability_mismatch,
                    "supports_bind_tools": r.supports_bind_tools,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }
    except Exception as exc:
        logger.error("Manual reprobe failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Reprobe failed: {exc}") from exc


@router.get("/llm/capabilities")
def list_llm_capabilities(request: Request) -> Dict[str, Any]:
    """Expose model-level desired/effective tool-calling capability view across all roles."""
    probe_store = _get_probe_store(request)
    if not probe_store:
        raise HTTPException(status_code=503, detail="Probe store unavailable")

    profile_id = get_active_profile_id()
    core = load_core_config()
    
    probe_rows = probe_store.list_probe_results(profile_id=profile_id, limit=500)
    probe_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in probe_rows:
        key = (str(row.get("provider_id") or ""), normalize_model_id(str(row.get("model_name") or "")))
        if key[0] and key[1] and key not in probe_by_key:
            probe_by_key[key] = row

    items: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # (provider_id, model_name, role)

    # Iterate through roles (chat first for primary visibility, then vision, auxiliary)
    for role_name in ["chat", "vision", "auxiliary"]:
        try:
            role_chain = core.llm.get_role_provider_chain_with_models(role_name, core.profile.language)
        except Exception:
            continue

        for provider_id, provider, resolved_model_name in role_chain:
            model_name = normalize_model_id(str(resolved_model_name))
            item_key = (str(provider_id), model_name, role_name)

            if item_key in seen:
                continue
            seen.add(item_key)

            desired_mode = str(provider.get_tool_calling_mode(model_name))
            probe_row = probe_by_key.get((str(provider_id), model_name))
            effective_mode, reason = _compute_effective_mode(desired_mode, probe_row)
            metadata = probe_row.get("metadata") if probe_row else {}
            capabilities = metadata.get("capabilities") if isinstance(metadata, dict) else {}

            items.append(
                {
                    "provider_id": str(provider_id),
                    "model_name": model_name,
                    "role": role_name,
                    "desired_mode": desired_mode,
                    "configured_tool_calling_mode": probe_row.get("configured_tool_calling_mode") if probe_row else desired_mode,
                    "effective_mode": effective_mode,
                    "effective_mode_reason": reason,
                    "probe_status": probe_row.get("status") if probe_row else "missing",
                    "capability_mismatch": bool(probe_row.get("capability_mismatch", False)) if probe_row else False,
                    "supports_bind_tools": probe_row.get("supports_bind_tools") if probe_row else None,
                    "supports_tool_schema": probe_row.get("supports_tool_schema") if probe_row else None,
                    "api_base": probe_row.get("api_base") if probe_row else None,
                    "error_message": probe_row.get("error_message") if probe_row else None,
                    "probed_at": probe_row.get("probed_at") if probe_row else None,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "capabilities": capabilities if isinstance(capabilities, dict) else {},
                    "supports_reasoning": bool(metadata.get("supports_reasoning", False)) if isinstance(metadata, dict) else False,
                    "supports_vision": metadata.get("supports_vision") if isinstance(metadata, dict) else None,
                }
            )

    return {
        "status": "ok",
        "profile_id": profile_id,
        "probe_ttl_seconds": int(os.getenv("LLM_CAPABILITY_PROBE_TTL_SECONDS", "3600")),
        "items": items,
    }


def _resolve_health_targets(core: Any) -> Tuple[Dict[str, List[str]], Optional[str]]:
    """Return (api_base -> sorted roles, chat_api_base).

    Collects the distinct provider endpoints across all configured roles so the
    health check pings each URL only once. Unconfigured optional roles
    (vision/auxiliary) raise and are skipped; the embedding endpoint is resolved
    via its dedicated helper.
    """
    targets: Dict[str, List[str]] = {}
    chat_base: Optional[str] = None

    for role_name in ("chat", "vision", "auxiliary"):
        try:
            base = str(getattr(core.llm.get_role_provider(role_name), "api_base", "") or "").rstrip("/")
        except Exception:
            continue  # role not configured
        if not base:
            continue
        targets.setdefault(base, []).append(role_name)
        if role_name == "chat":
            chat_base = base

    try:
        embed_base = str(core.llm.get_embedding_remote_endpoint() or "").rstrip("/")
    except Exception:
        embed_base = ""
    if embed_base:
        targets.setdefault(embed_base, []).append("embedding")

    return {base: sorted(roles) for base, roles in targets.items()}, chat_base


async def _ping_provider(client: httpx.AsyncClient, base: str) -> Tuple[bool, str]:
    """Lightweight reachability probe: any HTTP response means the server is up."""
    try:
        resp = await client.get(f"{base}/models")
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:  # connection refused / timeout / DNS → unreachable
        return False, type(exc).__name__


@router.get("/llm/health")
async def provider_health() -> Dict[str, Any]:
    """Live reachability of the configured LLM provider endpoint(s).

    Pings each distinct ``api_base`` (chat/embedding/vision/auxiliary, deduped) in
    parallel with a short timeout. Used by the frontend to drive the
    "Provider Offline" banner. Lightweight and live, unlike ``/llm/capabilities``
    (a stale DB read of the last capability probe).
    """
    try:
        core = load_core_config()
        targets, chat_base = _resolve_health_targets(core)
    except Exception as exc:
        logger.warning("Provider health: failed to resolve targets: %s", exc)
        return {
            "status": "offline",
            "providers": [],
            "error": str(exc),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    if not targets:
        return {
            "status": "offline",
            "providers": [],
            "error": "no_api_base_configured",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    bases = list(targets.keys())
    async with httpx.AsyncClient(timeout=3.0) as client:
        results = await asyncio.gather(*[_ping_provider(client, base) for base in bases])

    providers: List[Dict[str, Any]] = []
    chat_reachable = True  # if chat role is unconfigured, don't fail on its account
    any_unreachable = False
    for base, (reachable, detail) in zip(bases, results):
        providers.append(
            {"roles": targets[base], "api_base": base, "reachable": reachable, "detail": detail}
        )
        if not reachable:
            any_unreachable = True
        if base == chat_base:
            chat_reachable = reachable

    if not chat_reachable:
        status_value = "offline"
    elif any_unreachable:
        status_value = "degraded"
    else:
        status_value = "online"

    return {
        "status": status_value,
        "providers": providers,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/ingestion/reingest-all", response_model=ReingestAllResponse)
def trigger_reingest_all_documents() -> Dict[str, Any]:
    """Clear and re-index all documents from the configured ingestion source path."""
    try:
        defaults = resolve_runtime_ingestion_defaults()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to resolve ingestion config: {exc}") from exc

    source = defaults.source_path
    collection = defaults.collection_name
    language = defaults.language
    timeout_seconds = defaults.reingest_timeout_seconds

    source_path = Path(source)
    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ingestion source path does not exist: {source}. "
                "Set core.ingestion.source_path to a readable document directory."
            ),
        )

    command = [
        sys.executable,
        "-m",
        "universal_agentic_framework.cli.ingest",
        "reindex",
        "--source",
        source,
        "--collection",
        collection,
        "--language",
        language,
        "--yes",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Re-ingestion timed out. "
                f"Increase INGEST_REINGEST_TIMEOUT_SECONDS if needed. ({exc})"
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to start re-ingestion: {exc}") from exc

    combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    output_tail = _tail_lines(combined_output, max_lines=30)

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Re-ingestion failed",
                "exit_code": result.returncode,
                "output_tail": output_tail,
            },
        )

    return {
        "status": "ok",
        "source": source,
        "collection": collection,
        "language": language,
        "processed": _extract_cli_metric(result.stdout, "Files processed"),
        "skipped": _extract_cli_metric(result.stdout, "Files skipped"),
        "errors": _extract_cli_metric(result.stdout, "Files with errors"),
        "total_chunks": _extract_cli_metric(result.stdout, "Total chunks"),
        "output_tail": output_tail,
    }



class ResetOptions(BaseModel):
    """Selects which data categories to purge. All default to True."""
    conversations: bool = True
    workspace: bool = True
    memories: bool = True
    analytics: bool = True
    llm_probes: bool = True


class UserResetOptions(BaseModel):
    """Selects which personal data categories to purge for the current user only."""
    conversations: bool = True
    workspace: bool = True
    memories: bool = True


_CONVERSATION_TABLES = [
    "conversation_attachments",
    "messages",
    "conversations",
]

_WORKSPACE_TABLES = [
    "workspace_document_versions",
    "workspace_documents",
    "chat_document_refs",
    "conversation_workspace_operations",
    "conversation_workspaces",
]

_MEMORY_TABLES = [
    "co_occurrence_edges",
]

_ANALYTICS_TABLES = [
    "analytics_events",
    "daily_analytics_stats",
]

_LLM_PROBE_TABLES = [
    "llm_capability_probes",
]


@router.post("/admin/reset-all-databases")
def reset_all_databases(
    request: Request,
    options: Optional[ResetOptions] = Body(default=None),
    _admin: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    """Purge selected data categories across all users.

    Accepts an optional JSON body with boolean flags per category — all True by
    default so a no-body call still purges everything. The schema is preserved
    so the service continues running without a restart.
    """
    if options is None:
        options = ResetOptions()
    import shutil

    results: Dict[str, Any] = {
        "postgres": {"status": "ok", "tables_truncated": []},
        "qdrant": {"status": "ok", "collections_deleted": []},
        "workspace_files": {"status": "ok", "bytes_freed": 0},
    }
    errors: list[str] = []

    # Build the list of tables to truncate based on selected options
    tables_to_truncate: list[str] = []
    if options.conversations:
        tables_to_truncate.extend(_CONVERSATION_TABLES)
    if options.workspace:
        tables_to_truncate.extend(_WORKSPACE_TABLES)
    if options.memories:
        tables_to_truncate.extend(_MEMORY_TABLES)
    if options.analytics:
        tables_to_truncate.extend(_ANALYTICS_TABLES)
    if options.llm_probes:
        tables_to_truncate.extend(_LLM_PROBE_TABLES)

    # ── 1. Postgres ───────────────────────────────────────────────────────────
    # user_settings intentionally excluded so preferences survive a reset.
    if tables_to_truncate:
        db_pool = getattr(request.app.state, "db_pool", None)
        if db_pool is None:
            errors.append("postgres: db_pool unavailable")
            results["postgres"]["status"] = "error"
        else:
            try:
                with db_pool.connection() as conn:
                    with conn.cursor() as cur:
                        for table in tables_to_truncate:
                            cur.execute(f"TRUNCATE TABLE {table} CASCADE;")  # noqa: S608
                            results["postgres"]["tables_truncated"].append(table)
                    conn.commit()
            except Exception as exc:
                errors.append(f"postgres: {exc}")
                results["postgres"]["status"] = "error"

    # ── 2. Qdrant (memories + knowledge base) ─────────────────────────────────
    if options.memories:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]

            qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            qc = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
            collections = [c.name for c in qc.get_collections().collections]
            for name in collections:
                qc.delete_collection(name)
                results["qdrant"]["collections_deleted"].append(name)
        except Exception as exc:
            errors.append(f"qdrant: {exc}")
            results["qdrant"]["status"] = "error"

    # ── 3. Workspace files on disk ────────────────────────────────────────────
    if options.workspace:
        try:
            from backend.attachments import WorkspaceManagerConfig, AttachmentManagerConfig

            ws_config = WorkspaceManagerConfig.from_env()
            attach_config = AttachmentManagerConfig.from_env()
            freed = 0
            for root in {ws_config.root_dir, attach_config.root_dir}:
                if root.exists():
                    freed += sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
                    shutil.rmtree(root, ignore_errors=True)
                    root.mkdir(parents=True, exist_ok=True)
            results["workspace_files"]["bytes_freed"] = freed
        except Exception as exc:
            errors.append(f"workspace_files: {exc}")
            results["workspace_files"]["status"] = "error"

    if errors:
        logger.warning("reset_all_databases completed with errors: %s", errors)
        return {"status": "partial", "errors": errors, **results}

    logger.info("reset_all_databases completed successfully")
    return {"status": "ok", "errors": [], **results}


@router.post("/user/reset-my-data")
def reset_my_data(
    request: Request,
    options: Optional[UserResetOptions] = Body(default=None),
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Purge selected personal data for the current authenticated user only.

    Only deletes rows owned by this user — never touches other users' data or
    system-wide tables (analytics, LLM probes, RAG knowledge base).
    """
    if options is None:
        options = UserResetOptions()
    import shutil

    errors: list[str] = []
    results: Dict[str, Any] = {
        "postgres": {"status": "ok", "deleted": {}},
        "qdrant": {"status": "ok", "memories_deleted": False},
        "files": {"status": "ok", "bytes_freed": 0},
    }

    db_pool = getattr(request.app.state, "db_pool", None)

    # Collect conversation IDs before deletion so we can clean up files afterwards.
    conversation_ids: list[str] = []
    if options.conversations and db_pool:
        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM conversations WHERE user_id = %s;", (user_id,))
                    conversation_ids = [row[0] for row in cur.fetchall()]
        except Exception as exc:
            errors.append(f"postgres: failed to list conversations: {exc}")

    # ── 1. Postgres — user-scoped DELETEs ─────────────────────────────────────
    if db_pool is None:
        errors.append("postgres: db_pool unavailable")
        results["postgres"]["status"] = "error"
    else:
        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    if options.conversations:
                        cur.execute(
                            "DELETE FROM conversations WHERE user_id = %s;",  # noqa: S608
                            (user_id,),
                        )
                        results["postgres"]["deleted"]["conversations"] = cur.rowcount
                    if options.workspace:
                        cur.execute(
                            "DELETE FROM workspace_documents WHERE user_id = %s;",  # noqa: S608
                            (user_id,),
                        )
                        results["postgres"]["deleted"]["workspace_documents"] = cur.rowcount
                conn.commit()
        except Exception as exc:
            errors.append(f"postgres: {exc}")
            results["postgres"]["status"] = "error"

    # ── 2. Memories: Qdrant vectors + co_occurrence_edges for this user ──────
    if options.memories:
        # Co-occurrence edges are stored in Postgres and scoped by user_id.
        if db_pool is not None:
            try:
                with db_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM co_occurrence_edges WHERE user_id = %s;",  # noqa: S608
                            (user_id,),
                        )
                    conn.commit()
            except Exception as exc:
                errors.append(f"postgres co_occurrence_edges: {exc}")

        try:
            from universal_agentic_framework.memory.factory import build_memory_backend as _build_mb

            cfg = load_core_config()
            backend = _build_mb(cfg)
            if hasattr(backend, "_delete_all_memories"):
                backend._delete_all_memories(user_id=user_id)
                results["qdrant"]["memories_deleted"] = True
            else:
                errors.append("qdrant: memory backend does not support bulk delete")
                results["qdrant"]["status"] = "error"
        except Exception as exc:
            errors.append(f"qdrant: {exc}")
            results["qdrant"]["status"] = "error"

    # ── 3. Files ───────────────────────────────────────────────────────────────
    try:
        from backend.attachments import AttachmentManagerConfig, WorkspaceManagerConfig

        freed = 0

        if options.conversations and conversation_ids:
            attach_config = AttachmentManagerConfig.from_env()
            for conv_id in conversation_ids:
                try:
                    conv_dir = attach_config.root_dir / conv_id
                    if conv_dir.exists():
                        freed += sum(f.stat().st_size for f in conv_dir.rglob("*") if f.is_file())
                        shutil.rmtree(conv_dir, ignore_errors=True)
                except Exception:
                    pass

        if options.workspace:
            ws_config = WorkspaceManagerConfig.from_env()
            user_ws_dir = ws_config.root_dir / "user-workspaces" / user_id
            if user_ws_dir.exists():
                freed += sum(f.stat().st_size for f in user_ws_dir.rglob("*") if f.is_file())
                shutil.rmtree(user_ws_dir, ignore_errors=True)

        results["files"]["bytes_freed"] = freed
    except Exception as exc:
        errors.append(f"files: {exc}")
        results["files"]["status"] = "error"

    if errors:
        logger.warning(
            "reset_my_data completed with errors: %s (user=%s)",
            errors,
            user_id,
        )
        return {"status": "partial", "errors": errors, **results}

    logger.info("reset_my_data completed (user=%s, options=%s)", user_id, options.model_dump())
    return {"status": "ok", "errors": [], **results}


_ACRONYMS = {"ocr", "mcp", "rag", "ai", "llm"}

def _format_tool_name(name: str) -> str:
    """Convert tool name from snake_case to Title Case, preserving known acronyms."""
    pretty = name.replace("_tool", "").replace("_mcp", "").replace("_", " ").strip()
    return " ".join(
        word.upper() if word.lower() in _ACRONYMS else word.capitalize()
        for word in pretty.split()
    )


# The three UI tool groups (also the column order on the admin + settings pages).
TOOL_GROUPS = ("text", "vision", "auxiliary")


def _repo_root() -> Path:
    """Repository root, used to resolve relative tool paths from tools.yaml."""
    return Path(__file__).resolve().parents[2]


def _normalize_tool_group(category: Optional[str]) -> str:
    """Map a manifest ``category`` to one of the three UI tool groups.

    Unknown/missing categories fall back to ``auxiliary`` so a tool always lands
    in a column rather than disappearing.
    """
    cat = (category or "").strip().lower()
    return cat if cat in TOOL_GROUPS else "auxiliary"


def _resolve_tool_manifest_category(tool_path: str) -> Optional[str]:
    """Best-effort read of the ``category`` field from a tool's ``tool.yaml``."""
    if not tool_path:
        return None
    path = Path(tool_path)
    if not path.is_absolute():
        path = _repo_root() / tool_path
    category = _load_yaml_file(path / "tool.yaml").get("category")
    return category if isinstance(category, str) else None


def _load_tool_catalog() -> List[Dict[str, str]]:
    """Return the full tool catalog — ``{id, label, group}`` for every configured tool.

    Single source of truth for the tool list, reused by ``/system-config``, the
    admin role-tools endpoint, ``/settings/me``, and startup seeding. ``group`` is
    derived from each tool's manifest ``category`` (text | vision | auxiliary).
    """
    catalog: List[Dict[str, str]] = []
    for tool in load_tools_config().tools:
        tool_name = getattr(tool, "name", None)
        if not tool_name:
            continue
        group = _normalize_tool_group(_resolve_tool_manifest_category(getattr(tool, "path", "") or ""))
        catalog.append({"id": tool_name, "label": _format_tool_name(tool_name), "group": group})
    return catalog


def _catalog_tool_ids() -> List[str]:
    """All tool ids in the catalog (used for admin validation + seeding/admin defaults)."""
    return [item["id"] for item in _load_tool_catalog()]


def _resolve_env_placeholder(value: Any, default: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        env_name = value[1:]
        return os.getenv(env_name, default)
    if isinstance(value, str) and value:
        return value
    return default


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _extract_cli_metric(output: str, label: str) -> int:
    pattern = rf"^{re.escape(label)}:\s*(\d+)"
    for line in output.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            return int(match.group(1))
    return 0


def _tail_lines(text: str, max_lines: int = 30) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])

