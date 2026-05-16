from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.db import SettingsStore, LLMCapabilityProbeStore
from backend.llm_capability_probe import LLMCapabilityProbeRunner
from backend.single_user import get_effective_user_id, require_api_access
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
    rag_config: Dict[str, Any] = Field(default_factory=lambda: {"collection": "", "top_k": 5})
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


async def _fetch_provider_models(provider: Any, preferred_language: str = "en") -> List[str]:
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
            return []
        try:
            resp = await client.get(f"{base}/models")
            resp.raise_for_status()
            data = resp.json()
            model_names = [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            if provider_prefix != "ollama":
                raise
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            model_names = [m.get("name") for m in data.get("models", []) if m.get("name")]

    canonical_names = [_canonical_model_name(n) for n in model_names]
    canonical_names = [
        n for n in canonical_names
        if not any(skip in n.lower() for skip in ("bge-", "nomic-embed", "embed"))
    ]
    return sorted(set(canonical_names))


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


@router.get("/settings/user/{user_id}", response_model=UserSettingsResponse)
def get_user_settings(user_id: str, request: Request) -> Dict[str, Any]:
    """Retrieve persisted user settings or return defaults."""
    effective_user_id = get_effective_user_id(user_id)
    store = _get_settings_store(request)
    record = store.get_user_settings(effective_user_id)
    if record is None:
        return {
            "user_id": effective_user_id,
            "tool_toggles": {},
            "rag_config": {"collection": "", "top_k": 5},
            "analytics_preferences": {},
            "preferred_model": None,
            "preferred_models": {},
            "theme": "auto",
            "language": "en",
            "updated_at": None,
        }
    return record


@router.post("/settings/user/{user_id}", response_model=UserSettingsResponse)
async def update_user_settings(user_id: str, settings: UserSettings, request: Request) -> Dict[str, Any]:
    """Persist user settings for the given user id."""
    effective_user_id = get_effective_user_id(user_id)
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
    rag_config = _resolved_value("rag_config", {"collection": "", "top_k": 5})
    analytics_preferences = _resolved_value("analytics_preferences", {})
    theme = _resolved_value("theme", "auto")
    language = _resolved_value("language", "en")

    has_preferred_model_update = "preferred_model" in fields_set
    has_preferred_models_update = "preferred_models" in fields_set

    preferred_model = _normalize_preferred_model_value(_resolved_value("preferred_model", None))

    raw_preferred_models = _resolved_value("preferred_models", {}) or {}
    if not has_preferred_models_update and has_preferred_model_update:
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

    should_validate_chat_preference = has_preferred_model_update or (
        has_preferred_models_update and "chat" in raw_preferred_models
    )

    if should_validate_chat_preference and preferred_model:
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
    
    return record


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """Fetch available LLM models and return canonical provider-prefixed IDs."""
    try:
        core = load_core_config()
        models = await _fetch_provider_models(core.llm.get_role_provider("chat"), preferred_language=core.fork.language)
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.get("/system-config", response_model=SystemConfigResponse)
async def get_system_config() -> Dict[str, Any]:
    """Fetch system configuration: available tools, RAG defaults, default models."""

    try:
        profile_id = get_active_profile_id()
        core_config = load_core_config()
        tools_config = load_tools_config()
        profile_metadata = load_profile_metadata(profile_id=profile_id)
        profile_ui = load_profile_ui_config(profile_id=profile_id)

        available_tools = []
        for tool in tools_config.tools:
            if tool.enabled:
                tool_name = tool.name
                if tool_name:
                    available_tools.append({"id": tool_name, "label": _format_tool_name(tool_name)})

        rag_cfg = core_config.rag
        collection_name = _resolve_env_placeholder(
            rag_cfg.collection_name if rag_cfg else None,
            default="",
        )
        top_k = rag_cfg.top_k if rag_cfg else 5

        chat_provider = core_config.llm.get_role_provider("chat")
        try:
            default_model = core_config.llm.get_role_model_name("chat", core_config.fork.language)
        except Exception:
            default_model = getattr(chat_provider.models, core_config.fork.language, None) or chat_provider.models.en or ""
        model_roles: List[Dict[str, Any]] = []

        language = core_config.fork.language
        for role_name in ("chat", "embedding", "vision", "auxiliary"):
            try:
                role_chain = core_config.llm.get_role_provider_chain_with_models(role_name, language)
            except Exception:
                continue

            for provider_id, provider, role_default_model in role_chain:
                available_models: List[str] = []
                model_load_error: Optional[str] = None
                try:
                    available_models = await _fetch_provider_models(provider, preferred_language=language)
                except Exception as role_exc:
                    model_load_error = str(role_exc)

                model_roles.append(
                    {
                        "role": role_name,
                        "provider_id": str(provider_id),
                        "default_model": role_default_model,
                        "available_models": available_models,
                        "model_load_error": model_load_error,
                    }
                )
        display_name = profile_metadata.display_name if profile_metadata else "Base Profile"
        role_label = profile_ui.branding.role_label or display_name or os.getenv("NEXT_PUBLIC_SINGLE_USER_ROLE_LABEL", "Local Profile")
        app_name = profile_ui.branding.app_name or os.getenv("NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME", "Single User")

        supported_languages = core_config.fork.supported_languages
        if not supported_languages:
            # Derive from prompt files if not explicitly configured
            prompts_cfg = core_config.prompts
            if prompts_cfg and prompts_cfg.languages:
                supported_languages = sorted(prompts_cfg.languages.keys())
            else:
                supported_languages = [core_config.fork.language]

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
                {"id": "datetime_tool", "label": "Datetime Tool"},
                {"id": "extract_webpage_mcp", "label": "Extract Webpage"},
                {"id": "calculator_tool", "label": "Calculator"},
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
            role_chain = core.llm.get_role_provider_chain_with_models(role_name, core.fork.language)
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
                }
            )

    return {
        "status": "ok",
        "profile_id": profile_id,
        "probe_ttl_seconds": int(os.getenv("LLM_CAPABILITY_PROBE_TTL_SECONDS", "3600")),
        "items": items,
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



def _format_tool_name(name: str) -> str:
    """Convert tool name from snake_case to Title Case."""
    pretty = name.replace("_tool", "").replace("_mcp", "").replace("_", " ").strip()
    return " ".join(word.capitalize() for word in pretty.split())


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

