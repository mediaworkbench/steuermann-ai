from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.db import SettingsStore
from backend.single_user import get_effective_user_id, require_api_access
from backend.version import get_framework_version
from universal_agentic_framework.config import (
    get_active_profile_id,
    load_core_config,
    load_profile_metadata,
    load_profile_ui_config,
    load_tools_config,
)


router = APIRouter(prefix="/api", tags=["settings"], dependencies=[Depends(require_api_access)])

# OpenAI-compatible endpoint: works with LM Studio and Ollama (/v1)
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434")


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
    theme: str = Field(default="auto", description="light, dark, or auto")
    language: str = Field(default="en")


class UserSettingsResponse(BaseModel):
    user_id: str
    tool_toggles: Dict[str, bool]
    rag_config: Dict[str, Any]
    analytics_preferences: Dict[str, Any]
    preferred_model: Optional[str]
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
            "theme": "auto",
            "language": "en",
            "updated_at": None,
        }
    return record


@router.post("/settings/user/{user_id}", response_model=UserSettingsResponse)
def update_user_settings(user_id: str, settings: UserSettings, request: Request) -> Dict[str, Any]:
    """Persist user settings for the given user id."""
    effective_user_id = get_effective_user_id(user_id)
    store = _get_settings_store(request)
    record = store.upsert_user_settings(
        user_id=effective_user_id,
        tool_toggles=settings.tool_toggles,
        rag_config=settings.rag_config,
        analytics_preferences=settings.analytics_preferences,
        preferred_model=settings.preferred_model,
        theme=settings.theme,
        language=settings.language,
    )
    # Clear settings cache for this user to force reload on next chat request
    from backend.routers.chat import _settings_cache
    if effective_user_id in _settings_cache:
        del _settings_cache[effective_user_id]
    return record


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """Fetch available LLM models from OpenAI-compatible endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = str(LLM_ENDPOINT).rstrip("/")
            resp = await client.get(f"{base}/models")
            resp.raise_for_status()
            data = resp.json()
            model_names = [m.get("id") for m in data.get("data", []) if m.get("id")]
            # Filter out embedding-only models (e.g. bge-m3)
            model_names = [
                n for n in model_names
                if not any(skip in n.lower() for skip in ("bge-", "nomic-embed", "embed"))
            ]
            return {"models": sorted(model_names)}
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
            rag_cfg.collection_name if rag_cfg else "$INGEST_COLLECTION",
            default="framework",
        )
        top_k = rag_cfg.top_k if rag_cfg else 5

        default_model = core_config.llm.providers.primary.models.en or "gemma3:4b"
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
            "default_model": "gemma3:4b",
            "framework_version": get_framework_version(),
            "supported_languages": ["en"],
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


@router.post("/ingestion/reingest-all", response_model=ReingestAllResponse)
def trigger_reingest_all_documents() -> Dict[str, Any]:
    """Clear and re-index all documents from the configured ingestion source path."""
    source = os.getenv("RAG_DATA_PATH", "/data/rag-data")
    collection = os.getenv("INGEST_COLLECTION", "framework")
    language = os.getenv("INGEST_LANGUAGE", "en")
    timeout_seconds = int(os.getenv("INGEST_REINGEST_TIMEOUT_SECONDS", "1800"))

    source_path = Path(source)
    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ingestion source path does not exist: {source}. "
                "Set RAG_DATA_PATH to a readable document directory."
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

