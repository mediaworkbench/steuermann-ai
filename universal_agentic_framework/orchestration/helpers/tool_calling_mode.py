"""Tool-calling mode validation and resolution helpers."""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from universal_agentic_framework.llm.provider_registry import normalize_model_id

from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


def resolve_effective_tool_calling_mode(config: Any, state: Dict[str, Any]) -> Tuple[str, str]:
    """Resolve tool-calling mode with optional probe-aware downgrade.

    Curated downgrade: if configured mode is native but probe signals mismatch
    (bind_tools/tool schema failure), force structured mode.
    """
    language = state.get("language") or getattr(config.fork, "language", "en")
    provider_id = "primary"
    configured_mode = "structured"
    selected_model_name = None

    provider = None
    try:
        role_cfg = getattr(getattr(config.llm, "roles", None), "chat", None)
        role_providers = getattr(role_cfg, "providers", None)
        if role_providers:
            provider_id = str(role_providers[0].provider_id)
            providers = config.llm.providers
            registry_getter = getattr(providers, "get_registry", None)
            if callable(registry_getter):
                provider = registry_getter().get(provider_id)
    except Exception:
        provider = None

    if provider is None:
        provider = getattr(config.llm.providers, "primary", None)

    if provider is not None:
        models = getattr(provider, "models", None)
        if isinstance(models, dict):
            selected_model_name = models.get(language)
            if not selected_model_name:
                for model_value in models.values():
                    if model_value:
                        selected_model_name = model_value
                        break
        else:
            selected_model_name = getattr(models, language, None)
            if not selected_model_name and models is not None:
                if hasattr(models, "model_dump"):
                    model_values = list(models.model_dump().values())
                else:
                    model_values = list(vars(models).values())
                for model_value in model_values:
                    if model_value:
                        selected_model_name = model_value
                        break

        if selected_model_name:
            configured_mode = str(provider.get_tool_calling_mode(selected_model_name))

    if configured_mode != "native":
        return configured_mode, "model_config_non_native_mode"

    probe_rows = state.get("llm_capability_probes") or []
    if not probe_rows:
        return "structured", "probe_missing_forced_structured"

    matching_rows = [
        row for row in probe_rows
        if str(row.get("provider_id") or "") == provider_id
    ]
    if selected_model_name:
        normalized_selected = normalize_model_id(str(selected_model_name))
        model_specific = [
            row for row in matching_rows
            if normalize_model_id(str(row.get("model_name") or "")) == normalized_selected
        ]
        if model_specific:
            matching_rows = model_specific

    if not matching_rows:
        return "structured", "probe_model_not_found_forced_structured"

    probe = matching_rows[0]

    ttl_seconds = int(os.getenv("LLM_CAPABILITY_PROBE_TTL_SECONDS", "3600"))
    probed_at_raw = str(probe.get("probed_at") or "").strip()
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

    mismatch = bool(probe.get("capability_mismatch", False))
    bind_failed = probe.get("supports_bind_tools") is False
    schema_failed = probe.get("supports_tool_schema") is False
    status = str(probe.get("status") or "").lower()

    if mismatch or bind_failed or schema_failed or status in {"warning", "error"}:
        return "structured", "probe_capability_mismatch_downgrade"

    return configured_mode, "probe_confirmed_native"


def validate_and_log_tool_calling_mode(
    state: Dict[str, Any],
    expected_mode: str,
    node_name: str,
    fork_name: str,
) -> Tuple[bool, str]:
    """Validate that the resolved tool-calling mode matches expectations at invocation."""
    actual_mode = state.get("tool_calling_mode", "structured")
    mode_reason = state.get("tool_calling_mode_reason", "unknown")
    candidates = state.get("candidate_tools", [])

    # If no candidates, mode is irrelevant.
    if not candidates:
        logger.debug(
            f"Tool calling mode validation ({node_name}): no candidates, mode irrelevant",
            expected_mode=expected_mode,
            actual_mode=actual_mode,
        )
        return True, "no_candidates"

    is_valid = actual_mode == expected_mode
    log_fn = logger.info if is_valid else logger.warning

    log_fn(
        f"Tool calling mode validation ({node_name})",
        extra={
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
            "mode_reason": mode_reason,
            "is_valid": is_valid,
            "candidates": len(candidates),
            "fork_name": fork_name,
        },
    )

    if is_valid:
        return True, f"{actual_mode}_mode_valid"
    return False, f"mode_mismatch_expected_{expected_mode}_got_{actual_mode}"
