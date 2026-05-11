"""Tool-calling mode validation and resolution helpers."""

from typing import Tuple

def resolve_effective_tool_calling_mode(config, state):
    """Resolve the effective tool-calling mode based on LLM capability and config.

    Checks whether the selected model/provider supports native tool-calling
    via the capability probe registry, and automatically downgrades
    native → structured if the model lacks native support.

    Returns: (mode, reason)
    where mode is one of: "native", "structured", "react"
    and reason explains how/why it was selected.
    """
    if not hasattr(config, "llm"):
        return "structured", "config.llm not found"

    llm_config = config.llm
    preferred_mode = getattr(llm_config, "tool_calling", "native")
    preferred_mode = preferred_mode.lower() if preferred_mode else "native"
    preferred_mode = preferred_mode.lower()
    if preferred_mode not in ("native", "structured", "react"):
        preferred_mode = "native"

    model_id = state.get("model_id", "")
    provider_id = state.get("provider_id", "")

    # Try to load capability probe results (if available)
    try:
        from universal_agentic_framework.llm.llm_capability_probe import LLMCapabilityProbeStore
        probe_store = LLMCapabilityProbeStore()
        capabilities = probe_store.get_capabilities(model_id)

        if capabilities and preferred_mode == "native":
            supports_native = capabilities.get("native_tool_calling", False)
            if not supports_native:
                return "structured", f"Downgraded: model {model_id} does not support native tool-calling"
    except Exception:
        # Probe unavailable; proceed with configured mode
        pass

    reason = f"Configured mode: {preferred_mode}"
    if provider_id:
        reason += f" for provider {provider_id}"
    return preferred_mode, reason


def validate_and_log_tool_calling_mode(
    model_id: str,
    provider_id: str,
    effective_mode: str,
    routing_metadata: dict,
):
    """Validate and log tool-calling mode to routing metadata.

    Ensures that the effective mode is recorded for later audit/debugging
    and that there are no incompatibilities between mode and invocation strategy.
    """
    if effective_mode not in ("native", "structured", "react"):
        raise ValueError(f"Invalid tool-calling mode: {effective_mode}")

    routing_metadata["tool_calling_mode"] = effective_mode
    routing_metadata["tool_calling_mode_source"] = f"model:{model_id}|provider:{provider_id}"

    # Log mode selection for observability
    try:
        import structlog
        logger = structlog.get_logger()
        logger.info(
            "tool_calling_mode_resolved",
            model_id=model_id,
            provider_id=provider_id,
            effective_mode=effective_mode,
        )
    except Exception:
        pass
