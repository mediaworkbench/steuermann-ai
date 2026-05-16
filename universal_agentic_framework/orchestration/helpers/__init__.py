"""Helpers for graph building and orchestration."""

# Region inference
from .region_inference import (
    build_country_alias_map,
    infer_country_iso2,
    region_for_country,
)

# Directive parsing
from .directives import extract_exact_reply_directive

# Tool payloads
from .tool_payload import (
    normalize_tool_payload,
    error_tool_payload,
    record_tool_success,
    record_tool_error,
)

# Text processing
from .text_processing import (
    truncate_text_by_tokens,
    build_attachment_context_block,
    build_workspace_document_context_block,
    extract_json_object,
)

# Intent detection
from .intent_detection import detect_tool_routing_intents

# Tool scoring
from .tool_scoring import (
    score_tool_similarity,
    build_semantic_tool_kwargs as build_prefilter_tool_kwargs,
)

# Semantic execution
from .semantic_execution import (
    extract_calculator_expression,
    build_semantic_tool_kwargs,
    run_forced_tool,
    execute_semantic_scored_tools,
)

# Tool preparation
from .tool_preparation import (
    prepare_scored_tools_with_forced_execution,
    apply_top_k_scored_tools,
)

# Embedding provider
from .embedding_provider import (
    get_routing_embedding_provider,
    clear_embedding_cache,
)

# Model resolution
from .model_resolution import (
    safe_get_model,
    resolve_initial_model_metadata,
    invoke_with_model_fallback,
)

# Tool-calling mode
from .tool_calling_mode import (
    resolve_effective_tool_calling_mode,
    record_runtime_native_tool_leak,
    validate_and_log_tool_calling_mode,
)

__all__ = [
    # Region inference
    "build_country_alias_map",
    "infer_country_iso2",
    "region_for_country",
    # Directives
    "extract_exact_reply_directive",
    # Tool payloads
    "normalize_tool_payload",
    "error_tool_payload",
    "record_tool_success",
    "record_tool_error",
    # Text processing
    "truncate_text_by_tokens",
    "build_attachment_context_block",
    "build_workspace_document_context_block",
    "extract_json_object",
    # Intent detection
    "detect_tool_routing_intents",
    # Tool scoring
    "score_tool_similarity",
    "build_prefilter_tool_kwargs",
    # Semantic execution
    "extract_calculator_expression",
    "build_semantic_tool_kwargs",
    "run_forced_tool",
    "execute_semantic_scored_tools",
    # Tool preparation
    "prepare_scored_tools_with_forced_execution",
    "apply_top_k_scored_tools",
    # Embedding provider
    "get_routing_embedding_provider",
    "clear_embedding_cache",
    # Model resolution
    "safe_get_model",
    "resolve_initial_model_metadata",
    "invoke_with_model_fallback",
    # Tool-calling mode
    "resolve_effective_tool_calling_mode",
    "record_runtime_native_tool_leak",
    "validate_and_log_tool_calling_mode",
]
