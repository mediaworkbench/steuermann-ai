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
    truncate_tabular_by_rows,
    build_attachment_context_block,
    build_workspace_document_context_block,
    build_workspace_tool_paths,
    extract_json_object,
)

# Intent detection
from .intent_detection import detect_tool_routing_intents

# Tool scoring
from .tool_scoring import (
    score_tool_similarity,
    intent_boost_applies,
    apply_intent_override_floor,
    intent_override_signalled,
)

# Tool preparation
from .tool_preparation import apply_top_k_scored_tools

# Layer 2 tool-call argument preparation
from .tool_call_args import (
    apply_web_search_max_results,
    infer_extract_webpage_url,
    coerce_tool_args,
)

# Embedding provider
from .embedding_provider import (
    get_routing_embedding_provider,
    clear_embedding_cache,
)

# Model resolution
from .model_resolution import (
    get_auxiliary_model,
    get_model,
    resolve_initial_model_metadata,
    invoke_with_model_fallback,
)

# Tool-calling mode
from .tool_calling_mode import (
    resolve_effective_tool_calling_mode,
    record_runtime_native_tool_leak,
    validate_and_log_tool_calling_mode,
)

# RAG retrieval
from .rag_retrieval import (
    extract_rag_keyword,
    filter_and_deduplicate,
    resolve_rag_config,
    search_qdrant,
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
    "truncate_tabular_by_rows",
    "build_attachment_context_block",
    "build_workspace_document_context_block",
    "build_workspace_tool_paths",
    "extract_json_object",
    # Intent detection
    "detect_tool_routing_intents",
    # Tool scoring
    "score_tool_similarity",
    "intent_boost_applies",
    "apply_intent_override_floor",
    "intent_override_signalled",
    # Tool preparation
    "apply_top_k_scored_tools",
    # Layer 2 tool-call argument preparation
    "apply_web_search_max_results",
    "infer_extract_webpage_url",
    "coerce_tool_args",
    # Embedding provider
    "get_routing_embedding_provider",
    "clear_embedding_cache",
    # Model resolution
    "get_auxiliary_model",
    "get_model",
    "resolve_initial_model_metadata",
    "invoke_with_model_fallback",
    # Tool-calling mode
    "resolve_effective_tool_calling_mode",
    "record_runtime_native_tool_leak",
    "validate_and_log_tool_calling_mode",
    # RAG retrieval
    "extract_rag_keyword",
    "filter_and_deduplicate",
    "resolve_rag_config",
    "search_qdrant",
]
