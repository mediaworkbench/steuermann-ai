"""LangGraph orchestration: single source of truth for execution control and state management.

MEMORY LAYER INTEGRATION & SOURCE OF TRUTH OWNERSHIP:

1. SHORT-MEMORY (Digest Chain)
   - Owner: LangGraph graph orchestration via GraphState
   - Lifecycle:
     a) Created: performance_nodes.conversation_compression_node() extracts digests
     b) Stored: GraphState.digest_context (bounded list)
     c) Propagated: node_update_memory() → update_memory_node(state)
     d) Persisted: Backend stores in Mem0 record metadata
     e) Retrieved: load_memory_node() returns with digest context intact
   - Validation: Digest metadata must appear in loaded_memory after upsert (checkpoint #7)

2. LONG-MEMORY (Mem0 Records)
   - Owner: Mem0MemoryBackend + Qdrant vector store
   - Graph integration:
     - load_memory_node() queries backend → populates loaded_memory, digest_context, memory_analytics
     - update_memory_node() receives digest_chain → passes to backend.upsert()
    - Metadata consistency maintained via canonical Mem0 metadata fields

3. KNOWLEDGE GRAPH (Co-occurrence Links)
   - Current owner: MemoryCoOccurrenceTracker (in-memory, non-persistent)
   - Planned owner: PostgreSQL co_occurrence_edges (Phase 3)
   - Graph integration:
     - Populated during load_memory_node() retrieval
     - Used for related_memory expansion in memory_analytics
   - Caveat: Lost on restart (planned fix in Phase 3)

CRITICAL CONSTRAINTS:
- GraphState is session-scoped; one instance per user session
- No global/shared state across sessions (memory is explicit node operation)
- Digest chain bounded by core.memory.digest_max_items config

See: docs/technical_architecture.md (Memory Architecture) for full memory layer design
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.channels import UntrackedValue
from langgraph.graph import START, StateGraph

from universal_agentic_framework.llm.budget import (
    count_tokens_for_model,
    estimate_tokens,
)
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.config import get_active_profile_id, load_core_config, load_tools_config, load_features_config
from universal_agentic_framework.memory.nodes import load_memory_node, update_memory_node
from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.tools import ToolRegistry
from universal_agentic_framework.embeddings import build_embedding_provider, EmbeddingProvider
from universal_agentic_framework.monitoring.metrics import (
    track_node_execution,
    track_tokens,
    track_llm_call,
    track_memory_operation,
)
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.orchestration.performance_nodes import (
    initialize_performance_nodes,
    memory_query_cache_node,
    memory_cache_store_node,
    conversation_compression_node,
    cache_stats_node,
    get_summarizer,
)
from universal_agentic_framework.orchestration.crew_nodes import (
    node_research_crew,
    node_analytics_crew,
    node_code_generation_crew,
    node_planning_crew,
    node_crew_chain,
    node_crew_parallel,
    CREW_SPECS,
)
from universal_agentic_framework.orchestration.rag_node import node_retrieve_knowledge
from universal_agentic_framework.orchestration.checkpointing import build_checkpointer
from universal_agentic_framework.orchestration.respond.text_cleanup import (
    strip_control_tokens,
    filter_untrusted_urls,
)
from universal_agentic_framework.orchestration.respond.prompt_builder import (
    build_tool_results_block,
    select_synthesis_instruction,
    build_memory_context_block,
    build_crew_findings_block,
)
from universal_agentic_framework.orchestration.respond.guardrails import (
    retry_synthesis_if_empty,
    retry_on_attachment_refusal,
    retry_on_web_extract_contradiction,
    format_tool_based_fallback,
)

try:
    from litellm.exceptions import (
        ContextWindowExceededError as _LiteLLMContextWindowExceededError,
        RateLimitError as _LiteLLMRateLimitError,
        AuthenticationError as _LiteLLMAuthenticationError,
        ServiceUnavailableError as _LiteLLMServiceUnavailableError,
    )
except ImportError:
    _LiteLLMContextWindowExceededError = None  # type: ignore[assignment,misc]
    _LiteLLMRateLimitError = None  # type: ignore[assignment,misc]
    _LiteLLMAuthenticationError = None  # type: ignore[assignment,misc]
    _LiteLLMServiceUnavailableError = None  # type: ignore[assignment,misc]

# Import extracted helpers
from universal_agentic_framework.orchestration.helpers import (
    extract_exact_reply_directive,
    extract_json_object,
    record_tool_success,
    record_tool_error,
    build_attachment_context_block,
    build_workspace_document_context_block,
    build_workspace_tool_paths,
    detect_tool_routing_intents,
    score_tool_similarity,
    intent_boost_applies,
    apply_intent_override_floor,
    apply_top_k_scored_tools,
    apply_web_search_max_results,
    infer_extract_webpage_url,
    coerce_tool_args,
    get_routing_embedding_provider,
    get_auxiliary_model,
    get_model,
    resolve_initial_model_metadata,
    invoke_with_model_fallback,
    resolve_effective_tool_calling_mode,
    record_runtime_native_tool_leak,
    validate_and_log_tool_calling_mode,
)

logger = get_logger(__name__)

_DIGEST_CHAIN_MAX_ITEMS = 5

# Prefixes the crew nodes use when appending their results as assistant messages.
# node_generate_response filters these out of the LLM message history and injects them as
# system-level FINDINGS instead — keeping a crew result as a prior assistant turn makes the
# model treat it as its own answer and only add a short follow-up. Derived from CREW_SPECS so
# the append-prefix (crew_nodes) and the filter-prefix (here) can never drift apart again
# (the old W1.1 bug). Matched via str.startswith; "Chain Result (" — produced by
# node_crew_chain, which has no CrewSpec — is appended manually and covers every chain name.
_CREW_RESULT_PREFIXES = tuple(
    spec.message_prefix for spec in CREW_SPECS.values()
) + ("Chain Result (",)

# Cache of discovered tool lists (ToolRegistry.discover_and_load scans the filesystem and
# parses every tool manifest, which is wasteful to repeat per turn). Keyed on the inputs
# that change the discovered set: profile, language (descriptions are language-specific),
# and the profile tools dir. User tool_toggles and the vision-role exclusion are applied
# per-request on top of this, so they are NOT part of the key. The lock serializes the
# first build (sync load_tools node runs in an executor thread under ainvoke).
_tool_registry_cache: Dict[Tuple[str, str, str], List[Any]] = {}
_tool_registry_lock = threading.Lock()


def clear_tool_registry_cache() -> None:
    """Drop cached tool-discovery results (tests, or after tool config/file changes)."""
    with _tool_registry_lock:
        _tool_registry_cache.clear()


def _discover_tools_cached(
    profile_id: str,
    language: str,
    tools_config: Any,
    profile_tools_dir: Any,
) -> List[Any]:
    """Return the discovered (unfiltered) tool list, cached per profile+language+dir."""
    key = (str(profile_id), str(language), str(profile_tools_dir))
    cached = _tool_registry_cache.get(key)
    if cached is not None:
        return cached
    with _tool_registry_lock:
        cached = _tool_registry_cache.get(key)
        if cached is None:
            registry = ToolRegistry(
                config=tools_config,
                profile_language=language,
                extra_tools_dir=profile_tools_dir,
            )
            cached = registry.discover_and_load()
            _tool_registry_cache[key] = cached
    return cached


def _rag_label(file_name: str | None) -> str:
    """Human-readable RAG source label: strip ingestion hash prefix, replace dashes with spaces."""
    raw = str(file_name or "Unknown")
    # Strip only a trailing extension — removesuffix, not replace, so an extension
    # substring inside the name (e.g. "csv-export-notes.md") is preserved.
    for ext in (".md", ".txt", ".pdf", ".csv", ".html", ".xml", ".yaml", ".yml"):
        if raw.endswith(ext):
            raw = raw[: -len(ext)]
            break
    raw = re.sub(r"^[0-9a-f]{32}-", "", raw)  # strip leading ingestion hash
    return raw.replace("-", " ").strip() or "Unknown"


def _is_digest_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get("digest_id"))


# User turns this short, or matching these patterns, don't warrant a long-term memory
# write — nor the auxiliary-model fact-extraction summary that exists only to feed it.
_TRIVIAL_MEMORY_PATTERNS = frozenset({
    "ok", "okay", "thanks", "thank you", "yes", "no", "sure", "got it",
    "hi", "hello", "bye", "goodbye", "great", "perfect", "alright",
})


def _latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _should_write_memory(state: GraphState, features_config: Any) -> bool:
    """True when this turn's exchange is worth a long-term memory write.

    False when long-term memory is disabled or the latest user turn is trivial (a
    greeting/acknowledgement, or under 5 characters). Shared by node_summarize (to skip
    the auxiliary-model fact-extraction call) and node_update_memory (to skip the upsert),
    so the expensive summary is never produced for a turn that won't be persisted.
    """
    if not getattr(features_config, "long_term_memory", False):
        return False
    stripped = _latest_user_message(state.get("messages", [])).strip().lower().rstrip("!.,?")
    if len(stripped) < 5:
        return False
    if len(stripped) < 20 and stripped in _TRIVIAL_MEMORY_PATTERNS:
        return False
    return True


def _merge_digest_chains(
    existing: List[Dict[str, Any]],
    extracted: List[Dict[str, Any]],
    *,
    max_items: int = _DIGEST_CHAIN_MAX_ITEMS,
) -> List[Dict[str, Any]]:
    """Merge digest chains (newest-first), dedupe by digest_id, and cap length."""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(extracted) + list(existing):
        if not _is_digest_entry(item):
            continue
        digest_id = str(item.get("digest_id"))
        if digest_id in seen:
            continue
        seen.add(digest_id)
        merged.append(dict(item))
        if len(merged) >= max_items:
            break
    return merged

class GraphState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]  # Uploaded conversation attachments from adapter
    attachment_context: List[Dict[str, Any]]  # Prompt-ready normalized attachment snippets
    workspace_documents: List[Dict[str, Any]]  # User workspace documents resolved by adapter
    workspace_document_context: List[Dict[str, Any]]  # Prompt-ready normalized workspace snippets
    workspace_writeback_requested: bool
    workspace_writeback_document: Optional[Dict[str, Any]]  # Full document content for writeback (bypasses 600-token truncation)
    user_id: str
    session_id: str  # Session identifier for co-occurrence tracking
    language: str
    profile_name: str  # Fork identifier for metrics
    profile_id: str  # Active deployment profile identifier
    user_settings: Dict[str, Any]  # User preferences: tool_toggles, rag_config, preferred_model, theme, language
    llm_capability_probes: List[Dict[str, Any]]  # Adapter-provided probe snapshots per provider
    loaded_memory: List[Dict[str, Any]]
    digest_context: List[Dict[str, Any]]  # Digest subset from loaded memory retrieval
    memory_analytics: Dict[str, Any]  # Memory retrieval analytics (importance scores, related count)
    knowledge_context: List[Dict[str, Any]]  # RAG retrieved documents
    rag_attempted: bool  # True if Qdrant was queried this turn (False on all skip paths)
    rag_doc_count: int   # Number of docs above pill_score_threshold (injected into prompt + shown as pills)
    loaded_tools: Annotated[List[Any], UntrackedValue(list)]  # BaseTool instances — not checkpointed (not serializable)
    candidate_tools: Annotated[List[Dict[str, Any]], UntrackedValue(list)]  # Layer 1 pre-filter output — not checkpointed
    tool_calling_mode: str  # "native" | "structured" | "react" — from provider config
    tool_calling_mode_reason: str  # Why selected mode was chosen (configured or downgraded)
    prefilter_intents: Dict[str, Any]  # Intent detection results from Layer 1 for Layer 2 use
    tool_results: Dict[str, str]  # Tool name -> execution result
    tool_execution_results: Dict[str, Dict[str, Any]]  # Tool name -> structured execution envelope
    routing_metadata: Dict[str, str]  # Tool name -> reason for selection
    crew_results: Dict[str, Any]  # Multi-agent crew execution results
    tokens_used: int
    turn_tokens_used: int  # Current invocation token usage for per-turn budgeting
    input_tokens: int  # Input tokens count (cumulative across the conversation)
    output_tokens: int  # Output tokens count (cumulative across the conversation)
    last_input_tokens: int  # Prompt tokens of the most recent respond inference (per-turn, overwritten)
    last_output_tokens: int  # Output tokens of the most recent respond inference (per-turn, overwritten)
    provider_used: str  # Actual provider used for response generation
    model_used: str  # Actual model used for response generation
    summary_text: str
    digest_chain: List[Dict[str, Any]]  # Rolling conversation digest metadata
    last_compression_status: str  # "ok" | "skipped" | "error" — outcome of the last compress_state run
    context_breakdown: Dict[str, int]  # Approximate per-section prompt token estimates for the context-window menu
    sources: List[Dict[str, Any]]  # [{type: "web"|"rag", label: str, url: str|None}]
    query_embedding: List[float]  # Precomputed user-message embedding from node_prefilter_tools; reused by RAG to avoid duplicate encode


def _get_routing_embedding_provider(config: Any) -> Tuple[EmbeddingProvider, str]:
    """Return cached embedding provider and model name used for tool routing."""
    return get_routing_embedding_provider(
        config,
        logger=logger,
        build_provider_func=build_embedding_provider,
    )


class _ModelInvokeError(RuntimeError):
    def __init__(self, message: str, provider: str, model_name: str, error_type: str = "error"):
        super().__init__(message)
        self.provider = provider
        self.model_name = model_name
        self.error_type = error_type


def _tokens_from_usage(
    usage_metadata: Optional[dict], fallback_text: str, fallback_input_estimate: int = 0
) -> Tuple[int, int]:
    """Return (input_tokens, output_tokens) from usage_metadata or char/4 fallback.

    fallback_input_estimate is used as the input token count when usage_metadata is
    absent (e.g. the provider did not report usage). Pass it pre-computed from the
    prompt messages so the context ring always reflects the full accumulated history.
    """
    if usage_metadata:
        return (
            usage_metadata.get("input_tokens", 0),
            usage_metadata.get("output_tokens", 0),
        )
    output_approx = estimate_tokens(fallback_text)
    if fallback_input_estimate == 0:
        logger.warning("_tokens_from_usage: usage_metadata absent and no fallback estimate provided; input_tokens will be 0 — check that the LLM provider returns usage metadata")
    return fallback_input_estimate, output_approx


def _invoke_with_model_fallback(
    *,
    config: Any,
    language: str,
    payload: Any,
    initial_model: object,
    initial_provider: str,
    initial_model_name: str,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str, str, object, Optional[dict]]:
    """Thin wrapper over helper implementation preserving local error type."""
    return invoke_with_model_fallback(
        config=config,
        language=language,
        payload=payload,
        initial_model=initial_model,
        initial_provider=initial_provider,
        initial_model_name=initial_model_name,
        preferred_model=preferred_model,
        logger=logger,
        error_cls=_ModelInvokeError,
    )


def node_load_tools(state: GraphState) -> GraphState:
    """Load tools from registry based on config/tools.yaml."""
    tools_config = load_tools_config()
    core_config = load_core_config()
    profile_name = getattr(core_config.profile, "name", "default-profile")
    profile_id = state.get("profile_id") or get_active_profile_id()
    profile_language = getattr(core_config.profile, "language", "en")
    # Use conversation language from state; fall back to profile_language from config
    conversation_language = state.get("language") or profile_language

    logger.info("Loading tools from registry", profile_name=profile_name, language=conversation_language)

    with track_node_execution(profile_name, "load_tools"):
        try:
            from universal_agentic_framework.config import get_profile_dir

            profile_dir = get_profile_dir(profile_id=profile_id, require_exists=False)
            profile_tools_dir = profile_dir / "tools"

            # Cached filesystem discovery; copy so the per-request filters below never
            # mutate or alias the shared cached list.
            tools = list(
                _discover_tools_cached(
                    profile_id, conversation_language, tools_config, profile_tools_dir
                )
            )

            # Filter tools based on user settings (tool_toggles)
            user_settings = state.get("user_settings", {})
            tool_toggles = user_settings.get("tool_toggles", {})

            if tool_toggles:
                filtered_tools = [
                    t for t in tools
                    if tool_toggles.get(t.name, True)
                ]
                logger.info(
                    "Tools filtered by user settings",
                    original_count=len(tools),
                    filtered_count=len(filtered_tools),
                    disabled_tools=[t.name for t in tools if not tool_toggles.get(t.name, True)],
                )
                tools = filtered_tools

            # Exclude all vision-LLM tools when the vision role is not configured.
            _VISION_LLM_TOOLS = {"analyze_image_tool", "ocr_tool", "analyze_document_tool", "analyze_chart_tool"}
            vision_role = getattr(getattr(core_config.llm, "roles", None), "vision", None)
            if vision_role is None:
                before = len(tools)
                tools = [t for t in tools if t.name not in _VISION_LLM_TOOLS]
                excluded = before - len(tools)
                if excluded:
                    logger.info("Vision LLM tools excluded: llm.roles.vision not configured", count=excluded)

            state["loaded_tools"] = tools
            logger.info(
                "Tools loaded",
                tools_count=len(tools),
                tool_names=[t.name for t in tools],
            )
            return state
        except Exception as e:
            logger.error("Tool loading failed", error=str(e), exc_info=True)
            state["loaded_tools"] = []
            return state


def node_prefilter_tools(state: GraphState) -> GraphState:
    """Layer 1: Semantic pre-filter — score tools and return top-K candidates.

    Does NOT execute tools. Stores candidate tools in state for Layer 2
    (model-driven calling). Intent detection boosts scores instead of
    forcing execution.
    """
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")

    loaded_tools = state.get("loaded_tools", [])
    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )

    empty_state = {
        "candidate_tools": [],
        "tool_calling_mode": "structured",
        "tool_calling_mode_reason": "default_empty_state",
        "prefilter_intents": {},
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
    }

    if not loaded_tools or not user_msg:
        state.update(empty_state)
        return state

    logger.info(
        "Pre-filtering tools (Layer 1)",
        profile_name=profile_name,
        tools_count=len(loaded_tools),
        query_length=len(user_msg),
    )

    with track_node_execution(profile_name, "prefilter_tools"):
        try:
            # Skip for greetings
            greeting_pattern = (
                r"^\s*(hi|hello|hey|hallo|servus|moin|"
                r"guten\s+(tag|morgen|abend))\s*[!.?]*\s*$"
            )
            if re.match(greeting_pattern, user_msg.lower()):
                logger.info("Skipping tool pre-filter for greeting query")
                state.update(empty_state)
                # Greetings can't benefit from RAG — signal the RAG node to skip Qdrant.
                # (empty_state's prefilter_intents={} would otherwise let RAG run here, and
                # the meta-question path below already propagates skip_rag via its intents.)
                state["prefilter_intents"] = {"skip_rag": True}
                return state

            embedding_provider, embedding_model_name = _get_routing_embedding_provider(config)
            query_embedding = embedding_provider.encode(user_msg)
            # Cache in state so downstream nodes (RAG) can reuse without re-encoding.
            state["query_embedding"] = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)

            # Default aligned with docs/CLAUDE.md (0.55); profiles set this explicitly,
            # so the fallback only applies to a profile that omits tool_routing entirely.
            similarity_threshold = getattr(
                getattr(config, "tool_routing", None), "similarity_threshold", 0.55
            )
            top_k = getattr(getattr(config, "tool_routing", None), "top_k", 5)
            intent_boost = getattr(
                getattr(config, "tool_routing", None), "intent_boost", 0.2
            )
            routing_language = state.get("language") or getattr(
                config.profile, "language", "en"
            )

            # Detect intents for score boosting
            intents = detect_tool_routing_intents(
                user_msg=user_msg, language=routing_language
            )

            image_attachment_present = any(
                str(a.get("mime_type", "")).startswith("image/")
                for a in (state.get("attachments") or [])
            )
            csv_workspace_doc_present = any(
                str(d.get("mime_type", "")) == "text/csv" or str(d.get("filename", "")).endswith(".csv")
                for d in (state.get("workspace_documents") or [])
            )

            if intents["asks_about_tools"]:
                logger.info("Meta-question detected: skipping tool pre-filter")
                state.update(empty_state)
                state["prefilter_intents"] = intents
                return state

            # Score all tools with intent boosting
            scored_tools: List[Tuple[Any, float]] = []
            for tool in loaded_tools:
                tool_name = getattr(tool, "name", "unknown")
                tool_desc = getattr(tool, "description", "")
                if not tool_desc:
                    continue

                similarity = score_tool_similarity(
                    embedding_model_name=embedding_model_name,
                    tool_name=tool_name,
                    tool_desc=tool_desc,
                    embedding_provider=embedding_provider,
                    query_embedding=query_embedding,
                )

                # Intent boost: increase score for tools matching detected intents
                # (declarative rules live in helpers/tool_scoring.py; mirrors CLAUDE.md).
                if intent_boost_applies(
                    tool_name,
                    intents,
                    image_attachment_present=image_attachment_present,
                    csv_workspace_doc_present=csv_workspace_doc_present,
                ):
                    similarity += intent_boost

                # Hard intent override: web_search_mcp / map_tool get a forced floor above both
                # threshold gates when their intent is explicitly signalled, so Layer 2 decides.
                _min_top_score_cfg = getattr(
                    getattr(config, "tool_routing", None), "min_top_score", 0.7
                )
                similarity, _override_applied = apply_intent_override_floor(
                    tool_name,
                    similarity,
                    intents,
                    similarity_threshold=similarity_threshold,
                    min_top_score=_min_top_score_cfg,
                )
                if _override_applied:
                    logger.info(
                        "Applying intent override floor",
                        tool=tool_name,
                        forced_similarity=round(similarity, 4),
                    )

                logger.info(
                    "Tool scored (prefilter)",
                    tool=tool_name,
                    similarity=round(similarity, 4),
                    threshold=similarity_threshold,
                )
                scored_tools.append((tool, similarity))

            # Apply top-K
            scored_tools = apply_top_k_scored_tools(scored_tools, top_k)

            min_top_score = getattr(
                getattr(config, "tool_routing", None), "min_top_score", 0.7
            )
            min_spread = getattr(
                getattr(config, "tool_routing", None), "min_spread", 0.10
            )

            # Min-top-score gate: if no tool scores confidently, skip Layer 2
            if scored_tools:
                top_score = max(s for _, s in scored_tools)
                if top_score < min_top_score:
                    logger.info(
                        "Min-top-score gate: no tool scored confidently, clearing candidates",
                        top_score=round(top_score, 4),
                        min_top_score=min_top_score,
                    )
                    scored_tools = []

            # Score-spread gate: if remaining tools all scored similarly, clear
            if len(scored_tools) >= 3:
                scores = [s for _, s in scored_tools]
                max_score = max(scores)
                mean_score = sum(scores) / len(scores)
                spread = max_score - mean_score
                if spread < min_spread:
                    logger.info(
                        "Score-spread gate: all tools scored similarly, clearing candidates",
                        max_score=round(max_score, 4),
                        mean_score=round(mean_score, 4),
                        spread=round(spread, 4),
                        min_spread=min_spread,
                    )
                    scored_tools = []

            # Filter by threshold and build candidate list
            candidates = [
                {"tool": tool, "name": getattr(tool, "name", "unknown"), "score": score}
                for tool, score in scored_tools
                if score >= similarity_threshold
            ]

            tool_calling_mode, tool_calling_mode_reason = resolve_effective_tool_calling_mode(config, state)

            state["candidate_tools"] = candidates
            state["tool_calling_mode"] = tool_calling_mode
            state["tool_calling_mode_reason"] = tool_calling_mode_reason
            state["prefilter_intents"] = intents
            state["tool_results"] = {}
            state["tool_execution_results"] = {}
            state["routing_metadata"] = {}

            logger.info(
                "Tool pre-filter completed (Layer 1)",
                candidates=len(candidates),
                tool_calling_mode=tool_calling_mode,
                tool_calling_mode_reason=tool_calling_mode_reason,
                candidate_names=[c["name"] for c in candidates],
            )

        except Exception as e:
            logger.error("Tool pre-filter failed", error=str(e), exc_info=True)
            state.update(empty_state)

    return state


def node_call_tools_native(state: GraphState) -> GraphState:
    """Layer 2 (native): Bind candidate tools to LLM, let model decide which to call.

    Uses LangChain's ``bind_tools`` + ``tool_calls`` parsing.  Executes the
    tools the model requests, stores results in the same ``tool_results`` /
    ``tool_execution_results`` fields used by the rest of the pipeline.
    
    Mode enforcement: Validates that tool_calling_mode is 'native' before proceeding.
    """
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    max_retries = getattr(getattr(config, "tool_routing", None), "max_retries", 2)

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "native", "call_tools_native", profile_name
    )
    if not is_valid:
        logger.warning(
            "Native tool calling invoked but mode doesn't match; proceeding with caution",
            validation_reason=validation_reason,
            actual_mode=state.get("tool_calling_mode"),
        )

    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )
    if not user_msg:
        return state

    prefilter_intents = state.get("prefilter_intents") or {}
    url_in_query = prefilter_intents.get("url_in_query")
    wants_save_to_rag = bool(prefilter_intents.get("wants_save_to_rag"))

    logger.info(
        "Layer 2 native tool calling",
        profile_name=profile_name,
        candidates=len(candidates),
    )

    with track_node_execution(profile_name, "call_tools_native"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            lang = state.get("language") or getattr(config.profile, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = get_model(config, lang, preferred_model=preferred_model)

            # Extract BaseTool instances and bind to model
            tools = [c["tool"] for c in candidates]
            model_with_tools = model.bind_tools(tools)

            # Build minimal messages for the tool-calling invocation
            messages = [
                SystemMessage(content="You are a helpful assistant. Use the provided tools when they help answer the user's question."),
                HumanMessage(content=user_msg),
            ]

            tool_results: Dict[str, str] = state.get("tool_results", {})
            tool_execution_results: Dict[str, Dict[str, Any]] = state.get("tool_execution_results", {})
            routing_metadata: Dict[str, str] = state.get("routing_metadata", {})
            tool_lookup = {getattr(t, "name", ""): t for t in tools}

            # Retry loop (Layer 3 validation)
            for attempt in range(max_retries + 1):
                response = model_with_tools.invoke(messages)
                tool_calls = getattr(response, "tool_calls", None) or []

                if not tool_calls:
                    logger.info("Model chose not to call any tools", attempt=attempt)

                    # Deterministic fallback: if a URL is present and extract tool is available,
                    # run extraction even when native tool-calling returns no calls.
                    extract_tool = tool_lookup.get("extract_webpage_mcp")
                    if extract_tool and url_in_query and "extract_webpage_mcp" not in tool_results:
                        try:
                            fallback_result = extract_tool._run(
                                request_url=url_in_query,
                                save_to_rag=wants_save_to_rag,
                            )
                            record_tool_success(
                                tool_name="extract_webpage_mcp",
                                result=fallback_result,
                                reason="native URL fallback (no model tool call)",
                                tool_results=tool_results,
                                tool_execution_results=tool_execution_results,
                                routing_metadata=routing_metadata,
                                args={"request_url": url_in_query, "save_to_rag": wants_save_to_rag},
                            )
                            logger.info(
                                "Native fallback executed",
                                tool="extract_webpage_mcp",
                                result_length=len(str(fallback_result)),
                            )
                        except Exception as fallback_err:
                            record_tool_error(
                                tool_name="extract_webpage_mcp",
                                error=fallback_err,
                                tool_results=tool_results,
                                tool_execution_results=tool_execution_results,
                                args={"request_url": url_in_query, "save_to_rag": wants_save_to_rag},
                            )
                            logger.error(
                                "Native fallback execution failed",
                                tool="extract_webpage_mcp",
                                error=str(fallback_err),
                            )
                    break

                parse_error = False
                # Tools that already ran in a previous attempt. The retry loop exists only to
                # correct calls that failed to parse/validate; re-executing a tool that already
                # succeeded would duplicate its side effects (web search, save_to_rag, …).
                # Snapshotted per attempt so a model legitimately calling the same tool twice
                # within one response is still honoured.
                executed_before = set(tool_results.keys())
                _requested_results = (state.get("prefilter_intents") or {}).get("requested_web_results")
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})

                    tool_args = apply_web_search_max_results(tool_name, tool_args, _requested_results)
                    tool_args = infer_extract_webpage_url(tool_name, tool_args, url_in_query)

                    tool_obj = tool_lookup.get(tool_name)
                    if not tool_obj:
                        logger.warning("Model requested unknown tool", tool=tool_name, attempt=attempt)
                        parse_error = True
                        continue

                    # Already executed in a prior attempt — skip (not a parse error) so the
                    # retry only runs the tools that previously failed to parse/validate.
                    if tool_name in executed_before:
                        logger.info("Skipping already-executed tool on retry", tool=tool_name, attempt=attempt)
                        continue

                    # Validate args against schema if available; also strips unknown fields.
                    tool_args, _val_err = coerce_tool_args(tool_obj, tool_args)
                    if _val_err:
                        logger.warning(
                            "Tool call validation failed",
                            tool=tool_name,
                            error=_val_err,
                            attempt=attempt,
                        )
                        parse_error = True
                        continue

                    try:
                        result = tool_obj._run(**tool_args)

                        if (
                            tool_name == "extract_webpage_mcp"
                            and url_in_query
                            and isinstance(result, str)
                            and "Request URL is missing an 'http://' or 'https://' protocol" in result
                        ):
                            logger.warning(
                                "Extract tool returned protocol-missing error despite URL intent; retrying with inferred URL",
                                url=url_in_query,
                            )
                            result = tool_obj._run(request_url=url_in_query)

                        record_tool_success(
                            tool_name=tool_name,
                            result=result,
                            reason=f"native tool call (attempt {attempt})",
                            tool_results=tool_results,
                            tool_execution_results=tool_execution_results,
                            routing_metadata=routing_metadata,
                            args=tool_args,
                        )
                        logger.info("Tool executed (native)", tool=tool_name, result_length=len(str(result)))
                    except Exception as exec_err:
                        record_tool_error(
                            tool_name=tool_name,
                            error=exec_err,
                            tool_results=tool_results,
                            tool_execution_results=tool_execution_results,
                            args=tool_args,
                        )
                        logger.error("Tool execution failed (native)", tool=tool_name, error=str(exec_err))

                if not parse_error or attempt >= max_retries:
                    break

                # Re-prompt with error feedback for retry
                logger.info("Retrying tool call due to parse/validation error", attempt=attempt + 1)
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=response.content or ""))
                messages.append(HumanMessage(
                    content="Your previous tool call had errors. Please try again with valid tool names and arguments."
                ))

            state["tool_results"] = tool_results
            state["tool_execution_results"] = tool_execution_results
            state["routing_metadata"] = routing_metadata
            logger.info("Native tool calling completed", tools_executed=len(tool_results))

        except Exception as e:
            logger.error("Native tool calling failed", error=str(e), exc_info=True)

    return state


def node_call_tools_structured(state: GraphState) -> GraphState:
    """Layer 2 (structured): Inject tool schemas in prompt, let model output JSON tool calls.

    For models without native function-calling support.  Tool schemas are
    injected into the system prompt alongside an instruction to output JSON
    when a tool should be called.
    
    Mode enforcement: Validates that tool_calling_mode is 'structured' before proceeding.
    """
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    max_retries = getattr(getattr(config, "tool_routing", None), "max_retries", 2)

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "structured", "call_tools_structured", profile_name
    )
    if not is_valid:
        logger.warning(
            "Structured tool calling invoked but mode doesn't match; proceeding with caution",
            validation_reason=validation_reason,
            actual_mode=state.get("tool_calling_mode"),
        )

    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )
    if not user_msg:
        return state

    # Determine whether this turn requires the model to call a tool.
    # High-confidence web search intents (explicit "search the web" phrasing) and
    # any candidate scoring ≥ 0.75 both count as obligation signals.
    intents = state.get("prefilter_intents") or {}
    top_score = candidates[0].get("score", 0.0) if candidates else 0.0
    force_tool_use = bool(intents.get("force_tool_use") or top_score >= 0.75)

    logger.info(
        "Layer 2 structured tool calling",
        profile_name=profile_name,
        candidates=len(candidates),
        force_tool_use=force_tool_use,
    )

    with track_node_execution(profile_name, "call_tools_structured"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            lang = state.get("language") or getattr(config.profile, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = get_model(config, lang, preferred_model=preferred_model)
            try:
                _aux, _, _ = get_auxiliary_model(config, lang)
                retry_model = _aux if _aux else model
            except Exception:
                retry_model = model

            tools = [c["tool"] for c in candidates]
            tool_lookup = {getattr(t, "name", ""): t for t in tools}

            # Build tool schema descriptions for the prompt
            tool_schemas = []
            for tool in tools:
                name = getattr(tool, "name", "unknown")
                desc = getattr(tool, "description", "")
                schema = getattr(tool, "args_schema", None)
                args_desc = ""
                if schema:
                    try:
                        if hasattr(schema, "model_json_schema"):
                            args_desc = json.dumps(schema.model_json_schema(), indent=2)
                        elif hasattr(schema, "schema"):
                            args_desc = json.dumps(schema.schema(), indent=2)
                    except Exception:
                        if hasattr(schema, "model_fields"):
                            args_desc = str(schema.model_fields)
                        elif hasattr(schema, "__fields__"):
                            args_desc = str(schema.__fields__)
                        else:
                            args_desc = ""
                tool_schemas.append(f"- {name}: {desc}\n  Args: {args_desc}")

            tool_block = "\n".join(tool_schemas)
            # When the user clearly wants a tool (web search or high-confidence candidate),
            # replace the opt-out footer with a mandatory instruction so the model cannot
            # silently skip the call.
            tool_footer = (
                "You MUST call one of the available tools listed above. "
                "Do not respond with plain text — output ONLY the JSON tool call."
                if force_tool_use
                else "If no tool is needed, respond normally in plain text."
            )
            attachment_block, _ = build_attachment_context_block(state.get("attachments") or [])
            attachment_section = f"\n\n{attachment_block}" if attachment_block else ""

            csv_paths_block = build_workspace_tool_paths(state.get("workspace_documents") or [])
            csv_paths_section = f"\n\n{csv_paths_block}" if csv_paths_block else ""

            system_content = (
                "You are a helpful assistant with access to tools.\n"
                "If a tool would help answer the user's question, respond with ONLY a JSON object:\n"
                '{"tool": "<tool_name>", "args": {<arguments>}}\n'
                f"{tool_footer}\n\n"
                f"Available tools:\n{tool_block}"
                f"{attachment_section}"
                f"{csv_paths_section}"
            )

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_msg),
            ]

            tool_results: Dict[str, str] = state.get("tool_results", {})
            tool_execution_results: Dict[str, Dict[str, Any]] = state.get("tool_execution_results", {})
            routing_metadata: Dict[str, str] = state.get("routing_metadata", {})

            def _normalize_structured_response_text(raw_content: Any) -> str:
                """Normalize model content into plain text for structured JSON parsing."""

                def _extract_parts(value: Any) -> List[str]:
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [value]
                    if isinstance(value, list):
                        parts: List[str] = []
                        for item in value:
                            parts.extend(_extract_parts(item))
                        return parts
                    if isinstance(value, dict):
                        # Common content-block shape: {"type": "text", "text": "..."}
                        text_value = value.get("text")
                        if text_value is not None:
                            return _extract_parts(text_value)
                        # Some providers may wrap nested content arrays/objects.
                        nested_value = value.get("content")
                        if nested_value is not None:
                            return _extract_parts(nested_value)
                        return []
                    return [str(value)]

                return "\n".join(part for part in _extract_parts(raw_content) if part)

            for attempt in range(max_retries + 1):
                response = (retry_model if attempt > 0 else model).invoke(messages)
                response_text = _normalize_structured_response_text(
                    response.content if hasattr(response, "content") else response
                )

                # Try to parse JSON tool call from response
                tool_call = extract_json_object(response_text)

                if not tool_call or "tool" not in tool_call:
                    if force_tool_use and attempt < max_retries:
                        logger.info(
                            "Model declined tool call despite force_tool_use; retrying with stricter prompt",
                            attempt=attempt,
                            profile_name=profile_name,
                        )
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(
                            content=(
                                "You did not call a tool. You MUST call one of the listed tools now. "
                                "Output ONLY the JSON tool call — nothing else:\n"
                                '{"tool": "<tool_name>", "args": {<arguments>}}'
                            )
                        ))
                        continue
                    logger.info("Model chose not to call tools (structured)", attempt=attempt)
                    break

                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})

                _requested_results = (state.get("prefilter_intents") or {}).get("requested_web_results")
                tool_args = apply_web_search_max_results(tool_name, tool_args, _requested_results)

                tool_obj = tool_lookup.get(tool_name)

                if not tool_obj:
                    logger.warning("Model requested unknown tool (structured)", tool=tool_name, attempt=attempt)
                    if attempt < max_retries:
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(
                            content=f"Tool '{tool_name}' does not exist. Available tools: {', '.join(tool_lookup.keys())}. Please try again."
                        ))
                        continue
                    break

                # Validate args against schema; also strips unknown fields.
                tool_args, _val_err = coerce_tool_args(tool_obj, tool_args)
                if _val_err:
                    logger.warning(
                        "Structured tool call validation failed",
                        tool=tool_name,
                        error=_val_err,
                        attempt=attempt,
                    )
                    if attempt < max_retries:
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(
                            content=f"Tool call validation error: {_val_err}. Please fix the arguments and try again."
                        ))
                        continue
                    break

                try:
                    result = tool_obj._run(**tool_args)
                    record_tool_success(
                        tool_name=tool_name,
                        result=result,
                        reason=f"structured tool call (attempt {attempt})",
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
                        routing_metadata=routing_metadata,
                        args=tool_args,
                    )
                    logger.info("Tool executed (structured)", tool=tool_name, result_length=len(str(result)))
                except Exception as exec_err:
                    record_tool_error(
                        tool_name=tool_name,
                        error=exec_err,
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
                        args=tool_args,
                    )
                    logger.error("Tool execution failed (structured)", tool=tool_name, error=str(exec_err))
                break

            state["tool_results"] = tool_results
            state["tool_execution_results"] = tool_execution_results
            state["routing_metadata"] = routing_metadata
            logger.info("Structured tool calling completed", tools_executed=len(tool_results))

        except Exception as e:
            logger.error("Structured tool calling failed", error=str(e), exc_info=True)

    return state


def node_call_tools_react(state: GraphState) -> GraphState:
    """Layer 2 (react): ReAct-style tool calling loop (Thought → Action → Observation).

    For weaker models that need step-by-step reasoning.
    
    Mode enforcement: Validates that tool_calling_mode is 'react' before proceeding.
    """
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    max_iterations = getattr(getattr(config, "tool_routing", None), "max_retries", 2) + 1

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "react", "call_tools_react", profile_name
    )
    if not is_valid:
        logger.warning(
            "ReAct tool calling invoked but mode doesn't match; proceeding with caution",
            validation_reason=validation_reason,
            actual_mode=state.get("tool_calling_mode"),
        )

    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )
    if not user_msg:
        return state

    logger.info(
        "Layer 2 ReAct tool calling",
        profile_name=profile_name,
        candidates=len(candidates),
    )

    with track_node_execution(profile_name, "call_tools_react"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lang = state.get("language") or getattr(config.profile, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = get_model(config, lang, preferred_model=preferred_model)

            tools = [c["tool"] for c in candidates]
            tool_lookup = {getattr(t, "name", ""): t for t in tools}

            tool_descs = []
            for tool in tools:
                name = getattr(tool, "name", "unknown")
                desc = getattr(tool, "description", "").split("\n")[0]
                tool_descs.append(f"- {name}: {desc}")
            tool_block = "\n".join(tool_descs)

            react_prompt = (
                "Answer the user's question using the available tools.\n"
                "Use this format:\n\n"
                "Thought: <your reasoning>\n"
                'Action: {"tool": "<tool_name>", "args": {<arguments>}}\n'
                "Observation: <tool result will be inserted here>\n"
                "... (repeat if needed)\n"
                "Final Answer: <your answer to the user>\n\n"
                f"Available tools:\n{tool_block}"
            )

            messages = [
                SystemMessage(content=react_prompt),
                HumanMessage(content=user_msg),
            ]

            tool_results: Dict[str, str] = state.get("tool_results", {})
            tool_execution_results: Dict[str, Dict[str, Any]] = state.get("tool_execution_results", {})
            routing_metadata: Dict[str, str] = state.get("routing_metadata", {})

            for iteration in range(max_iterations):
                response = model.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)

                # Check for Final Answer
                final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
                if final_match:
                    logger.info("ReAct loop completed with final answer", iterations=iteration + 1)
                    break

                # Parse Action line
                action_marker = re.search(r'Action:\s*', response_text, re.DOTALL)
                action = extract_json_object(response_text[action_marker.end():]) if action_marker else None
                if not action:
                    if action_marker:
                        logger.warning("Failed to parse ReAct action JSON", iteration=iteration)
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(content="Observation: Invalid JSON in Action. Please fix the format."))
                        continue
                    logger.info("No action found in ReAct response, treating as final answer", iteration=iteration)
                    break

                tool_name = action.get("tool", "")
                tool_args = action.get("args", {})

                _requested_results = (state.get("prefilter_intents") or {}).get("requested_web_results")
                tool_args = apply_web_search_max_results(tool_name, tool_args, _requested_results)

                tool_obj = tool_lookup.get(tool_name)

                if not tool_obj:
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(
                        content=f"Observation: Tool '{tool_name}' not found. Available: {', '.join(tool_lookup.keys())}"
                    ))
                    continue

                # Strip unknown fields before calling _run(); a schema mismatch is left for
                # the tool's own error path (react ignores the validation error).
                tool_args, _ = coerce_tool_args(tool_obj, tool_args)

                try:
                    result = tool_obj._run(**tool_args)
                    record_tool_success(
                        tool_name=tool_name,
                        result=result,
                        reason=f"react tool call (iteration {iteration})",
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
                        routing_metadata=routing_metadata,
                        args=tool_args,
                    )
                    observation = str(result)
                    logger.info("Tool executed (react)", tool=tool_name, iteration=iteration)
                except Exception as exec_err:
                    record_tool_error(
                        tool_name=tool_name,
                        error=exec_err,
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
                        args=tool_args,
                    )
                    observation = f"Error: {exec_err}"

                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=f"Observation: {observation}"))

            state["tool_results"] = tool_results
            state["tool_execution_results"] = tool_execution_results
            state["routing_metadata"] = routing_metadata
            logger.info("ReAct tool calling completed", tools_executed=len(tool_results))

        except Exception as e:
            logger.error("ReAct tool calling failed", error=str(e), exc_info=True)

    return state


def node_load_memory(state: GraphState) -> GraphState:
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    features_config = load_features_config()
    
    logger.info("Loading memory", user_id=state.get("user_id"), profile_name=profile_name)

    if not getattr(features_config, "long_term_memory", False):
        logger.info("Long-term memory disabled via features flag", profile_name=profile_name)
        state["loaded_memory"] = []
        return state
    
    # Get memory feature flags
    include_related = getattr(features_config, "memory_include_related", False)
    top_k = getattr(features_config, "memory_top_k", 5)
    
    logger.info(
        "Memory features",
        include_related=include_related,
        top_k=top_k,
        profile_name=profile_name
    )
    
    with track_node_execution(profile_name, "load_memory"):
        backend = build_memory_backend(config)
        try:
            result = load_memory_node(
                state, 
                backend=backend,
                top_k=top_k,
                include_related=include_related
            )
            track_memory_operation(profile_name, "load", "success")
            
            # Log analytics if available
            analytics = result.get("memory_analytics", {})
            logger.info(
                "Memory loaded",
                user_id=state.get("user_id"),
                primary_count=analytics.get("primary_count", 0),
                related_count=analytics.get("related_count", 0),
                total_count=analytics.get("total_count", 0)
            )
            return result
        except Exception as e:
            track_memory_operation(profile_name, "load", "error")
            logger.error("Memory load failed", error=str(e), user_id=state.get("user_id"))
            raise


def node_generate_response(state: GraphState) -> GraphState:
    config = load_core_config()
    lang = state.get("language") or config.profile.language
    profile_name = getattr(config.profile, "name", "default-profile")
    
    # Check for user's preferred model
    user_settings = state.get("user_settings", {})
    preferred_model = user_settings.get("preferred_model")
    
    logger.info(
        "Generating response",
        language=lang,
        profile_name=profile_name,
        preferred_model=preferred_model or "default"
    )

    model = get_model(config, lang, preferred_model=preferred_model)
    provider, model_name = resolve_initial_model_metadata(config, lang, preferred_model)

    # Find the last user-role message so crew-appended assistant messages don't pollute user_msg.
    user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break
    
    # Build context-aware prompt. Resolution order: profile prompt files → emergency fallback.
    prompts_cfg = getattr(config, "prompts", None)
    configured_prompt = prompts_cfg.get_prompt(lang, "response_system", fallback_lang="en") if prompts_cfg else None
    system_prompt = configured_prompt or (
        "You are a helpful AI assistant.\n"
        "Answer the user's question clearly and factually.\n"
        "Use the knowledge base and available tools when relevant to provide better answers."
    )

    # Enforce response language to avoid drift into unintended languages.
    language_instruction = (
        (prompts_cfg.get_prompt(lang, "language_enforcement", fallback_lang="en") if prompts_cfg else None)
        or f"Respond exclusively in language code '{lang}'."
    )
    system_prompt += f"\n\n=== RESPONSE LANGUAGE ===\n{language_instruction}\n=== END RESPONSE LANGUAGE ===\n"

    system_prompt += f"\n[Today: {datetime.date.today().isoformat()}. Treat this as 'now' for any recent/current queries.]"

    # Add only enabled tools to system prompt (skip in three-tier mode — Layer 2 already handled tools)
    loaded_tools = state.get("loaded_tools", [])
    user_settings = state.get("user_settings", {})
    tool_toggles = user_settings.get("tool_toggles", {})
    tool_calling_mode = state.get("tool_calling_mode")
    
    if loaded_tools and not tool_calling_mode:
        # Legacy path: no three-tier architecture active, inject tool catalog + suppression
        enabled_tools = [
            tool for tool in loaded_tools
            if tool_toggles.get(tool.name, True)
        ]
        
        if enabled_tools:
            _default_tooling = (
                "The orchestration layer handles tool execution.\n"
                "Do not emit tool calls, function calls, or control tokens.\n"
                "Never output strings like <|tool_call_start|>, <|tool_call_end|>, or JSON function-call payloads.\n"
                "You must return plain natural-language text only."
            )
            system_prompt += f"\n\n=== TOOLING MODE ===\n{_default_tooling}\n=== END TOOLING MODE ===\n"
            
            # Add tool catalog so model understands capabilities
            tool_catalog_lines = ["\n\n=== AVAILABLE TOOLS ==="]
            for tool in enabled_tools:
                tool_name = getattr(tool, "name", "unknown")
                tool_desc = getattr(tool, "description", "")
                if tool_desc:
                    # Extract first line of description for conciseness
                    desc_first_line = tool_desc.split("\n")[0].strip()
                    tool_catalog_lines.append(f"• {tool_name}: {desc_first_line}")
            tool_catalog_lines.append("=== END AVAILABLE TOOLS ===\n")
            system_prompt += "\n".join(tool_catalog_lines)
    
    # Add knowledge base context if available (RAG)
    import re as _re_urls
    knowledge_context = state.get("knowledge_context", [])
    allowed_sources = []
    allowed_urls = set()
    # Seed with URLs the user themselves pasted — echoing a user's own link back must not
    # be scrubbed to "source omitted" by the anti-hallucination filter below.
    for _u in _re_urls.findall(r"https?://[^\s)]+", user_msg or ""):
        allowed_urls.add(_u.rstrip(".,;:!?"))
    collected_sources: List[Dict[str, Any]] = []  # structured source tracking
    context_text = ""  # RAG knowledge text; populated below when knowledge_context is present

    # Utility-only tool responses (datetime/calculator/file ops) should not force citation-style footnotes.
    tool_results = state.get("tool_results", {})
    tool_execution_results = state.get("tool_execution_results", {})
    routing_metadata = state.get("routing_metadata", {})
    utility_tool_names = {"datetime_tool", "calculator_tool"}
    used_tool_names = set(tool_results.keys())
    utility_only_response = bool(used_tool_names) and used_tool_names.issubset(utility_tool_names)

    if knowledge_context:
        context_text = "\n\n".join([
            f"[Quelle: {_rag_label(doc.get('file_name'))}]\n{doc.get('text', '')}"
            for doc in knowledge_context[:3]  # Top 3 results
        ])
        system_prompt += f"\n\n=== WISSENSDATENBANK ===\n{context_text}\n=== ENDE WISSENSDATENBANK ===\n"
        _seen_rag_labels = set()
        for doc in knowledge_context[:3]:
            file_name = doc.get("file_name")
            file_path = doc.get("file_path")
            if file_name:
                allowed_sources.append(str(file_name))
            if file_path and file_path != file_name:
                allowed_sources.append(str(file_path))
            # Extract URLs from RAG document text so they survive _filter_urls
            doc_text = doc.get("text", "")
            for url in _re_urls.findall(r"https?://[^\s)]+", str(doc_text)):
                allowed_urls.add(url)
            if not utility_only_response:
                label = _rag_label(file_name)
                if label not in _seen_rag_labels:
                    _seen_rag_labels.add(label)
                    collected_sources.append({"type": "rag", "label": label, "url": None})
    
    # Add past memories/conversation context if available
    loaded_memory = state.get("loaded_memory", [])
    web_tools_used = any(
        name in tool_results for name in ("extract_webpage_mcp", "web_search_mcp")
    )
    _memory_block = build_memory_context_block(loaded_memory, web_tools_used)
    if _memory_block:
        system_prompt += _memory_block
        logger.info(
            "Loaded memory injected into context",
            memory_count=len(loaded_memory),
            low_priority=web_tools_used,
        )

    # Add user-provided uploaded attachments as explicitly labeled reference context.
    attachments = state.get("attachments") or []
    attachment_block, normalized_attachment_context = build_attachment_context_block(attachments)
    if attachment_block:
        system_prompt += f"\n\n{attachment_block}"
        state["attachment_context"] = normalized_attachment_context
        logger.info(
            "Attachment context injected into prompt",
            attachment_count=len(normalized_attachment_context),
        )
        # Track attachment injection metric
        from universal_agentic_framework.monitoring.metrics import ATTACHMENTS_INJECTED_TOTAL
        profile_name = state.get("profile_name", "unknown")
        ATTACHMENTS_INJECTED_TOTAL.labels(profile_name=profile_name).inc()
    else:
        state["attachment_context"] = []
        # Track requests with no attachments
        from universal_agentic_framework.monitoring.metrics import ATTACHMENTS_NONE_TOTAL
        profile_name = state.get("profile_name", "unknown")
        ATTACHMENTS_NONE_TOTAL.labels(profile_name=profile_name).inc()

    # Add resolved workspace documents as explicitly labeled reference context.
    workspace_documents = state.get("workspace_documents") or []
    workspace_block, normalized_workspace_context = build_workspace_document_context_block(workspace_documents)
    if workspace_block:
        system_prompt += f"\n\n{workspace_block}"
        state["workspace_document_context"] = normalized_workspace_context
        logger.info(
            "Workspace document context injected into prompt",
            workspace_document_count=len(normalized_workspace_context),
        )
    else:
        state["workspace_document_context"] = []

    # Add tool results if available (semantic routing)
    if tool_results:
        system_prompt += build_tool_results_block(
            tool_results, tool_execution_results, routing_metadata
        )
        logger.info("Tool results injected into context", tools_count=len(tool_results))

        # Extract URLs from tool results so citations can be limited to known sources
        for result in tool_results.values():
            for url in re.findall(r"https?://[^\s)]+", str(result)):
                allowed_urls.add(url)

        # Collect structured web sources from tool results
        for _tr_name, _tr_result in tool_results.items():
            for _ws_url in re.findall(r"https?://[^\s)]+", str(_tr_result)):
                # Derive a short label from the URL hostname
                try:
                    from urllib.parse import urlparse
                    _host = urlparse(_ws_url).hostname or _ws_url
                    _label = _host.removeprefix("www.")
                except Exception:
                    _label = _ws_url[:40]
                collected_sources.append({"type": "web", "label": _label, "url": _ws_url})

        has_citable_sources = bool(collected_sources)

        # Inject synthesis instruction so the LLM writes a summary, not a raw list (verbatim
        # relay for OCR/barcode/metadata/structured tools is handled inside the helper).
        synthesis_text = select_synthesis_instruction(
            used_tool_names, has_citable_sources, getattr(config, "prompts", None), lang
        )
        system_prompt += f"\n\n=== SYNTHESIS INSTRUCTION ===\n{synthesis_text}\n=== END SYNTHESIS INSTRUCTION ===\n"

        # Number sources and inject as a reference list for the LLM
        if collected_sources:
            source_lines = []
            for idx, src in enumerate(collected_sources, 1):
                src["index"] = idx  # store the number for frontend mapping
                if src.get("url"):
                    source_lines.append(f"[{idx}] {src['label']} - {src['url']}")
                else:
                    source_lines.append(f"[{idx}] {src['label']} (knowledge base)")
            system_prompt += "\n\n=== SOURCES ===\n" + "\n".join(source_lines) + "\n=== END SOURCES ===\n"

    if allowed_sources or allowed_urls:
        allowed_lines = [f"- {src}" for src in allowed_sources]
        allowed_lines.extend([f"- {url}" for url in sorted(allowed_urls)])
        allowed_block = "\n".join(allowed_lines)
        sources_instruction = (
            "Wenn du Quellen nennst, nutze NUR Eintraege aus ALLOWED_SOURCES. "
            "Erfinde keine neuen Links oder Quellen."
        )
        system_prompt += (
            "\n\n=== ALLOWED_SOURCES ===\n"
            f"{allowed_block}\n"
            "=== ENDE ALLOWED_SOURCES ===\n"
            f"{sources_instruction}"
        )

    # Add crew results as system-level context so the LLM synthesizes a fresh response
    # (crew results must NOT appear as prior assistant turns — that causes the model to
    # just add a short follow-up instead of producing a full answer).
    _crew_result_prefixes = _CREW_RESULT_PREFIXES
    crew_results = state.get("crew_results", {})
    _crew_block = build_crew_findings_block(crew_results)
    if _crew_block:
        system_prompt += _crew_block
        logger.info(
            "Crew results injected into context",
            crew_count=sum(
                1 for c in crew_results.values()
                if isinstance(c, dict) and c.get("success") and c.get("result")
            ),
        )

    workspace_docs_raw = state.get("workspace_documents") or []
    workspace_document_context = state.get("workspace_document_context") or []
    # Key off the document the adapter actually selected for writeback (exactly one
    # eligible text doc) so the prompt injection and the adapter's save gate agree.
    writeback_doc_state = state.get("workspace_writeback_document")
    if state.get("workspace_writeback_requested") and writeback_doc_state:
        filename = writeback_doc_state.get("filename") or "document.txt"
        writeback_mime = str(writeback_doc_state.get("mime_type") or "")
        is_csv_writeback = writeback_mime == "text/csv" or filename.endswith(".csv")
        csv_extra = (
            "\nOutput raw CSV only — preserve the exact delimiter, the header row, and every column; "
            "do not convert to a markdown table or add code fences."
            if is_csv_writeback
            else ""
        )
        system_prompt += (
            "\n\n=== WORKSPACE WRITEBACK MODE ===\n"
            f"The user asked you to update and save the workspace document '{filename}'.\n"
            "Structure your response EXACTLY as two sections:\n\n"
            "SUMMARY:\n"
            "<One or two sentences describing what you changed and why.>\n\n"
            "DOCUMENT:\n"
            "<Complete revised file content — no code fences, no preamble, no commentary.>\n\n"
            f"IMPORTANT: The DOCUMENT section is saved verbatim. Do not wrap it in code fences.{csv_extra}\n"
            "=== END WORKSPACE WRITEBACK MODE ==="
        )
    
    # Build messages in proper chat format for LangChain
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    all_msgs = state.get("messages") or []

    # Inject compression digests (role="system", type="summary") into the system prompt
    # rather than emitting mid-conversation SystemMessages — many providers only honour a
    # single leading system turn. Without this the summary that compression produced would
    # never reach the model and compressed history would silently lose context.
    summary_blocks = [
        str(msg.get("content") or "").strip()
        for msg in all_msgs
        if msg.get("type") == "summary" and str(msg.get("content") or "").strip()
    ]
    if summary_blocks:
        system_prompt += (
            "\n\n=== CONVERSATION SUMMARY (earlier messages) ===\n"
            + "\n\n".join(summary_blocks)
            + "\n=== END CONVERSATION SUMMARY ===\n"
        )

    # Start with system prompt
    messages = [SystemMessage(content=system_prompt)]

    # Add full conversation history (all previous user and assistant messages).
    # Skip crew-appended assistant messages — their content is already injected into
    # the system prompt as FINDINGS context above.  Keeping them as AIMessage turns
    # causes the model to treat the crew output as its own prior answer and only add
    # a short follow-up instead of synthesising a full response.
    for msg in all_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if msg.get("type") == "summary":
            continue  # already folded into the system prompt above
        if role == "assistant" and any(content.startswith(p) for p in _crew_result_prefixes):
            continue  # already in system prompt as FINDINGS context
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Some models underweight system-side attachment blocks and claim they cannot
    # access attachments. Mirror attachment context as an explicit turn-scoped
    # human context message to improve reliability.
    attachment_context = state.get("attachment_context") or []
    if attachment_context:
        attachment_lines = [
            "The following attachment content is available for this turn. Use it directly when answering.",
        ]
        for item in attachment_context:
            name = item.get("original_name") or "attachment.txt"
            text = item.get("text") or ""
            if text:
                attachment_lines.append(f"[Attachment: {name}]\n{text}")
        if len(attachment_lines) > 1:
            messages.append(HumanMessage(content="\n\n".join(attachment_lines)))

    if workspace_document_context:
        workspace_lines = [
            "The following workspace documents are available for this turn. Use them directly when answering.",
        ]
        for item in workspace_document_context:
            filename = item.get("filename") or "document.txt"
            version = item.get("version")
            text = item.get("text") or ""
            if not text:
                continue
            label = f"[Workspace Document: {filename}"
            if version is not None:
                label += f" | v{version}"
            label += "]"
            workspace_lines.append(f"{label}\n{text}")
        if len(workspace_lines) > 1:
            messages.append(HumanMessage(content="\n\n".join(workspace_lines)))

    # In writeback mode inject the FULL document content (bypasses the 600-token context truncation)
    writeback_doc = state.get("workspace_writeback_document")
    if writeback_doc and state.get("workspace_writeback_requested"):
        full_text = str(writeback_doc.get("content_text") or "").strip()
        fname = writeback_doc.get("filename") or "document"
        ver = writeback_doc.get("version")
        if full_text:
            ver_label = f" | v{ver}" if ver is not None else ""
            messages.append(HumanMessage(
                content=f"[Document to revise: {fname}{ver_label}]\n\n{full_text}"
            ))

    tool_results = state.get("tool_results", {})

    # Approximate per-section prompt size for the context-window menu. The headline ring
    # still uses the provider-reported total; these are cheap chars/4 estimates so the
    # user can see *where* their context is going (instructions+RAG+memory+tools vs.
    # prior turns vs. this message vs. attached files). Estimates only — never summed
    # against the authoritative total.
    def _ctx_attachment_tokens() -> int:
        total = 0
        for item in attachment_context or []:
            total += estimate_tokens(str(item.get("text") or ""))
        for item in workspace_document_context or []:
            total += estimate_tokens(str(item.get("text") or ""))
        if writeback_doc and state.get("workspace_writeback_requested"):
            total += estimate_tokens(str(writeback_doc.get("content_text") or ""))
        return total

    _history_tokens = 0
    for _m in all_msgs:
        if _m.get("type") == "summary":
            continue
        _c = _m.get("content", "")
        if _m.get("role") == "user" and _c == user_msg:
            continue  # counted under "user" below
        if isinstance(_c, str):
            _history_tokens += estimate_tokens(_c)
    state["context_breakdown"] = {
        "system": estimate_tokens(system_prompt),
        "history": _history_tokens,
        "user": estimate_tokens(user_msg),
        "attachments": _ctx_attachment_tokens(),
    }

    logger.info("Sending to LLM",
                system_prompt_length=len(system_prompt),
                user_msg_length=len(user_msg),
                knowledge_context_docs=len(knowledge_context),
                messages_count=len(messages),
                workspace_document_context_docs=len(workspace_document_context),
                tools_executed=len(tool_results))

    with track_node_execution(profile_name, "respond"):
        _usage_meta: Optional[dict] = None
        try:
            response_text, provider, model_name, active_model, _usage_meta = _invoke_with_model_fallback(
                config=config,
                language=lang,
                payload=messages,
                initial_model=model,
                initial_provider=provider,
                initial_model_name=model_name,
                preferred_model=preferred_model,
            )
            model = active_model

            # Validate that we actually got a response object, not the input messages
            if isinstance(response_text, list):
                logger.error("LLM returned input messages instead of response", out_type=type(response_text).__name__)
                raise ValueError("LLM returned unexpected list instead of response message")

            logger.info("LLM response received", response_length=len(response_text), response_preview=response_text[:200])
            track_llm_call(profile_name, provider, model_name, "success")
        except _ModelInvokeError as e:
            error_type = getattr(e, "error_type", "error")
            track_llm_call(profile_name, e.provider, e.model_name, error_type)
            log_fn = logger.error if error_type in ("error", "auth_error") else logger.warning
            log_fn(
                "LLM call failed",
                error=str(e),
                exc_info=(error_type in ("error", "auth_error")),
                provider=e.provider,
                model_name=e.model_name,
                error_type=error_type,
            )
            provider = e.provider
            model_name = e.model_name
            if error_type == "context_window_exceeded":
                response_text = "The request is too long for this model. Please shorten your message or start a new conversation."
            elif error_type == "rate_limit":
                response_text = "The model is temporarily unavailable due to rate limits. Please try again in a moment."
            elif error_type == "auth_error":
                response_text = "There is a configuration error with the AI provider. Please contact the administrator."
            elif error_type == "service_unavailable":
                response_text = "The AI model is currently unavailable. Please try again shortly."
            else:
                response_text = f"LLM Error: {str(e)[:100]}"
        except Exception as e:
            track_llm_call(profile_name, provider, model_name, "error")
            logger.error("LLM call failed", error=str(e), exc_info=True, provider=provider, model_name=model_name)
            response_text = f"LLM Error: {str(e)[:100]}"

        # Remove leaked tool-call/control tokens from model output
        if response_text:
            original_text = response_text
            response_text = strip_control_tokens(response_text)

            if response_text != original_text:
                if tool_calling_mode == "native":
                    logger.warning(
                        "Control tokens leaked in native tool-calling mode — possible LM Studio parsing issue",
                    )
                    record_runtime_native_tool_leak(config, state, model_name)
                else:
                    logger.info("Sanitized control tokens from LLM response")

            # Post-generation guardrails: empty-response synthesis retry, attachment / web-extract
            # contradiction retries, then a tool-based fallback so the user never gets a blank reply.
            response_text = retry_synthesis_if_empty(
                response_text,
                model=model,
                lang=lang,
                user_msg=user_msg,
                knowledge_context=knowledge_context,
                context_text=context_text,
                tool_results=tool_results,
                collected_sources=collected_sources,
            )
            response_text = retry_on_attachment_refusal(
                response_text,
                model=model,
                state=state,
                system_prompt=system_prompt,
                all_msgs=all_msgs,
            )
            response_text = retry_on_web_extract_contradiction(
                response_text,
                model=model,
                tool_results=tool_results,
                user_msg=user_msg,
            )
            response_text = format_tool_based_fallback(
                response_text,
                tool_results=tool_results,
                lang=lang,
            )

        # Remove any URLs that were not present in tool results / RAG / the user message.
        if response_text:
            response_text, _urls_removed = filter_untrusted_urls(response_text, allowed_urls)
            if _urls_removed:
                logger.info("Removed untrusted URLs from response", removed=_urls_removed)

        # Honor strict literal-response directives when explicitly requested.
        exact_reply = extract_exact_reply_directive(user_msg)
        if exact_reply and response_text != exact_reply:
            logger.info(
                "Enforcing exact reply directive",
                requested=exact_reply,
                original_preview=(response_text or "")[:120],
            )
            response_text = exact_reply

        # Estimate prompt tokens from the messages list so the fallback is never 0
        # (used when the provider does not report usage_metadata in this turn).
        _prompt_estimate = estimate_tokens(" ".join(
            m.content if isinstance(m.content, str)
            else " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in m.content
            )
            for m in messages
        ))
        actual_input_tokens, output_tokens = _tokens_from_usage(_usage_meta, response_text, _prompt_estimate)
        node_tokens = actual_input_tokens + output_tokens

        tokens_used = (state.get("tokens_used") or 0) + node_tokens

        track_tokens(profile_name, model_name, "respond", node_tokens)

        state["tokens_used"] = tokens_used
        state["turn_tokens_used"] = (state.get("turn_tokens_used") or 0) + node_tokens
        state["input_tokens"] = (state.get("input_tokens") or 0) + actual_input_tokens
        state["output_tokens"] = (state.get("output_tokens") or 0) + output_tokens
        # Per-inference snapshot (overwritten each turn) — the context-window indicator
        # reads this so it reflects the current prompt size, not the cumulative lifetime sum.
        state["last_input_tokens"] = actual_input_tokens
        state["last_output_tokens"] = output_tokens
        state["provider_used"] = provider
        state["model_used"] = model_name
        state["sources"] = collected_sources
        # Append assistant message
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": response_text})
        state["messages"] = msgs
        
        logger.info(
            "Response generated",
            tokens=actual_input_tokens + output_tokens,
            total_tokens=tokens_used,
            model_used=model_name
        )

    return state


def node_summarize(state: GraphState) -> GraphState:
    config = load_core_config()
    lang = state.get("language") or config.profile.language
    profile_name = getattr(config.profile, "name", "default-profile")
    features = load_features_config()
    logger.info("Summarizing conversation", profile_name=profile_name)

    # Skip the auxiliary-model fact-extraction entirely when this turn won't be persisted
    # (memory disabled or a trivial exchange). node_update_memory would discard the result
    # anyway, so producing it just burns an LLM call. node_update_memory keeps its own
    # meaningful exchange-based fallback summary for the rare case it does write.
    if not _should_write_memory(state, features):
        logger.info("Skipping summarize LLM call (memory write not warranted)", profile_name=profile_name)
        state["summary_text"] = ""
        return state

    # Keep digest_chain normalized even when compression does not generate a new summary.
    try:
        if getattr(features, "memory_digest_chain_enabled", True):
            summarizer = get_summarizer()
            existing_chain = state.get("digest_chain") or []
            extracted_chain = summarizer.extract_digest_chain(
                state.get("messages", []),
                max_items=_DIGEST_CHAIN_MAX_ITEMS,
            )
            state["digest_chain"] = _merge_digest_chains(
                existing_chain,
                extracted_chain,
                max_items=_DIGEST_CHAIN_MAX_ITEMS,
            )
        else:
            state["digest_chain"] = []
    except Exception as e:
        logger.warning("digest_chain_normalization_failed", error=str(e))

    model, provider, model_name = get_auxiliary_model(config, lang)

    # Build a window of the last 3 user+assistant exchanges for richer fact extraction.
    _msgs = state.get("messages", [])
    _exchange_pairs: list[tuple[str, str]] = []
    _i = len(_msgs) - 1
    while _i >= 0 and len(_exchange_pairs) < 3:
        if _msgs[_i].get("role") == "assistant":
            _asst = _msgs[_i].get("content", "")
            _j = _i - 1
            while _j >= 0 and _msgs[_j].get("role") != "user":
                _j -= 1
            if _j >= 0:
                _exchange_pairs.append((_msgs[_j].get("content", ""), _asst))
                _i = _j - 1
            else:
                break
        else:
            _i -= 1
    _exchange_pairs.reverse()
    _exchange_text = "\n\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in _exchange_pairs
    ) if _exchange_pairs else ""
    prompt = (
        "Extract facts about the user from this conversation.\n"
        "Focus ONLY on what the USER stated: preferences, personal information, goals, "
        "constraints, corrections, and context they provided.\n"
        "Ignore assistant explanations. If no user facts are present, respond with 'No user facts'.\n"
        "Respond with a single concise statement.\n\n"
        f"{_exchange_text}"
    )

    input_tokens = count_tokens_for_model(model_name, prompt)

    with track_node_execution(profile_name, "summarize"):
        try:
            from langchain_core.messages import HumanMessage as _HumanMessage
            out = model.invoke([_HumanMessage(content=prompt)])
            _raw_content = out.content
            if isinstance(_raw_content, list):
                _raw_content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in _raw_content
                )
            # Empty (not the prompt echo) on a blank reply: node_update_memory then builds
            # a meaningful "User: … -> Assistant: …" fallback instead of persisting the
            # extraction instruction + raw exchange as if it were a user fact.
            summary = (_raw_content or "").strip()
            _sum_usage_meta = getattr(out, "usage_metadata", None)
            track_llm_call(profile_name, provider, model_name, "success")
        except Exception as e:
            track_llm_call(profile_name, provider, model_name, "error")
            logger.warning("Summarization LLM call failed, using fallback", error=str(e))
            summary = ""
            _sum_usage_meta = None

        _, output_tokens = _tokens_from_usage(_sum_usage_meta, summary)
        node_tokens = input_tokens + output_tokens

        tokens_used = (state.get("tokens_used") or 0) + node_tokens

        track_tokens(profile_name, model_name, "summarize", node_tokens)
        
        state["tokens_used"] = tokens_used
        state["turn_tokens_used"] = (state.get("turn_tokens_used") or 0) + node_tokens
        state["summary_text"] = summary
        
        logger.info(
            "Summary generated",
            tokens=node_tokens,
            summary_length=len(summary)
        )
        
        return state


def node_update_memory(state: GraphState) -> GraphState:
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    features_config = load_features_config()
    
    user_id = state.get("user_id")
    logger.info("Updating memory", user_id=user_id, profile_name=profile_name)

    # Skip writes for disabled memory or trivial exchanges (shared gate with node_summarize,
    # which already skipped producing a summary for the same turns).
    if not _should_write_memory(state, features_config):
        logger.info("Skipping memory update (memory disabled or trivial exchange)", user_id=user_id)
        return state

    _current_user_msg = _latest_user_message(state.get("messages", []))

    logger.info("Building memory backend for upsert", user_id=user_id)
    backend = build_memory_backend(config)
    logger.info("Memory backend built", backend_type=type(backend).__name__, user_id=user_id)

    # Build structured exchange messages (last 3 turns) for Mem0's native inference.
    _msgs = state.get("messages", [])
    _exchange_messages: list[dict] = []
    _exchange_count = 0
    _i = len(_msgs) - 1
    while _i >= 0 and _exchange_count < 3:
        _m = _msgs[_i]
        _role = _m.get("role") if isinstance(_m, dict) else None
        if _role in ("user", "assistant"):
            _exchange_messages.insert(0, {"role": _role, "content": _m.get("content", "")})
            if _role == "user":
                _exchange_count += 1
        _i -= 1

    summary_text = state.get("summary_text")
    if not summary_text:
        # Fallback: build summary from current exchange using correct message selection.
        _asst_fallback = ""
        for _m in reversed(_msgs):
            if isinstance(_m, dict) and _m.get("role") == "assistant":
                _asst_fallback = _m.get("content", "")
                break
        summary_text = f"Summary: {_current_user_msg} -> {_asst_fallback}"

    logger.info("Summary text prepared", summary_length=len(summary_text), user_id=user_id)

    update_tokens = estimate_tokens(summary_text)

    with track_node_execution(profile_name, "update_memory"):
        try:
            digest_chain = state.get("digest_chain") or []
            if not getattr(features_config, "memory_digest_chain_enabled", True):
                digest_chain = []
            latest_digest = digest_chain[0] if digest_chain else None
            metadata = {"type": "summary"}
            if latest_digest:
                metadata["digest_id"] = latest_digest.get("digest_id")
                metadata["previous_digest_id"] = latest_digest.get("previous_digest_id")
            if digest_chain:
                metadata["digest_chain_ids"] = [
                    d.get("digest_id") for d in digest_chain if isinstance(d, dict) and d.get("digest_id")
                ]
                metadata["digest_chain_length"] = len(digest_chain)

            logger.info("Calling update_memory_node", user_id=user_id, text_length=len(summary_text))
            result = update_memory_node(
                state,
                text=summary_text,
                metadata=metadata,
                backend=backend,
                messages=_exchange_messages if _exchange_messages else None,
                digest_chain=digest_chain,
            )
            result["tokens_used"] = (result.get("tokens_used") or 0) + update_tokens
            result["turn_tokens_used"] = (result.get("turn_tokens_used") or 0) + update_tokens
            track_memory_operation(profile_name, "update", "success")
            logger.info("Memory updated successfully", user_id=user_id)
            return result
        except Exception as e:
            track_memory_operation(profile_name, "update", "error")
            logger.error("Memory update failed", error=str(e), user_id=user_id, exc_info=True)
            # Don't re-raise - allow conversation to continue even if memory update fails
            return state


def build_graph() -> StateGraph:
    """Build a LangGraph with three-tier tool selection architecture.

    Layer 1: Semantic pre-filter (always runs) — scores tools, returns top-K candidates
    Layer 2: Model-driven tool calling (native | structured | react per provider config)
    Layer 3: Output validation + retry (built into Layer 2 nodes)

    Plus: crew routing, memory, RAG retrieval, response generation, summarization.
    """
    # Initialize performance infrastructure
    try:
        _perf_config = load_core_config()
        initialize_performance_nodes(llm_factory=LLMFactory(_perf_config))
        logger.info("Performance nodes initialized (caching + compression)")
    except Exception as e:
        logger.warning(f"Performance nodes initialization failed: {e}, continuing without caching")
    
    graph = StateGraph(GraphState)

    # --- Pass-through convergence node after Layer 2 tool calling ---
    def _after_tool_call(state: GraphState) -> GraphState:
        return state
    
    # Add core nodes
    graph.add_node("load_tools", node_load_tools)
    graph.add_node("prefilter_tools", node_prefilter_tools)  # Layer 1
    graph.add_node("call_tools_native", node_call_tools_native)  # Layer 2 native
    graph.add_node("call_tools_structured", node_call_tools_structured)  # Layer 2 structured
    graph.add_node("call_tools_react", node_call_tools_react)  # Layer 2 react
    graph.add_node("after_tool_call", _after_tool_call)  # Convergence point
    graph.add_node("research_crew", node_research_crew)
    graph.add_node("analytics_crew", node_analytics_crew)
    graph.add_node("code_generation_crew", node_code_generation_crew)
    graph.add_node("planning_crew", node_planning_crew)
    graph.add_node("crew_chain", node_crew_chain)
    graph.add_node("crew_parallel", node_crew_parallel)
    graph.add_node("load_memory", node_load_memory)
    graph.add_node("retrieve_knowledge", node_retrieve_knowledge)
    graph.add_node("respond", node_generate_response)
    graph.add_node("summarize", node_summarize)
    graph.add_node("update_memory", node_update_memory)
    
    # Add performance optimization nodes. Registered as native async coroutines so
    # langgraph awaits them on the event loop under ainvoke — no per-call worker thread
    # or nested asyncio.run (the old *_sync wrappers existed only to bridge sync invoke).
    graph.add_node("memory_query_cache", memory_query_cache_node)
    graph.add_node("memory_cache_store", memory_cache_store_node)
    graph.add_node("compress_conversation", conversation_compression_node)
    graph.add_node("cache_stats", cache_stats_node)

    # --- Execution flow ---
    # Crew routing happens first — before tools load — so crew-destined queries skip
    # the tool pipeline entirely and go directly to the crew node.
    from universal_agentic_framework.orchestration.crew_nodes import (
        route_to_research_crew,
        route_to_analytics_crew,
        route_to_code_generation_crew,
        route_to_planning_crew,
    )

    def _route_start(state):
        """Route at START: crew queries bypass tool execution."""
        if route_to_research_crew(state):
            return "research_crew"
        elif route_to_analytics_crew(state):
            return "analytics_crew"
        elif route_to_code_generation_crew(state):
            return "code_generation_crew"
        elif route_to_planning_crew(state):
            return "planning_crew"
        return "load_tools"

    graph.add_conditional_edges(
        START,
        _route_start,
        {
            "research_crew": "research_crew",
            "analytics_crew": "analytics_crew",
            "code_generation_crew": "code_generation_crew",
            "planning_crew": "planning_crew",
            "load_tools": "load_tools",
        },
    )
    graph.add_edge("load_tools", "prefilter_tools")

    # Layer 2: Route to appropriate tool-calling strategy based on prefilter results
    def route_tool_strategy(state):
        """Route to the appropriate tool-calling strategy based on resolved mode.
        
        Logs comprehensive mode routing information for debugging and monitoring.
        """
        candidates = state.get("candidate_tools", [])
        if not candidates:
            logger.info("Tool strategy routing: no candidates, skipping tool calling")
            return "no_tools"
            
        mode = state.get("tool_calling_mode", "structured")
        mode_reason = state.get("tool_calling_mode_reason", "unknown")
        
        if mode not in ("native", "structured", "react"):
            logger.warning(
                "Tool strategy routing: invalid mode, falling back to structured",
                invalid_mode=mode,
                candidates=len(candidates),
                mode_reason=mode_reason,
            )
            return "structured"

        logger.info(
            "Tool strategy routing: routing to strategy node",
            mode=mode,
            mode_reason=mode_reason,
            candidates=len(candidates),
            candidate_names=[c.get("name", "unknown") for c in candidates],
        )
        return mode

    graph.add_conditional_edges(
        "prefilter_tools",
        route_tool_strategy,
        {
            "native": "call_tools_native",
            "structured": "call_tools_structured",
            "react": "call_tools_react",
            "no_tools": "after_tool_call",
        },
    )

    # All Layer 2 nodes converge
    graph.add_edge("call_tools_native", "after_tool_call")
    graph.add_edge("call_tools_structured", "after_tool_call")
    graph.add_edge("call_tools_react", "after_tool_call")

    # after_tool_call always proceeds to memory. Crew queries are routed to their crew at
    # START (bypassing tools entirely), so a query only reaches after_tool_call when no crew
    # matched there; tool nodes don't change messages[-1], so re-running the crew routers here
    # would always return "no crew" (W1.7: removed that dead conditional re-check).
    graph.add_edge("after_tool_call", "memory_query_cache")

    # All crews flow to memory cache for further processing
    graph.add_edge("research_crew", "memory_query_cache")
    graph.add_edge("analytics_crew", "memory_query_cache")
    graph.add_edge("code_generation_crew", "memory_query_cache")
    graph.add_edge("planning_crew", "memory_query_cache")
    graph.add_edge("crew_chain", "memory_query_cache")
    graph.add_edge("crew_parallel", "memory_query_cache")
    
    graph.add_edge("memory_query_cache", "load_memory")
    graph.add_edge("load_memory", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "memory_cache_store")
    graph.add_edge("memory_cache_store", "respond")
    graph.add_edge("respond", "compress_conversation")
    graph.add_edge("compress_conversation", "summarize")
    graph.add_edge("summarize", "update_memory")
    graph.add_edge("update_memory", "cache_stats")

    return graph.compile(checkpointer=build_checkpointer(config=load_core_config()))
