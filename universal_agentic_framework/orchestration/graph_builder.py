"""LangGraph orchestration: single source of truth for execution control and state management.

MEMORY LAYER INTEGRATION & SOURCE OF TRUTH OWNERSHIP:

1. SHORT-MEMORY (Digest Chain)
   - Owner: LangGraph graph orchestration via GraphState
   - Lifecycle:
     a) Created: performance_nodes.conversation_compression_node_sync() extracts digests
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
    - Metadata consistency maintained via _metadata_cache and canonical Mem0 metadata fields

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
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import START, StateGraph

from universal_agentic_framework.llm.budget import (
    estimate_tokens,
    TokenBudgetExceeded,
    get_budget_context,
    get_node_budget,
    get_response_reserve_tokens,
    per_node_hard_limit_enabled,
    require_tokens,
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
    memory_query_cache_node_sync,
    memory_cache_store_node_sync,
    conversation_compression_node_sync,
    cache_stats_node_sync,
    get_summarizer,
)
from universal_agentic_framework.orchestration.crew_nodes import (
    node_research_crew,
    node_analytics_crew,
    node_code_generation_crew,
    node_planning_crew,
    node_crew_chain,
    node_crew_parallel,
)
from universal_agentic_framework.orchestration.checkpointing import build_checkpointer

# Import extracted helpers
from universal_agentic_framework.orchestration.helpers import (
    extract_exact_reply_directive,
    record_tool_success,
    record_tool_error,
    build_attachment_context_block,
    build_workspace_document_context_block,
    detect_tool_routing_intents,
    score_tool_similarity,
    extract_calculator_expression,
    build_semantic_tool_kwargs,
    run_forced_tool,
    execute_semantic_scored_tools,
    prepare_scored_tools_with_forced_execution,
    apply_top_k_scored_tools,
    get_routing_embedding_provider,
    safe_get_model,
    resolve_initial_model_metadata,
    invoke_with_model_fallback,
    resolve_effective_tool_calling_mode,
    record_runtime_native_tool_leak,
    validate_and_log_tool_calling_mode,
)

logger = get_logger(__name__)

_DIGEST_CHAIN_MAX_ITEMS = 5


def _is_digest_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get("digest_id"))


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
    user_id: str
    session_id: str  # Session identifier for co-occurrence tracking
    language: str
    fork_name: str  # Fork identifier for metrics
    profile_id: str  # Active deployment profile identifier
    user_settings: Dict[str, Any]  # User preferences: tool_toggles, rag_config, preferred_model, theme, language
    llm_capability_probes: List[Dict[str, Any]]  # Adapter-provided probe snapshots per provider
    loaded_memory: List[Dict[str, Any]]
    digest_context: List[Dict[str, Any]]  # Digest subset from loaded memory retrieval
    memory_analytics: Dict[str, Any]  # Memory retrieval analytics (importance scores, related count)
    knowledge_context: List[Dict[str, Any]]  # RAG retrieved documents
    loaded_tools: List[Any]  # BaseTool instances from registry
    candidate_tools: List[Dict[str, Any]]  # Layer 1 pre-filter output: [{tool, name, score}]
    tool_calling_mode: str  # "native" | "structured" | "react" — from provider config
    tool_calling_mode_reason: str  # Why selected mode was chosen (configured or downgraded)
    prefilter_intents: Dict[str, Any]  # Intent detection results from Layer 1 for Layer 2 use
    tool_results: Dict[str, str]  # Tool name -> execution result
    tool_execution_results: Dict[str, Dict[str, Any]]  # Tool name -> structured execution envelope
    routing_metadata: Dict[str, str]  # Tool name -> reason for selection
    crew_results: Dict[str, Any]  # Multi-agent crew execution results
    tokens_used: int
    turn_tokens_used: int  # Current invocation token usage for per-turn budgeting
    input_tokens: int  # Input tokens count (separate tracking)
    output_tokens: int  # Output tokens count (separate tracking)
    provider_used: str  # Actual provider used for response generation
    model_used: str  # Actual model used for response generation
    summary_text: str
    digest_chain: List[Dict[str, Any]]  # Rolling conversation digest metadata
    sources: List[Dict[str, Any]]  # [{type: "web"|"rag", label: str, url: str|None}]


def _get_routing_embedding_provider(config: Any) -> Tuple[EmbeddingProvider, str]:
    """Return cached embedding provider and model name used for tool routing."""
    return get_routing_embedding_provider(
        config,
        logger=logger,
        build_provider_func=build_embedding_provider,
    )


class _ModelInvokeError(RuntimeError):
    def __init__(self, message: str, provider: str, model_name: str):
        super().__init__(message)
        self.provider = provider
        self.model_name = model_name


def _invoke_with_model_fallback(
    *,
    config: Any,
    language: str,
    payload: Any,
    initial_model: object,
    initial_provider: str,
    initial_model_name: str,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str, str, object]:
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
    fork_name = getattr(core_config.fork, "name", "default-fork")
    profile_id = state.get("profile_id") or get_active_profile_id()
    fork_language = getattr(core_config.fork, "language", "en")
    # Use conversation language from state; fall back to fork_language from config
    conversation_language = state.get("language") or fork_language

    logger.info("Loading tools from registry", fork_name=fork_name, language=conversation_language)

    with track_node_execution(fork_name, "load_tools"):
        try:
            from universal_agentic_framework.config import get_profile_dir

            profile_dir = get_profile_dir(profile_id=profile_id, require_exists=False)
            profile_tools_dir = profile_dir / "tools"

            registry = ToolRegistry(
                config=tools_config,
                fork_language=conversation_language,
                extra_tools_dir=profile_tools_dir,
            )
            tools = registry.discover_and_load()

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
    fork_name = getattr(config.fork, "name", "default-fork")

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
        fork_name=fork_name,
        tools_count=len(loaded_tools),
        query_length=len(user_msg),
    )

    with track_node_execution(fork_name, "prefilter_tools"):
        try:
            import re

            # Skip for greetings
            greeting_pattern = (
                r"^\s*(hi|hello|hey|hallo|servus|moin|"
                r"guten\s+(tag|morgen|abend))\s*[!.?]*\s*$"
            )
            if re.match(greeting_pattern, user_msg.lower()):
                logger.info("Skipping tool pre-filter for greeting query")
                state.update(empty_state)
                return state

            embedding_provider, embedding_model_name = _get_routing_embedding_provider(config)
            query_embedding = embedding_provider.encode(user_msg)

            similarity_threshold = getattr(
                getattr(config, "tool_routing", None), "similarity_threshold", 0.3
            )
            top_k = getattr(getattr(config, "tool_routing", None), "top_k", 5)
            intent_boost = getattr(
                getattr(config, "tool_routing", None), "intent_boost", 0.2
            )
            routing_language = state.get("language") or getattr(
                config.fork, "language", "en"
            )

            # Detect intents for score boosting
            intents = detect_tool_routing_intents(
                user_msg=user_msg, language=routing_language
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
                if tool_name == "datetime_tool" and intents["mentions_datetime"]:
                    similarity += intent_boost
                elif tool_name == "calculator_tool" and intents["mentions_calculation"]:
                    similarity += intent_boost
                elif tool_name == "file_ops_tool" and intents["mentions_file_ops"]:
                    similarity += intent_boost
                elif tool_name == "extract_webpage_mcp" and intents["url_in_query"]:
                    similarity += intent_boost
                elif tool_name == "web_search_mcp" and intents.get("mentions_web_search"):
                    similarity += intent_boost

                # Hard intent override: when user explicitly asks for web search,
                # keep web_search_mcp above both threshold gates so Layer 2 can decide.
                if tool_name == "web_search_mcp" and intents.get("mentions_web_search"):
                    min_top_score_cfg = getattr(
                        getattr(config, "tool_routing", None), "min_top_score", 0.7
                    )
                    forced_floor = max(similarity_threshold, min_top_score_cfg) + 0.01
                    if similarity < forced_floor:
                        logger.info(
                            "Applying web-search intent override",
                            original_similarity=round(similarity, 4),
                            forced_similarity=round(forced_floor, 4),
                        )
                        similarity = forced_floor

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


def node_route_tools(state: GraphState) -> GraphState:
    """Semantic tool routing: score user query against tool descriptions and auto-execute matching tools.
    
    Model-agnostic approach: works with any LLM, regardless of native tool-calling support.
    Uses cosine similarity + keyword heuristics for tool selection.
    Results injected into context for LLM.
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    
    loaded_tools = state.get("loaded_tools", [])
    user_msg = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""
    
    if not loaded_tools or not user_msg:
        state["tool_results"] = {}
        state["tool_execution_results"] = {}
        return state
    
    logger.info("Routing tools semantically", fork_name=fork_name, tools_count=len(loaded_tools), query_length=len(user_msg))
    
    with track_node_execution(fork_name, "route_tools"):
        try:
            import re

            # Skip tool routing for short greeting-only inputs
            greeting_pattern = r"^\s*(hi|hello|hey|hallo|servus|moin|guten\s+(tag|morgen|abend))\s*[!.?]*\s*$"
            if re.match(greeting_pattern, user_msg.lower()):
                logger.info("Skipping tool routing for greeting query")
                state["tool_results"] = {}
                state["tool_execution_results"] = {}
                return state
            
            embedding_provider, embedding_model_name = _get_routing_embedding_provider(config)
            
            # Embed user query
            query_embedding = embedding_provider.encode(user_msg)
            logger.info("Tool routing: query embedded", embedding_size=len(query_embedding))
            
            tool_results: Dict[str, str] = {}
            tool_execution_results: Dict[str, Dict[str, Any]] = {}
            similarity_threshold = getattr(getattr(config, "tool_routing", None), "similarity_threshold", 0.3)
            top_k = getattr(getattr(config, "tool_routing", None), "top_k", None)
            timezone = getattr(getattr(config, "fork", None), "timezone", None)
            routing_language = state.get("language") or getattr(config.fork, "language", "en")

            intents = detect_tool_routing_intents(
                user_msg=user_msg,
                language=routing_language,
            )
            search_language = intents["search_language"]
            search_region = intents["search_region"]
            mentions_datetime = intents["mentions_datetime"]
            mentions_calculation = intents["mentions_calculation"]
            mentions_file_ops = intents["mentions_file_ops"]
            mentions_web_search = intents.get("mentions_web_search", False)
            url_in_query = intents["url_in_query"]
            if intents["asks_about_tools"]:
                logger.info("Meta-question detected: skipping tool execution", ask_topic="available_tools")
                state["tool_results"] = {}
                state["tool_execution_results"] = {}
                return state
            wants_save_to_rag = intents["wants_save_to_rag"]
            enhanced_web_query = intents["enhanced_web_query"]
            requested_web_results = intents["requested_web_results"]

            # ── Tool scoring & execution ────────────────────────────────

            executed_forced = set()
            routing_metadata = {}  # Track why each tool was selected
            def _score_tool_adapter(**kwargs):
                return score_tool_similarity(
                    embedding_model_name=kwargs.get("embedding_model_name", embedding_model_name),
                    tool_name=kwargs.get("tool_name", "unknown"),
                    tool_desc=kwargs.get("tool_desc", ""),
                    embedding_provider=kwargs.get("embedding_provider", embedding_provider),
                    query_embedding=kwargs.get("query_embedding", query_embedding),
                )

            scored_tools = prepare_scored_tools_with_forced_execution(
                loaded_tools=loaded_tools,
                config=config,
                user_msg=user_msg,
                embedding_model_name=embedding_model_name,
                embedding_provider=embedding_provider,
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                mentions_datetime=mentions_datetime,
                mentions_calculation=mentions_calculation,
                mentions_file_ops=mentions_file_ops,
                mentions_web_search=mentions_web_search,
                enhanced_web_query=enhanced_web_query,
                requested_web_results=requested_web_results,
                search_region=search_region,
                url_in_query=url_in_query,
                wants_save_to_rag=wants_save_to_rag,
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                score_tool_func=_score_tool_adapter,
            )

            scored_tools = apply_top_k_scored_tools(scored_tools, top_k)

            # Score-spread gate: when all tools score similarly the query is
            # not tool-specific (e.g. casual greetings).  Only apply the gate
            # when no forced tool was already executed (forced tools bypass
            # similarity entirely) and when there are enough tools to measure
            # spread meaningfully.
            if len(scored_tools) >= 3 and not executed_forced:
                scores = [s for _, s in scored_tools]
                max_score = max(scores)
                mean_score = sum(scores) / len(scores)
                spread = max_score - mean_score
                min_spread = 0.05  # Empirically tuned: tool queries show spread > 0.08
                if spread < min_spread:
                    logger.info(
                        "Score-spread gate: all tools scored similarly, skipping semantic execution",
                        max_score=round(max_score, 4),
                        mean_score=round(mean_score, 4),
                        spread=round(spread, 4),
                        min_spread=min_spread,
                    )
                    scored_tools = []

            execute_semantic_scored_tools(
                scored_tools=scored_tools,
                similarity_threshold=similarity_threshold,
                executed_forced=executed_forced,
                mentions_calculation=mentions_calculation,
                mentions_datetime=mentions_datetime,
                mentions_file_ops=mentions_file_ops,
                user_msg=user_msg,
                url_in_query=url_in_query,
                wants_save_to_rag=wants_save_to_rag,
                enhanced_web_query=enhanced_web_query,
                requested_web_results=requested_web_results,
                search_language=search_language,
                search_region=search_region,
                timezone=timezone,
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
            )
            
            state["tool_results"] = tool_results
            state["tool_execution_results"] = tool_execution_results
            state["routing_metadata"] = routing_metadata
            logger.info("Tool routing completed", tools_executed=len(tool_results))
            
        except Exception as e:
            logger.error("Tool routing failed", error=str(e), exc_info=True)
            state["tool_results"] = {}
            state["tool_execution_results"] = {}
    
    return state


def node_call_tools_native(state: GraphState) -> GraphState:
    """Layer 2 (native): Bind candidate tools to LLM, let model decide which to call.

    Uses LangChain's ``bind_tools`` + ``tool_calls`` parsing.  Executes the
    tools the model requests, stores results in the same ``tool_results`` /
    ``tool_execution_results`` fields used by the rest of the pipeline.
    
    Mode enforcement: Validates that tool_calling_mode is 'native' before proceeding.
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    max_retries = getattr(getattr(config, "tool_routing", None), "max_retries", 2)

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "native", "call_tools_native", fork_name
    )
    if not is_valid:
        logger.warning(
            "Native tool calling invoked but mode doesn't match; proceeding with caution",
            extra={"validation_reason": validation_reason, "actual_mode": state.get("tool_calling_mode")},
        )

    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )
    if not user_msg:
        return state

    native_intents = detect_tool_routing_intents(user_msg, state.get("language", "en"))
    url_in_query = native_intents.get("url_in_query")
    wants_save_to_rag = bool(native_intents.get("wants_save_to_rag"))

    logger.info(
        "Layer 2 native tool calling",
        fork_name=fork_name,
        candidates=len(candidates),
    )

    with track_node_execution(fork_name, "call_tools_native"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            lang = state.get("language") or getattr(config.fork, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = safe_get_model(config, lang, preferred_model=preferred_model)

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
                            )
                            logger.error(
                                "Native fallback execution failed",
                                tool="extract_webpage_mcp",
                                error=str(fallback_err),
                            )
                    break

                parse_error = False
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})

                    if tool_name == "extract_webpage_mcp" and isinstance(tool_args, dict):
                        def _contains_url_arg(value: Any) -> bool:
                            if isinstance(value, str):
                                candidate = value.strip().strip('"\'')
                                return bool(re.match(r"^(https?://|www\.)", candidate))
                            if isinstance(value, dict):
                                return any(_contains_url_arg(v) for v in value.values())
                            if isinstance(value, (list, tuple)):
                                return any(_contains_url_arg(v) for v in value)
                            return False

                        if not _contains_url_arg(tool_args) and url_in_query:
                            tool_args = dict(tool_args)
                            tool_args["request_url"] = url_in_query

                    tool_obj = tool_lookup.get(tool_name)
                    if not tool_obj:
                        logger.warning("Model requested unknown tool", tool=tool_name, attempt=attempt)
                        parse_error = True
                        continue

                    # Validate args against schema if available
                    schema = getattr(tool_obj, "args_schema", None)
                    if schema:
                        try:
                            schema(**tool_args)
                        except Exception as val_err:
                            logger.warning(
                                "Tool call validation failed",
                                tool=tool_name,
                                error=str(val_err),
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
                        )
                        logger.info("Tool executed (native)", tool=tool_name, result_length=len(str(result)))
                    except Exception as exec_err:
                        record_tool_error(
                            tool_name=tool_name,
                            error=exec_err,
                            tool_results=tool_results,
                            tool_execution_results=tool_execution_results,
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
    fork_name = getattr(config.fork, "name", "default-fork")
    max_retries = getattr(getattr(config, "tool_routing", None), "max_retries", 2)

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "structured", "call_tools_structured", fork_name
    )
    if not is_valid:
        logger.warning(
            "Structured tool calling invoked but mode doesn't match; proceeding with caution",
            extra={"validation_reason": validation_reason, "actual_mode": state.get("tool_calling_mode")},
        )

    user_msg = (
        state.get("messages", [])[-1].get("content", "")
        if state.get("messages")
        else ""
    )
    if not user_msg:
        return state

    logger.info(
        "Layer 2 structured tool calling",
        fork_name=fork_name,
        candidates=len(candidates),
    )

    with track_node_execution(fork_name, "call_tools_structured"):
        try:
            import json
            from langchain_core.messages import HumanMessage, SystemMessage

            lang = state.get("language") or getattr(config.fork, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = safe_get_model(config, lang, preferred_model=preferred_model)

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
            system_content = (
                "You are a helpful assistant with access to tools.\n"
                "If a tool would help answer the user's question, respond with ONLY a JSON object:\n"
                '{"tool": "<tool_name>", "args": {<arguments>}}\n'
                "If no tool is needed, respond normally in plain text.\n\n"
                f"Available tools:\n{tool_block}"
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
                response = model.invoke(messages)
                response_text = _normalize_structured_response_text(
                    response.content if hasattr(response, "content") else response
                )

                # Try to parse JSON tool call from response
                tool_call = None
                try:
                    stripped = response_text.strip()
                    if stripped.startswith("{"):
                        tool_call = json.loads(stripped)
                except json.JSONDecodeError:
                    # Try to extract JSON from within the response
                    import re
                    json_match = re.search(r'\{[^{}]*"tool"\s*:.*?\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            tool_call = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass

                if not tool_call or "tool" not in tool_call:
                    logger.info("Model chose not to call tools (structured)", attempt=attempt)
                    break

                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})
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

                # Validate args
                schema = getattr(tool_obj, "args_schema", None)
                if schema:
                    try:
                        schema(**tool_args)
                    except Exception as val_err:
                        logger.warning(
                            "Structured tool call validation failed",
                            tool=tool_name,
                            error=str(val_err),
                            attempt=attempt,
                        )
                        if attempt < max_retries:
                            from langchain_core.messages import AIMessage
                            messages.append(AIMessage(content=response_text))
                            messages.append(HumanMessage(
                                content=f"Tool call validation error: {val_err}. Please fix the arguments and try again."
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
                    )
                    logger.info("Tool executed (structured)", tool=tool_name, result_length=len(str(result)))
                except Exception as exec_err:
                    record_tool_error(
                        tool_name=tool_name,
                        error=exec_err,
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
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
    fork_name = getattr(config.fork, "name", "default-fork")
    max_iterations = getattr(getattr(config, "tool_routing", None), "max_retries", 2) + 1

    candidates = state.get("candidate_tools", [])
    if not candidates:
        return state

    # Validate that this node should be executing (mode enforcement)
    is_valid, validation_reason = validate_and_log_tool_calling_mode(
        state, "react", "call_tools_react", fork_name
    )
    if not is_valid:
        logger.warning(
            "ReAct tool calling invoked but mode doesn't match; proceeding with caution",
            extra={"validation_reason": validation_reason, "actual_mode": state.get("tool_calling_mode")},
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
        fork_name=fork_name,
        candidates=len(candidates),
    )

    with track_node_execution(fork_name, "call_tools_react"):
        try:
            import re
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lang = state.get("language") or getattr(config.fork, "language", "en")
            user_settings = state.get("user_settings", {})
            preferred_model = user_settings.get("preferred_model")
            model = safe_get_model(config, lang, preferred_model=preferred_model)

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

            import json

            for iteration in range(max_iterations):
                response = model.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)

                # Check for Final Answer
                final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
                if final_match:
                    logger.info("ReAct loop completed with final answer", iterations=iteration + 1)
                    break

                # Parse Action line
                action_match = re.search(r'Action:\s*(\{.*?\})', response_text, re.DOTALL)
                if not action_match:
                    logger.info("No action found in ReAct response, treating as final answer", iteration=iteration)
                    break

                try:
                    action = json.loads(action_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse ReAct action JSON", iteration=iteration)
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content="Observation: Invalid JSON in Action. Please fix the format."))
                    continue

                tool_name = action.get("tool", "")
                tool_args = action.get("args", {})
                tool_obj = tool_lookup.get(tool_name)

                if not tool_obj:
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(
                        content=f"Observation: Tool '{tool_name}' not found. Available: {', '.join(tool_lookup.keys())}"
                    ))
                    continue

                try:
                    result = tool_obj._run(**tool_args)
                    record_tool_success(
                        tool_name=tool_name,
                        result=result,
                        reason=f"react tool call (iteration {iteration})",
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
                        routing_metadata=routing_metadata,
                    )
                    observation = str(result)
                    logger.info("Tool executed (react)", tool=tool_name, iteration=iteration)
                except Exception as exec_err:
                    record_tool_error(
                        tool_name=tool_name,
                        error=exec_err,
                        tool_results=tool_results,
                        tool_execution_results=tool_execution_results,
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
    fork_name = getattr(config.fork, "name", "default-fork")
    features_config = load_features_config()
    
    logger.info("Loading memory", user_id=state.get("user_id"), fork_name=fork_name)

    if not getattr(features_config, "long_term_memory", False):
        logger.info("Long-term memory disabled via features flag", fork_name=fork_name)
        state["loaded_memory"] = []
        return state
    
    # Get memory feature flags
    include_related = getattr(features_config, "memory_include_related", False)
    top_k = getattr(features_config, "memory_top_k", 5)
    
    logger.info(
        "Memory features",
        include_related=include_related,
        top_k=top_k,
        fork_name=fork_name
    )
    
    with track_node_execution(fork_name, "load_memory"):
        backend = build_memory_backend(config)
        try:
            result = load_memory_node(
                state, 
                backend=backend,
                top_k=top_k,
                include_related=include_related
            )
            track_memory_operation(fork_name, "load", "success")
            
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
            track_memory_operation(fork_name, "load", "error")
            logger.error("Memory load failed", error=str(e), user_id=state.get("user_id"))
            raise


def node_retrieve_knowledge(state: GraphState) -> GraphState:
    """Retrieve relevant documents from knowledge base (RAG)."""
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    rag_config = getattr(config, "rag", None)
    features_config = load_features_config()
    
    user_msg = state["messages"][-1]["content"] if state.get("messages") else ""
    
    logger.info("RAG node started", fork_name=fork_name, has_query=bool(user_msg))
    
    if not user_msg:
        logger.info("No query for knowledge retrieval", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    if not getattr(features_config, "rag_retrieval", True):
        logger.info("RAG disabled via features flag", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    if rag_config is not None and not rag_config.enabled:
        logger.info("RAG disabled via config", fork_name=fork_name)
        state["knowledge_context"] = []
        return state
    
    logger.info("Retrieving knowledge", fork_name=fork_name, query_length=len(user_msg))
    
    with track_node_execution(fork_name, "retrieve_knowledge"):
        try:
            import httpx
            
            # Generate query embedding
            embedding_model_name = config.llm.get_role_model_name("embedding", config.fork.language)
            embedding_dimension = config.memory.embeddings.dimension
            embedding_provider_type = config.llm.get_embedding_provider_type()
            embedding_remote_endpoint = config.llm.get_embedding_remote_endpoint()

            logger.info(
                "RAG: Creating embedder",
                model=embedding_model_name,
                provider=embedding_provider_type,
            )
            embedder = build_embedding_provider(
                model_name=embedding_model_name,
                dimension=embedding_dimension,
                provider_type=embedding_provider_type,
                remote_endpoint=embedding_remote_endpoint,
            )
            query_embedding = embedder.encode(user_msg)
            
            logger.info("RAG: Generated embedding", embedding_size=len(query_embedding))
            
            # Use Qdrant REST API directly for compatibility
            qdrant_url = f"http://{config.memory.vector_store.host}:{config.memory.vector_store.port}"
            collection_name = "framework"
            top_k = 5
            score_threshold = None
            with_payload = True
            with_vector = False
            timeout_seconds = 30

            # First check user settings, then fall back to config
            user_settings = state.get("user_settings", {})
            user_rag_config = user_settings.get("rag_config", {})
            
            if user_rag_config:
                # User has custom RAG settings
                if user_rag_config.get("collection"):
                    collection_name = user_rag_config["collection"]
                if user_rag_config.get("top_k") is not None:
                    top_k = user_rag_config["top_k"]
                logger.info("Using user RAG config", collection=collection_name, top_k=top_k)
            elif rag_config is not None:
                # Fall back to system config
                if rag_config.collection_name:
                    collection_name = rag_config.collection_name
                top_k = rag_config.top_k
                score_threshold = rag_config.score_threshold
                with_payload = rag_config.with_payload
                with_vector = rag_config.with_vectors
                timeout_seconds = rag_config.timeout_seconds
                logger.info("Using system RAG config", collection=collection_name, top_k=top_k)
            
            def _search_qdrant(embedding_vector: list[float], label: str) -> list[dict]:
                payload = {
                    "vector": embedding_vector,
                    "limit": top_k,
                    "with_payload": with_payload,
                    "with_vector": with_vector,
                }
                if score_threshold is not None:
                    payload["score_threshold"] = score_threshold

                logger.info("RAG: Searching Qdrant", url=qdrant_url, collection=collection_name, query_label=label)
                response = httpx.post(
                    f"{qdrant_url}/collections/{collection_name}/points/search",
                    json=payload,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                result = response.json().get("result", [])
                logger.info("RAG: Search completed", query_label=label, results_count=len(result))
                return result

            search_results = _search_qdrant(query_embedding, "full_query")

            # Heuristic: if the query contains a strong keyword, run a focused search too
            def _extract_keyword(query: str) -> str | None:
                import re

                tokens = [t.lower() for t in re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", query)]
                if not tokens:
                    return None
                stopwords = {
                    "kennst", "kannst", "weißt", "wissen", "sagen", "bitte",
                    "about", "tell", "what", "which", "with", "have", "this",
                    "that", "from", "oder", "aber", "denn", "dann", "doch",
                }
                candidates = [t for t in tokens if t not in stopwords]
                if not candidates:
                    return None
                return max(candidates, key=len)

            keyword = _extract_keyword(user_msg)
            if keyword and keyword != user_msg.lower():
                keyword_embedding = embedder.encode(keyword)
                keyword_results = _search_qdrant(keyword_embedding, "keyword")
                if keyword_results:
                    search_results = search_results + keyword_results
            
            # Extract relevant documents with client-side relevance filter
            # Documents below this score are noise and get dropped regardless of Qdrant config
            min_relevance_score = score_threshold if score_threshold is not None else 0.6
            knowledge_docs = []
            seen = set()
            for result in search_results:
                doc_score = result.get("score", 0.0)
                if doc_score < min_relevance_score:
                    continue
                payload = result.get("payload", {})
                file_path = payload.get("file_name") or payload.get("file_path") or "Unknown"
                file_name = file_path.split("/")[-1] if isinstance(file_path, str) else "Unknown"
                doc_id = result.get("id") or f"{file_path}:{payload.get('chunk_index')}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc = {
                    "text": payload.get("text", ""),
                    "file_name": file_name,
                    "file_path": file_path,
                    "score": doc_score
                }
                knowledge_docs.append(doc)
                logger.info("RAG: Found document", file=doc["file_name"], score=doc["score"])

            if len(knowledge_docs) > top_k:
                knowledge_docs = sorted(knowledge_docs, key=lambda d: d.get("score", 0.0), reverse=True)[:top_k]
            
            state["knowledge_context"] = knowledge_docs
            
            logger.info(
                "Knowledge retrieved successfully",
                fork_name=fork_name,
                results_count=len(knowledge_docs),
                top_score=knowledge_docs[0]["score"] if knowledge_docs else 0.0
            )
            
        except Exception as e:
            logger.error("Knowledge retrieval failed", error=str(e), fork_name=fork_name, exc_info=True)
            state["knowledge_context"] = []
    
    return state


def node_generate_response(state: GraphState) -> GraphState:
    config = load_core_config()
    lang = state.get("language") or config.fork.language
    fork_name = getattr(config.fork, "name", "default-fork")
    
    # Check for user's preferred model
    user_settings = state.get("user_settings", {})
    preferred_model = user_settings.get("preferred_model")
    
    logger.info(
        "Generating response",
        language=lang,
        fork_name=fork_name,
        preferred_model=preferred_model or "default"
    )

    model = safe_get_model(config, lang, preferred_model=preferred_model)
    provider, model_name = resolve_initial_model_metadata(config, lang, preferred_model)

    # Hybrid budget enforcement: global + per-turn hard limits; per-node optional hard guardrail.
    budget_ctx = get_budget_context(state, config.tokens)
    node_budget = get_node_budget(config.tokens, "response_node", budget_ctx["per_turn_budget"])
    enforce_node_hard_limit = per_node_hard_limit_enabled(config.tokens)
    reserve_tokens = get_response_reserve_tokens(config.tokens, budget_ctx["per_turn_budget"])

    available_response_budget = max(1, budget_ctx["turn_remaining"] - reserve_tokens)
    if enforce_node_hard_limit:
        available_response_budget = min(available_response_budget, node_budget)

    # Find the last user-role message so crew-appended assistant messages don't pollute user_msg.
    user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break
    input_tokens = estimate_tokens(user_msg)
    require_tokens(input_tokens, available_response_budget, "Response input")
    
    # Build context-aware prompt with clear role definition.
    # Resolution order: env var → config prompt files → emergency fallback
    env_prompt = os.environ.get(f"PROMPT_SYSTEM_{lang.upper()}", "").strip()
    if env_prompt:
        system_prompt = env_prompt.replace("\\n", "\n")
    else:
        prompts_cfg = getattr(config, "prompts", None)
        configured_prompt = prompts_cfg.get_prompt(lang, "response_system", fallback_lang="en") if prompts_cfg else None
        system_prompt = configured_prompt or (
            "You are a helpful AI assistant.\n"
            "Answer the user's question clearly and factually.\n"
            "Use the knowledge base and available tools when relevant to provide better answers."
        )
    
    # Enforce response language to avoid drift into unintended languages (env-configurable)
    env_lang = os.environ.get(f"PROMPT_LANG_{lang.upper()}", "").strip()
    if env_lang:
        language_instruction = env_lang
    else:
        prompts_cfg = getattr(config, "prompts", None)
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
            env_tooling = os.environ.get("PROMPT_TOOLING_MODE", "").strip()
            tooling_text = env_tooling.replace("\\n", "\n") if env_tooling else _default_tooling
            system_prompt += f"\n\n=== TOOLING MODE ===\n{tooling_text}\n=== END TOOLING MODE ===\n"
            
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
    collected_sources: List[Dict[str, Any]] = []  # structured source tracking

    # Utility-only tool responses (datetime/calculator/file ops) should not force citation-style footnotes.
    tool_results = state.get("tool_results", {})
    tool_execution_results = state.get("tool_execution_results", {})
    routing_metadata = state.get("routing_metadata", {})
    utility_tool_names = {"datetime_tool", "calculator_tool", "file_ops_tool"}
    used_tool_names = set(tool_results.keys())
    utility_only_response = bool(used_tool_names) and used_tool_names.issubset(utility_tool_names)

    if knowledge_context:
        context_text = "\n\n".join([
            f"[Quelle: {doc.get('file_name', 'Unknown').split('-')[-1].replace('.md', '')}]\n{doc.get('text', '')}"
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
            # Collect structured RAG source (deduplicated by label) for citation-capable responses.
            if not utility_only_response:
                label = str(file_name or "Unknown").split("-")[-1].replace(".md", "")
                if label not in _seen_rag_labels:
                    _seen_rag_labels.add(label)
                    collected_sources.append({"type": "rag", "label": label, "url": None})
    
    # Add past memories/conversation context if available
    loaded_memory = state.get("loaded_memory", [])
    web_tools_used = any(
        name in tool_results for name in ("extract_webpage_mcp", "web_search_mcp")
    )
    if loaded_memory:
        memory_context = "\n\n".join([
            f"[Memory]\n{mem.text if hasattr(mem, 'text') else mem.get('text', '')}"
            for mem in loaded_memory[:5]  # Top 5 memories
        ])
        if web_tools_used:
            system_prompt += (
                "\n\n=== PAST CONTEXT (LOW PRIORITY) ===\n"
                "Use this only as background. If it conflicts with current-turn TOOL RESULTS, "
                "always trust current-turn TOOL RESULTS.\n\n"
                f"{memory_context}\n"
                "=== END PAST CONTEXT (LOW PRIORITY) ===\n"
            )
            logger.info(
                "Loaded memory injected as low-priority context due to fresh web tool results",
                memory_count=len(loaded_memory),
                tools_count=len(tool_results),
            )
        else:
            system_prompt += f"\n\n=== PAST CONTEXT ===\n{memory_context}\n=== END PAST CONTEXT ===\n"
            logger.info("Loaded memory injected into context", memory_count=len(loaded_memory))

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
        fork_name = state.get("fork_name", "unknown")
        ATTACHMENTS_INJECTED_TOTAL.labels(fork_name=fork_name).inc()
    else:
        state["attachment_context"] = []
        # Track requests with no attachments
        from universal_agentic_framework.monitoring.metrics import ATTACHMENTS_NONE_TOTAL
        fork_name = state.get("fork_name", "unknown")
        ATTACHMENTS_NONE_TOTAL.labels(fork_name=fork_name).inc()

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
        tool_context_lines = []
        for tool_name, result in tool_results.items():
            # Add routing reason if available
            reason = routing_metadata.get(tool_name, "semantic match")
            tool_header = f"[Tool: {tool_name} | Reason: {reason}]"

            envelope = tool_execution_results.get(tool_name, {})
            summary = envelope.get("summary")
            rendered = summary if summary else result
            tool_context_lines.append(f"{tool_header}\n{rendered}")
        
        tool_context = "\n\n".join(tool_context_lines)
        system_prompt += f"\n\n=== TOOL RESULTS ===\n{tool_context}\n=== ENDE TOOL RESULTS ===\n"
        system_prompt += (
            "\n\n=== CONTEXT PRIORITY ===\n"
            "Use current-turn TOOL RESULTS as the primary source of truth. "
            "Use PAST CONTEXT only as secondary background. "
            "If there is any conflict, follow TOOL RESULTS. "
            "Do not mention model training data, knowledge-cutoff dates, or stale prior knowledge when TOOL RESULTS provide current information.\n"
            "=== END CONTEXT PRIORITY ===\n"
        )
        logger.info("Tool results injected into context", tools_count=len(tool_results))

        # Extract URLs from tool results so citations can be limited to known sources
        import re
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

        # Inject synthesis instruction so LLM writes a summary, not a raw list (env-configurable)
        env_synthesis = os.environ.get(f"PROMPT_SYNTHESIS_{lang.upper()}", "").strip()
        if env_synthesis:
            synthesis_text = env_synthesis.replace("\\n", "\n")
        else:
            prompts_cfg = getattr(config, "prompts", None)
            prompt_key = "synthesis_with_sources" if has_citable_sources else "synthesis"
            synthesis_text = (prompts_cfg.get_prompt(lang, prompt_key, fallback_lang="en") if prompts_cfg else None)
            if not synthesis_text:
                # Emergency fallback
                synthesis_text = (
                    "Synthesize a coherent, well-structured answer from the tool results and knowledge base above. "
                    "Do NOT list raw result items. Write a fluent summary that directly answers the user's question."
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
        env_sources_instr = os.environ.get("PROMPT_SOURCES", "").strip()
        sources_instruction = env_sources_instr or (
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
    _crew_result_prefixes = (
        "Research Result:",
        "Analytics Result:",
        "Code Generation Result:",
        "Planning Result:",
    )
    crew_results = state.get("crew_results", {})
    for crew_name, crew_result in crew_results.items():
        if isinstance(crew_result, dict) and crew_result.get("success") and crew_result.get("result"):
            section = crew_name.upper().replace("_", " ")
            system_prompt += (
                f"\n\n=== {section} FINDINGS ===\n"
                f"{crew_result['result']}\n"
                f"=== END {section} FINDINGS ===\n"
            )
            logger.info("Crew result injected into context", crew=crew_name, result_length=len(crew_result["result"]))

    workspace_document_context = state.get("workspace_document_context") or []
    if state.get("workspace_writeback_requested") and len(workspace_document_context) == 1:
        target = workspace_document_context[0]
        filename = target.get("filename") or "document.txt"
        system_prompt += (
            "\n\n=== WORKSPACE WRITEBACK MODE ===\n"
            f"The user asked you to update and save the workspace document '{filename}'.\n"
            "Return ONLY the complete revised file content to be saved.\n"
            "Do not add explanations, preambles, commentary, or markdown code fences.\n"
            "=== END WORKSPACE WRITEBACK MODE ===\n"
        )
    
    # Build messages in proper chat format for LangChain
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    
    # Start with system prompt
    messages = [SystemMessage(content=system_prompt)]
    
    # Add full conversation history (all previous user and assistant messages).
    # Skip crew-appended assistant messages — their content is already injected into
    # the system prompt as FINDINGS context above.  Keeping them as AIMessage turns
    # causes the model to treat the crew output as its own prior answer and only add
    # a short follow-up instead of synthesising a full response.
    all_msgs = state.get("messages") or []
    for msg in all_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
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
    
    # Debug: log what we're sending to the model
    tool_results = state.get("tool_results", {})
    logger.info("Sending to LLM", 
                system_prompt_length=len(system_prompt),
                user_msg_length=len(user_msg),
                knowledge_context_docs=len(knowledge_context),
                messages_count=len(messages),
                workspace_document_context_docs=len(workspace_document_context),
                tools_executed=len(tool_results))

    with track_node_execution(fork_name, "respond"):
        try:
            response_text, provider, model_name, active_model = _invoke_with_model_fallback(
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
            track_llm_call(fork_name, provider, model_name, "success")
        except _ModelInvokeError as e:
            track_llm_call(fork_name, provider, model_name, "error")
            logger.error(
                "LLM call failed",
                error=str(e),
                exc_info=True,
                provider=e.provider,
                model_name=e.model_name,
            )
            provider = e.provider
            model_name = e.model_name
            response_text = f"LLM Error: {str(e)[:100]}"
        except Exception as e:
            track_llm_call(fork_name, provider, model_name, "error")
            logger.error("LLM call failed", error=str(e), exc_info=True, provider=provider, model_name=model_name)
            response_text = f"LLM Error: {str(e)[:100]}"

        # Remove leaked tool-call/control tokens from model output
        if response_text:
            import re

            original_text = response_text
            response_text = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", response_text, flags=re.DOTALL)
            response_text = re.sub(r"<\|[^|>]+\|>", "", response_text)
            response_text = response_text.strip()

            if response_text != original_text:
                if tool_calling_mode == "native":
                    logger.warning(
                        "Control tokens leaked in native tool-calling mode — possible LM Studio parsing issue",
                    )
                    record_runtime_native_tool_leak(config, state, model_name)
                else:
                    logger.info("Sanitized control tokens from LLM response")

            if not response_text:
                # LLM emitted only tool-call tokens — retry with a synthesis-only prompt
                # (no tool catalog, explicit instruction to summarize the data)
                logger.info("LLM produced empty response after sanitization, retrying with synthesis prompt")

                _retry_has_citable_sources = bool(collected_sources)
                _synth_default = {
                    "en": (
                        "Synthesize a coherent, well-structured answer from the tool results and knowledge base above. "
                        "Do NOT list raw result items. Write a fluent summary that directly answers the user's question. "
                        "Treat current-turn tool results as current facts. Do not mention training cutoffs, outdated knowledge limits, or inability to browse."
                        + (
                            " Cite sources using numbered references like [1], [2] etc. matching the SOURCES list below."
                            if _retry_has_citable_sources
                            else ""
                        )
                    ),
                    "de": (
                        "Fasse die obigen Tool-Ergebnisse und die Wissensdatenbank zu einer zusammenhaengenden, "
                        "gut strukturierten Antwort zusammen. Liste KEINE rohen Ergebnis-Eintraege auf. "
                        "Schreibe eine fluessige Zusammenfassung, die die Frage des Benutzers direkt beantwortet. "
                        "Behandle aktuelle Tool-Ergebnisse als aktuelle Fakten. Erwaehne keine Wissensgrenzen, Trainingsdaten-Grenzen oder fehlende Browsing-Faehigkeit."
                        + (
                            " Zitiere Quellen mit nummerierten Referenzen wie [1], [2] usw. passend zur SOURCES-Liste unten."
                            if _retry_has_citable_sources
                            else ""
                        )
                    ),
                }
                env_synth = os.environ.get(f"PROMPT_SYNTHESIS_{lang.upper()}", "").strip()
                synth_instr = env_synth.replace("\\n", "\n") if env_synth else _synth_default.get(lang, _synth_default["en"])

                # Build a focused data-only prompt — no tool catalog at all
                retry_parts = [
                    "You are a helpful AI assistant. Do NOT emit tool calls or control tokens. "
                    "Return ONLY plain natural-language text. "
                    "Use current-turn tool results as the source of truth and do not mention training cutoffs or browsing limitations.\n"
                ]
                if knowledge_context:
                    retry_parts.append(f"=== KNOWLEDGE BASE ===\n{context_text}\n=== END KNOWLEDGE BASE ===\n")
                if tool_results:
                    for _tn, _tr in tool_results.items():
                        retry_parts.append(f"=== TOOL: {_tn} ===\n{_tr}\n=== END TOOL ===\n")
                # Include numbered source list so LLM can reference [N]
                if collected_sources:
                    src_lines = []
                    for src in collected_sources:
                        _idx = src.get("index", 0)
                        if src.get("url"):
                            src_lines.append(f"[{_idx}] {src['label']} - {src['url']}")
                        else:
                            src_lines.append(f"[{_idx}] {src['label']} (knowledge base)")
                    retry_parts.append("=== SOURCES ===\n" + "\n".join(src_lines) + "\n=== END SOURCES ===\n")
                retry_parts.append(f"=== TASK ===\n{synth_instr}\n=== END TASK ===\n")

                retry_prompt = "\n".join(retry_parts)
                retry_messages = [
                    SystemMessage(content=retry_prompt),
                    HumanMessage(content=user_msg),
                ]

                try:
                    retry_out = model.invoke(retry_messages)
                    if hasattr(retry_out, "content"):
                        response_text = retry_out.content
                    elif isinstance(retry_out, str):
                        response_text = retry_out
                    else:
                        response_text = str(retry_out)

                    # Strip any remaining control tokens from retry
                    response_text = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", response_text, flags=re.DOTALL)
                    response_text = re.sub(r"<\|[^|>]+\|>", "", response_text).strip()
                    logger.info("Synthesis retry succeeded", response_length=len(response_text))
                except Exception as retry_err:
                    logger.error("Synthesis retry failed", error=str(retry_err))
                    response_text = ""

            # Guardrail: if attachments were injected but the model still claims it cannot
            # access attachments, force one corrective retry with explicit instruction.
            attachment_context = state.get("attachment_context") or []
            if response_text and attachment_context:
                attachment_refusal_pattern = re.compile(
                    r"("
                    r"can(?:not|'t)\s+(?:view|access|see).{0,60}(?:attach|workspace\s+document|document)|"
                    r"don['’]t\s+see.{0,60}(?:attach|workspace\s+document|document)|"
                    r"(?:attach\w*|workspace\s+document\w*|document\w*).{0,60}"
                    r"(?:not\s+)?(?:available|accessible|provided|found|readable|unavailable|inaccessible)|"
                    r"not\s+accessible.{0,60}(?:content\s+extraction|extract)"
                    r")",
                    flags=re.IGNORECASE,
                )
                if attachment_refusal_pattern.search(response_text):
                    logger.warning(
                        "Model claimed attachments were unavailable despite injected attachment context; retrying once",
                        attachment_count=len(attachment_context),
                    )
                    # Track attachment refusal retry trigger
                    from universal_agentic_framework.monitoring.metrics import ATTACHMENT_REFUSAL_RETRIES_TOTAL
                    fork_name = state.get("fork_name", "unknown")
                    ATTACHMENT_REFUSAL_RETRIES_TOTAL.labels(fork_name=fork_name).inc()
                    
                    correction_prompt = (
                        system_prompt
                        + "\n\n=== ATTACHMENT HANDLING ===\n"
                        + "The USER ATTACHMENTS section is already provided in this prompt and is readable context. "
                        + "Do not state that attachments are unavailable. Use that content directly in your answer.\n"
                        + "=== END ATTACHMENT HANDLING ===\n"
                    )
                    correction_messages = [SystemMessage(content=correction_prompt)]
                    for msg in all_msgs:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            correction_messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            correction_messages.append(AIMessage(content=content))
                    try:
                        correction_out = model.invoke(correction_messages)
                        if hasattr(correction_out, "content"):
                            response_text = correction_out.content
                        elif isinstance(correction_out, str):
                            response_text = correction_out
                        else:
                            response_text = str(correction_out)
                        response_text = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", response_text, flags=re.DOTALL)
                        response_text = re.sub(r"<\|[^|>]+\|>", "", response_text).strip()
                        logger.info("Attachment correction retry succeeded", response_length=len(response_text))
                        # Track successful attachment retry
                        from universal_agentic_framework.monitoring.metrics import ATTACHMENT_REFUSAL_RETRIES_SUCCESS_TOTAL
                        ATTACHMENT_REFUSAL_RETRIES_SUCCESS_TOTAL.labels(fork_name=fork_name).inc()
                    except Exception as correction_err:
                        logger.error("Attachment correction retry failed", error=str(correction_err))

            # Guardrail: if webpage extraction succeeded but the model still claims
            # it cannot access the site, force one corrective retry using tool output.
            extract_result = str(tool_results.get("extract_webpage_mcp", "")).strip()
            if response_text and extract_result and not extract_result.lower().startswith("error:"):
                access_refusal_pattern = re.compile(
                    r"(unable\s+to\s+(?:access|retrieve|visit)|"
                    r"cannot\s+(?:access|retrieve|visit|reach)|"
                    r"can't\s+(?:access|retrieve|visit|reach)|"
                    r"connection\s+error|not\s+directly\s+retrievable|"
                    r"verbindungsproblem|nicht\s+abrufbar|nicht\s+zugaenglich|"
                    r"kann\s+(?:nicht|keine)\s+(?:zugreifen|abrufen|erreichen))",
                    flags=re.IGNORECASE,
                )
                if access_refusal_pattern.search(response_text):
                    logger.warning(
                        "Model contradicted successful extract_webpage_mcp result; retrying once",
                        extract_length=len(extract_result),
                    )
                    correction_prompt = (
                        "You are given successfully extracted webpage content. "
                        "Answer the user's question ONLY from this extracted content. "
                        "Do not claim connection or access errors.\n\n"
                        "=== EXTRACTED WEBPAGE CONTENT ===\n"
                        f"{extract_result[:12000]}\n"
                        "=== END EXTRACTED WEBPAGE CONTENT ==="
                    )
                    correction_messages = [
                        SystemMessage(content=correction_prompt),
                        HumanMessage(content=user_msg),
                    ]
                    try:
                        correction_out = model.invoke(correction_messages)
                        if hasattr(correction_out, "content"):
                            response_text = correction_out.content
                        elif isinstance(correction_out, str):
                            response_text = correction_out
                        else:
                            response_text = str(correction_out)
                        response_text = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", response_text, flags=re.DOTALL)
                        response_text = re.sub(r"<\|[^|>]+\|>", "", response_text).strip()
                        logger.info("Web extract contradiction retry succeeded", response_length=len(response_text))
                    except Exception as correction_err:
                        logger.error("Web extract contradiction retry failed", error=str(correction_err))

            # Final fallback — if retry also produced nothing, format raw results
            if not response_text:
                tool_based_text = None

                # Prefer readable formatting of web search results over raw dict string
                web_search_raw = str(tool_results.get("web_search_mcp", "")).strip()
                if web_search_raw and not web_search_raw.lower().startswith("tool execution failed"):
                    try:
                        import ast
                        parsed = ast.literal_eval(web_search_raw)
                        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                            items = parsed.get("results", [])[:5]
                            lines = []
                            for idx, item in enumerate(items, 1):
                                title = str(item.get("title", "Untitled"))
                                url = str(item.get("url", ""))
                                snippet = str(item.get("snippet", "")).replace("\n", " ").strip()
                                if len(snippet) > 180:
                                    snippet = snippet[:177] + "..."
                                lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")
                            intro = (
                                "Here are the most relevant web results I found:"
                                if lang != "de"
                                else "Hier sind die relevantesten Web-Ergebnisse, die ich gefunden habe:"
                            )
                            tool_based_text = intro + "\n\n" + "\n\n".join(lines)
                    except Exception:
                        tool_based_text = None

                if tool_based_text is None:
                    # Prioritize utility tools over web search in fallback
                    for tool_name in ["calculator_tool", "datetime_tool", "file_ops_tool", "extract_webpage_mcp", "web_search_mcp"]:
                        candidate = str(tool_results.get(tool_name, "")).strip()
                        if candidate and not candidate.lower().startswith("tool execution failed"):
                            tool_based_text = candidate
                            break

                if tool_based_text:
                    prefix = (
                        "Hier ist das Ergebnis aus den ausgefuehrten Tools:\n\n"
                        if lang == "de"
                        else "Here is the result from the executed tools:\n\n"
                    )
                    response_text = f"{prefix}{tool_based_text[:2500]}"
                else:
                    response_text = (
                        "Entschuldigung, ich habe intern Werkzeuge ausgefuehrt, aber keine lesbare Antwort erhalten. "
                        "Bitte formuliere deine Frage kurz neu."
                        if lang == "de"
                        else "Sorry, I executed tools internally but did not receive a readable answer. "
                        "Please rephrase your question briefly."
                    )
        
        # Remove any URLs that were not present in tool results
        if response_text:
            import re

            def _filter_urls(text: str) -> str:
                urls = re.findall(r"https?://[^\s)]+", text)
                if not urls:
                    return text
                allowed = allowed_urls
                removed = 0
                for url in urls:
                    if url not in allowed:
                        text = text.replace(url, "source omitted")
                        removed += 1
                if removed:
                    logger.info("Removed untrusted URLs from response", removed=removed)
                return text

            response_text = _filter_urls(response_text)

        # Honor strict literal-response directives when explicitly requested.
        exact_reply = extract_exact_reply_directive(user_msg)
        if exact_reply and response_text != exact_reply:
            logger.info(
                "Enforcing exact reply directive",
                requested=exact_reply,
                original_preview=(response_text or "")[:120],
            )
            response_text = exact_reply

        output_tokens = estimate_tokens(response_text)
        node_tokens = input_tokens + output_tokens
        require_tokens(node_tokens, available_response_budget, "Response node")
        if enforce_node_hard_limit and node_tokens > node_budget:
            raise TokenBudgetExceeded(
                f"Response node exceeds per-node budget: {node_tokens}/{node_budget}"
            )

        tokens_used = (state.get("tokens_used") or 0) + node_tokens

        track_tokens(fork_name, model_name, "respond", node_tokens)
        
        state["tokens_used"] = tokens_used
        state["turn_tokens_used"] = (state.get("turn_tokens_used") or 0) + node_tokens
        state["input_tokens"] = (state.get("input_tokens") or 0) + input_tokens
        state["output_tokens"] = (state.get("output_tokens") or 0) + output_tokens
        state["provider_used"] = provider
        state["model_used"] = model_name
        state["sources"] = collected_sources
        # Append assistant message
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": response_text})
        state["messages"] = msgs
        
        logger.info(
            "Response generated",
            tokens=input_tokens + output_tokens,
            total_tokens=tokens_used,
            model_used=model_name
        )

    return state


def node_summarize(state: GraphState) -> GraphState:
    config = load_core_config()
    lang = state.get("language") or config.fork.language
    fork_name = getattr(config.fork, "name", "default-fork")
    user_settings = state.get("user_settings", {})
    preferred_model = user_settings.get("preferred_model")
    
    logger.info("Summarizing conversation", fork_name=fork_name)

    # Keep digest_chain normalized even when compression does not generate a new summary.
    try:
        features = load_features_config()
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

    model = safe_get_model(config, lang, preferred_model=preferred_model)
    provider, model_name = resolve_initial_model_metadata(config, lang, preferred_model=preferred_model)

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

    budget_ctx = get_budget_context(state, config.tokens)
    sum_budget = get_node_budget(config.tokens, "summarization_node", budget_ctx["per_turn_budget"])
    enforce_node_hard_limit = per_node_hard_limit_enabled(config.tokens)
    available_budget = min(sum_budget, budget_ctx["turn_remaining"]) if enforce_node_hard_limit else budget_ctx["turn_remaining"]

    input_tokens = estimate_tokens(prompt)
    require_tokens(input_tokens, available_budget, "Summarization input")

    with track_node_execution(fork_name, "summarize"):
        try:
            summary, provider, model_name, _ = _invoke_with_model_fallback(
                config=config,
                language=lang,
                payload=prompt,
                initial_model=model,
                initial_provider=provider,
                initial_model_name=model_name,
                preferred_model=preferred_model,
            )
            track_llm_call(fork_name, provider, model_name, "success")
        except _ModelInvokeError as e:
            track_llm_call(fork_name, e.provider, e.model_name, "error")
            logger.warning(
                "Summarization LLM call failed, using fallback",
                error=str(e),
                provider=e.provider,
                model_name=e.model_name,
            )
            summary = f"LLM: {prompt}"
        except Exception as e:
            track_llm_call(fork_name, provider, model_name, "error")
            logger.warning("Summarization LLM call failed, using fallback", error=str(e))
            summary = f"LLM: {prompt}"
        
        output_tokens = estimate_tokens(summary)
        node_tokens = input_tokens + output_tokens
        require_tokens(node_tokens, available_budget, "Summarization node")
        if enforce_node_hard_limit and node_tokens > sum_budget:
            raise TokenBudgetExceeded(
                f"Summarization node exceeds per-node budget: {node_tokens}/{sum_budget}"
            )

        tokens_used = (state.get("tokens_used") or 0) + node_tokens

        track_tokens(fork_name, model_name, "summarize", node_tokens)
        
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
    fork_name = getattr(config.fork, "name", "default-fork")
    features_config = load_features_config()
    
    user_id = state.get("user_id")
    logger.info("Updating memory", user_id=user_id, fork_name=fork_name)

    if not getattr(features_config, "long_term_memory", False):
        logger.info("Long-term memory disabled via features flag", fork_name=fork_name)
        return state

    # Trivial exchange filter: skip memory write for low-content turns.
    _current_user_msg = ""
    for _msg in reversed(state.get("messages", [])):
        if isinstance(_msg, dict) and _msg.get("role") == "user":
            _current_user_msg = _msg.get("content", "")
            break
    _trivial_patterns = {
        "ok", "okay", "thanks", "thank you", "yes", "no", "sure", "got it",
        "hi", "hello", "bye", "goodbye", "great", "perfect", "alright",
    }
    _stripped = _current_user_msg.strip().lower().rstrip("!.,?")
    if len(_stripped) < 5 or (len(_stripped) < 20 and _stripped in _trivial_patterns):
        logger.info("Skipping memory update for trivial exchange", user_id=user_id, preview=_stripped[:30])
        return state

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

    budget_ctx = get_budget_context(state, config.tokens)
    upd_budget = get_node_budget(config.tokens, "update_memory", budget_ctx["per_turn_budget"])
    enforce_node_hard_limit = per_node_hard_limit_enabled(config.tokens)
    update_tokens = estimate_tokens(summary_text)
    available_budget = min(upd_budget, budget_ctx["turn_remaining"]) if enforce_node_hard_limit else budget_ctx["turn_remaining"]
    require_tokens(update_tokens, available_budget, "Update memory")
    if enforce_node_hard_limit and update_tokens > upd_budget:
        raise TokenBudgetExceeded(
            f"Update memory exceeds per-node budget: {update_tokens}/{upd_budget}"
        )

    with track_node_execution(fork_name, "update_memory"):
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
            track_memory_operation(fork_name, "update", "success")
            logger.info("Memory updated successfully", user_id=user_id)
            return result
        except Exception as e:
            track_memory_operation(fork_name, "update", "error")
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
        initialize_performance_nodes()
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
    
    # Add performance optimization nodes
    graph.add_node("memory_query_cache", memory_query_cache_node_sync)
    graph.add_node("memory_cache_store", memory_cache_store_node_sync)
    graph.add_node("compress_conversation", conversation_compression_node_sync)
    graph.add_node("cache_stats", cache_stats_node_sync)

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
                extra={
                    "invalid_mode": mode,
                    "candidates": len(candidates),
                    "mode_reason": mode_reason,
                },
            )
            return "structured"
        
        logger.info(
            "Tool strategy routing: routing to strategy node",
            extra={
                "mode": mode,
                "mode_reason": mode_reason,
                "candidates": len(candidates),
                "candidate_names": [c.get("name", "unknown") for c in candidates],
            },
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

    # Crew conditional routing from convergence point (secondary path — most crew queries
    # are routed at START, but this handles any edge case where a crew is signalled after tools).
    def route_to_crew(state):
        """Route to appropriate crew or standard flow."""
        if route_to_research_crew(state):
            return "research_crew"
        elif route_to_analytics_crew(state):
            return "analytics_crew"
        elif route_to_code_generation_crew(state):
            return "code_generation_crew"
        elif route_to_planning_crew(state):
            return "planning_crew"
        else:
            return "memory_query_cache"
    
    graph.add_conditional_edges(
        "after_tool_call",
        route_to_crew,
        {
            "research_crew": "research_crew",
            "analytics_crew": "analytics_crew",
            "code_generation_crew": "code_generation_crew",
            "planning_crew": "planning_crew",
            "memory_query_cache": "memory_query_cache",
        }
    )
    
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

    checkpointer = build_checkpointer(config=load_core_config())
    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)
