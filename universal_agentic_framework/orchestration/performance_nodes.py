"""LangGraph nodes for performance optimization.

Integrates caching and summarization into the graph execution pipeline.
"""

import logging
from typing import Optional, Any
import os

from universal_agentic_framework.caching import (
    CacheManager,
    RedisCacheBackend,
    MemoryCacheBackend,
)
from universal_agentic_framework.memory.summarization import ConversationSummarizer

logger = logging.getLogger(__name__)


# Global instances
_cache_manager: Optional[CacheManager] = None
_summarizer: Optional[ConversationSummarizer] = None


def initialize_performance_nodes(llm_factory=None):
    """Initialize cache and summarization nodes.
    
    Args:
        llm_factory: Optional LLM factory for summarization
    """
    global _cache_manager, _summarizer
    
    # Initialize cache
    use_redis = os.getenv("REDIS_URL") is not None
    if use_redis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            backend = RedisCacheBackend(redis_url)
            # Validate connectivity eagerly so tests/dev environments can fall back reliably.
            import asyncio
            _coro = backend._ensure_connected()
            try:
                asyncio.run(_coro)
            except RuntimeError:
                # Already inside a running event loop (e.g. uvicorn) — close the
                # coroutine explicitly to avoid "never awaited" RuntimeWarning.
                # Connectivity will be checked lazily on first cache operation.
                _coro.close()
            _cache_manager = CacheManager(backend)
            logger.info("Initialized Redis cache for graph")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}, using memory cache")
            _cache_manager = CacheManager(MemoryCacheBackend())
    else:
        _cache_manager = CacheManager(MemoryCacheBackend())
        logger.info("Initialized memory cache for graph")
    
    # Initialize summarizer
    _summarizer = ConversationSummarizer(llm_factory)
    logger.info("Initialized conversation summarizer")


def get_cache_manager() -> CacheManager:
    """Get cache manager, initializing if needed."""
    global _cache_manager
    if _cache_manager is None:
        initialize_performance_nodes()
    return _cache_manager


def get_summarizer() -> ConversationSummarizer:
    """Get summarizer, initializing if needed."""
    global _summarizer
    if _summarizer is None:
        initialize_performance_nodes()
    return _summarizer


async def memory_query_cache_node(state: dict) -> dict:
    """Node for caching memory query results.
    
    Checks cache before executing memory query, stores result after.
    This is a wrapper node that should precede memory retrieval.
    
    State keys used:
        - user_id: User identifier
        - loaded_memory: Preloaded memory (output of this node)
        - messages: Conversation messages for context
    
    Returns:
        Updated state with loaded_memory
    """
    try:
        cache = get_cache_manager()
        user_id = state.get("user_id", "unknown")
        query = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""

        if not query:
            logger.debug("No query found for memory cache")
            return state

        # Check cache
        cached_results = await cache.get_memory_query(user_id, query)
        if cached_results:
            logger.info(f"Memory query cache hit for user {user_id}")
            state["loaded_memory"] = cached_results
            return state
        
        logger.debug(f"Memory query cache miss for user {user_id}")
        
    except Exception as e:
        logger.warning(f"Memory cache node error: {e}")
    
    return state


async def memory_cache_store_node(state: dict) -> dict:
    """Node for storing memory query results in cache.
    
    Should follow memory retrieval node. Caches the loaded_memory
    for future similar queries.
    
    State keys used:
        - user_id: User identifier
        - loaded_memory: Memory results to cache
        - messages: For query context
    
    Returns:
        Unchanged state
    """
    try:
        cache = get_cache_manager()
        user_id = state.get("user_id", "unknown")
        memory = state.get("loaded_memory")
        query = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""

        if not memory or not query:
            return state

        # Store in cache
        success = await cache.set_memory_query(user_id, query, memory)
        if success:
            logger.debug(f"Cached memory results for user {user_id}")
    
    except Exception as e:
        logger.warning(f"Memory store cache node error: {e}")
    
    return state


# Number of most-recent messages compress_conversation() preserves verbatim. Kept in
# sync with ConversationSummarizer.compress_conversation(keep_recent_count=...).
_KEEP_RECENT_COUNT = 5


def _resolve_context_window(state: dict) -> int:
    """Resolve the chat model's real context window for the compression threshold.

    Precedence (highest first):
      1. Profile override ``llm.roles.chat.context_window_tokens``.
      2. Capability-probe snapshot forwarded into graph state, matched to the model
         that produced the last response (``state["model_used"]``).
      3. Conservative 32768 fallback.

    Never uses ``max_tokens`` — that is the output cap, not the context window.
    """
    try:
        from universal_agentic_framework.config import load_core_config as _load_cfg
        _cfg = _load_cfg()
        config_ctx = getattr(_cfg.llm.roles.chat, "context_window_tokens", None)
        if config_ctx:
            return int(config_ctx)
    except Exception:
        pass

    # Probe snapshot: the adapter forwards context_window_tokens (top-level on the
    # graph-state row; DB rows nest it under metadata). Prefer the row matching the
    # model that actually answered; otherwise take any probed window.
    model_used = str(state.get("model_used") or "")
    fallback_probe_ctx: Optional[int] = None
    for row in state.get("llm_capability_probes") or []:
        try:
            ctx = row.get("context_window_tokens") or (row.get("metadata") or {}).get(
                "context_window_tokens"
            )
            if not ctx:
                continue
            ctx = int(ctx)
        except Exception:
            continue
        if model_used and str(row.get("model_name") or "") == model_used:
            return ctx
        if fallback_probe_ctx is None:
            fallback_probe_ctx = ctx
    if fallback_probe_ctx is not None:
        return fallback_probe_ctx

    return 32768


async def compress_state(state: dict, force: bool = False) -> dict:
    """Core compression logic shared by the auto-compression graph node and the manual
    /compact endpoint.

    Args:
        state: Graph state dict containing at least ``messages`` and ``user_id``.
        force: When True, skip the token-fill threshold check and always attempt
            compression (still a no-op when there are too few messages to compress).

    Returns:
        Updated state. ``state["last_compression_status"]`` is set to one of
        ``"ok"`` (history compressed), ``"skipped"`` (nothing to compress), or
        ``"error"`` (summary generation failed — history left untouched).
    """
    state["last_compression_status"] = "skipped"

    try:
        summarizer = get_summarizer()
        messages = state.get("messages", [])
        user_id = state.get("user_id", "unknown")

        # Compressing a conversation at or below the keep-recent window is a no-op.
        if len(messages) <= _KEEP_RECENT_COUNT:
            return state

        if not force:
            # Fire when the real prompt size crosses 75% of the context window. Prefer
            # the provider-reported prompt tokens of the last respond call (exactly what
            # the context-ring shows); fall back to a chars/4 estimate when absent.
            context_window = _resolve_context_window(state)
            compression_threshold = int(context_window * 0.75)
            fill = state.get("last_input_tokens") or 0
            if fill <= 0:
                fill = summarizer.calculate_conversation_tokens(messages)
            if fill <= compression_threshold:
                return state

        logger.info(f"Compressing conversation for user {user_id} (force={force})")

        compressed = await summarizer.compress_conversation(messages, user_id)

        if len(compressed) < len(messages):
            savings = summarizer.calculate_savings(len(messages), len(compressed))
            logger.info(
                f"Compression savings for {user_id}: "
                f"removed {savings['messages_removed']} messages, "
                f"saved ~{savings['estimated_tokens_saved']} tokens"
            )
            state["messages"] = compressed
            state["digest_chain"] = summarizer.extract_digest_chain(compressed)
            state["tokens_used"] = summarizer.calculate_conversation_tokens(compressed)
            state["last_compression_status"] = "ok"
        else:
            # compress_conversation returns the input unchanged when summary
            # generation fails (it never truncates without a summary). Since we
            # already gated on len > keep_recent_count above, an unchanged result
            # here means the summary LLM failed — surface it as an error.
            logger.warning(f"Compression failed for {user_id}: summary generation did not produce a digest")
            state["last_compression_status"] = "error"

    except Exception as e:
        logger.warning(f"Compression error: {e}")
        state["last_compression_status"] = "error"

    return state


async def conversation_compression_node(state: dict) -> dict:
    """Node for compressing conversations to manage token usage.

    Checks if conversation has grown too large and summarizes old messages.

    State keys used:
        - messages: Conversation messages
        - user_id: User identifier
        - tokens_used: Current token count (output: updated after compression)

    Returns:
        Updated state with optionally compressed messages
    """
    return await compress_state(state, force=False)


async def cache_stats_node(state: dict) -> dict:
    """Node for tracking cache statistics.
    
    Periodically logs cache performance metrics for monitoring.
    Should be called infrequently (e.g., every N messages).
    
    Returns:
        Unchanged state
    """
    try:
        cache = get_cache_manager()
        stats = cache.get_stats()
        if stats["total_requests"] > 0:
            logger.info(
                f"Cache stats - "
                f"Hits: {stats['hits']}, "
                f"Misses: {stats['misses']}, "
                f"Hit Rate: {stats['hit_rate_percent']:.1f}%, "
                f"Errors: {stats['errors']}"
            )
    except Exception as e:
        logger.warning(f"Cache stats error: {e}")
    
    return state


# NOTE: these nodes are registered directly as async coroutines in
# graph_builder.build_graph() and awaited on the event loop under GRAPH.ainvoke().
# The previous sync wrappers (memory_query_cache_node_sync, …) and the
# _run_async_node_sync thread-bridge were removed when /invoke switched to ainvoke —
# they only existed to call these coroutines from a synchronous GRAPH.invoke().
