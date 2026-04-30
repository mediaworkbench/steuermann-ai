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
    cache = get_cache_manager()
    
    try:
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
    cache = get_cache_manager()
    
    try:
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
    summarizer = get_summarizer()
    
    try:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "unknown")
        
        # Check if summarization needed
        if not summarizer.should_summarize(messages):
            return state
        
        logger.info(f"Compressing conversation for user {user_id}")
        
        # Compress conversation
        compressed = await summarizer.compress_conversation(messages, user_id)
        
        # Log savings
        savings = summarizer.calculate_savings(len(messages), len(compressed))
        logger.info(
            f"Compression savings for {user_id}: "
            f"removed {savings['messages_removed']} messages, "
            f"saved ~{savings['estimated_tokens_saved']} tokens"
        )
        
        state["messages"] = compressed

        # Persist rolling digest metadata for downstream memory updates.
        state["digest_chain"] = summarizer.extract_digest_chain(compressed)
        
        # Update token count estimate
        new_tokens = summarizer.calculate_conversation_tokens(compressed)
        state["tokens_used"] = new_tokens
        
    except Exception as e:
        logger.warning(f"Compression node error: {e}")
    
    return state


async def cache_stats_node(state: dict) -> dict:
    """Node for tracking cache statistics.
    
    Periodically logs cache performance metrics for monitoring.
    Should be called infrequently (e.g., every N messages).
    
    Returns:
        Unchanged state
    """
    cache = get_cache_manager()
    
    try:
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


# Synchronous wrappers for LangGraph integration (existing nodes are sync)
def _run_async_node_sync(async_node, skip_message: str, error_label: str, state: dict) -> dict:
    """Run an async node in sync context.

    If an event loop is already running in this thread, execute in a worker
    thread with its own event loop instead of skipping execution.
    """
    import asyncio
    import threading

    def _run_in_worker_thread() -> dict:
        result_box: dict[str, dict] = {}
        error_box: dict[str, Exception] = {}

        def _worker() -> None:
            try:
                result_box["state"] = asyncio.run(async_node(state))
            except Exception as thread_exc:  # pragma: no cover - defensive branch
                error_box["error"] = thread_exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("state", state)

    try:
        asyncio.get_running_loop()
        logger.debug("Event loop already running, executing async node in worker thread")
        return _run_in_worker_thread()
    except RuntimeError:
        # No running loop in this thread: safe to run coroutine to completion.
        pass

    try:
        return asyncio.run(async_node(state))
    except Exception as e:
        logger.warning(f"{error_label}: {e}")
        return state


def memory_query_cache_node_sync(state: dict) -> dict:
    """Sync wrapper for memory_query_cache_node."""
    return _run_async_node_sync(
        memory_query_cache_node,
        "Event loop already running, skipping cache check",
        "Sync wrapper error",
        state,
    )


def memory_cache_store_node_sync(state: dict) -> dict:
    """Sync wrapper for memory_cache_store_node."""
    return _run_async_node_sync(
        memory_cache_store_node,
        "Event loop already running, skipping cache store",
        "Sync wrapper error",
        state,
    )


def conversation_compression_node_sync(state: dict) -> dict:
    """Sync wrapper for conversation_compression_node."""
    return _run_async_node_sync(
        conversation_compression_node,
        "Event loop already running, skipping compression",
        "Sync wrapper error",
        state,
    )


def cache_stats_node_sync(state: dict) -> dict:
    """Sync wrapper for cache_stats_node."""
    return _run_async_node_sync(
        cache_stats_node,
        "Event loop already running, skipping stats",
        "Sync wrapper error",
        state,
    )


def add_performance_nodes_to_graph(graph_builder):
    """Add performance optimization nodes to graph.
    
    Inserts caching and compression nodes into the graph execution flow.
    
    Args:
        graph_builder: LangGraph GraphBuilder instance
        
    Returns:
        Updated graph_builder
    """
    try:
        # Add memory caching nodes (use sync wrappers)
        graph_builder.add_node("memory_query_cache", memory_query_cache_node_sync)
        graph_builder.add_node("memory_cache_store", memory_cache_store_node_sync)
        
        # Add compression node
        graph_builder.add_node("compress_conversation", conversation_compression_node_sync)
        
        # Add stats monitoring
        graph_builder.add_node("cache_stats", cache_stats_node_sync)
    except Exception as e:
        logger.warning(f"Error adding performance nodes: {e}")
    
    return graph_builder
