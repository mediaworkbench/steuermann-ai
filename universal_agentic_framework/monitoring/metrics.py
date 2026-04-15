"""Prometheus metrics for monitoring the framework."""

from contextlib import contextmanager
from typing import Any, Generator
import time
from prometheus_client import Counter, Histogram, Gauge, Info


# Graph execution metrics
GRAPH_REQUESTS_TOTAL = Counter(
    "langgraph_requests_total",
    "Total number of graph invocations",
    ["fork_name", "status"],  # status: success, error
)

GRAPH_REQUEST_DURATION = Histogram(
    "langgraph_request_duration_seconds",
    "Duration of graph request processing",
    ["fork_name"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Token usage metrics
TOKENS_USED_TOTAL = Counter(
    "langgraph_tokens_used_total",
    "Total tokens consumed",
    ["fork_name", "model", "node"],
)

# Session metrics
ACTIVE_SESSIONS = Gauge(
    "langgraph_active_sessions",
    "Currently active sessions",
    ["fork_name"],
)

# Node execution metrics
NODE_EXECUTION_DURATION = Histogram(
    "langgraph_node_duration_seconds",
    "Duration of individual node execution",
    ["fork_name", "node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# LLM call metrics
LLM_CALLS_TOTAL = Counter(
    "langgraph_llm_calls_total",
    "Total LLM API calls",
    ["fork_name", "provider", "model", "status"],
)

# Memory operation metrics
MEMORY_OPERATIONS_TOTAL = Counter(
    "langgraph_memory_operations_total",
    "Total memory operations",
    ["fork_name", "operation", "status"],  # operation: load, update, query
)

# Memory analytics metrics
MEMORY_IMPORTANCE_RANKING_DURATION = Histogram(
    "langgraph_memory_importance_ranking_duration_seconds",
    "Duration of importance-based memory reranking",
    ["fork_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

MEMORY_CO_OCCURRENCE_GRAPH_NODES = Gauge(
    "langgraph_memory_co_occurrence_graph_nodes",
    "Number of nodes in the memory co-occurrence graph",
    ["fork_name"],
)

MEMORY_CO_OCCURRENCE_GRAPH_EDGES = Gauge(
    "langgraph_memory_co_occurrence_graph_edges",
    "Number of edges in the memory co-occurrence graph",
    ["fork_name"],
)

MEMORY_QUALITY_SCORE = Histogram(
    "langgraph_memory_quality_score",
    "Distribution of memory importance scores",
    ["fork_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MEMORY_RELATED_RETRIEVED = Histogram(
    "langgraph_memory_related_retrieved",
    "Number of related memories retrieved via co-occurrence linking",
    ["fork_name"],
    buckets=(0, 1, 2, 3, 5, 10, 15, 20),
)

MEMORY_AGE_DAYS = Histogram(
    "langgraph_memory_age_days",
    "Age of retrieved memories in days",
    ["fork_name"],
    buckets=(0, 1, 7, 14, 30, 60, 90, 180, 365),
)

# Cache metrics
CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Total cache hits across all cache types",
    ["cache_type", "fork_name"],  # cache_type: llm, memory, crew, semantic
)

CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Total cache misses across all cache types",
    ["cache_type", "fork_name"],
)

CACHE_ERRORS_TOTAL = Counter(
    "cache_errors_total",
    "Total cache operation errors",
    ["cache_type", "fork_name", "operation"],  # operation: get, set, delete
)

CACHE_HIT_RATE = Gauge(
    "cache_hit_rate",
    "Cache hit rate as percentage (0-100)",
    ["cache_type", "fork_name"],
)

CACHE_OPERATION_DURATION = Histogram(
    "cache_operation_duration_seconds",
    "Duration of cache operations",
    ["cache_type", "fork_name", "operation"],  # operation: get, set
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

EMBEDDING_GENERATION_DURATION = Histogram(
    "embedding_generation_duration_seconds",
    "Duration of embedding generation for semantic queries",
    ["fork_name"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

EMBEDDING_CACHE_SIZE = Gauge(
    "embedding_cache_size",
    "Number of embeddings in local cache",
    ["fork_name"],
)

SEMANTIC_SIMILARITY_MATCHES = Counter(
    "semantic_similarity_matches_total",
    "Total semantic similarity cache matches",
    ["fork_name", "crew_name"],
)

# Vector database metrics
VECTOR_DB_SEARCH_DURATION = Histogram(
    "vector_db_search_duration_seconds",
    "Duration of vector database similarity search",
    ["fork_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

VECTOR_DB_COLLECTION_SIZE = Gauge(
    "vector_db_collection_size",
    "Number of embeddings stored in Qdrant collection",
    ["fork_name", "collection_name"],
)

VECTOR_DB_CLEANUP_TOTAL = Counter(
    "vector_db_cleanup_total",
    "Total number of expired embeddings cleaned up",
    ["fork_name"],
)

VECTOR_DB_FALLBACK_TOTAL = Counter(
    "vector_db_fallback_total",
    "Total fallbacks to in-memory search when Qdrant unavailable",
    ["fork_name", "reason"],  # reason: error, unavailable, timeout
)

CACHE_SIZE_BYTES = Gauge(
    "cache_size_bytes",
    "Approximate cache size in bytes",
    ["cache_type", "fork_name"],
)

# ── Crew execution metrics ──────────────────────────────────────────────

CREW_EXECUTION_DURATION = Histogram(
    "crew_execution_duration_seconds",
    "Duration of individual crew execution (including retries)",
    ["fork_name", "crew_name", "status"],  # status: success, error, timeout
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

CREW_EXECUTIONS_TOTAL = Counter(
    "crew_executions_total",
    "Total crew execution attempts",
    ["fork_name", "crew_name", "status"],  # status: success, error, timeout
)

CREW_RETRIES_TOTAL = Counter(
    "crew_retries_total",
    "Total crew retry attempts",
    ["fork_name", "crew_name"],
)

CREW_TIMEOUTS_TOTAL = Counter(
    "crew_timeouts_total",
    "Total crew execution timeouts",
    ["fork_name", "crew_name"],
)

CREW_CHAIN_EXECUTION_DURATION = Histogram(
    "crew_chain_execution_duration_seconds",
    "Duration of a full crew-chain pipeline",
    ["fork_name", "chain_name"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

CREW_PARALLEL_EXECUTION_DURATION = Histogram(
    "crew_parallel_execution_duration_seconds",
    "Duration of parallel crew execution batch",
    ["fork_name"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

CREW_VALIDATION_FAILURES = Counter(
    "crew_validation_failures_total",
    "Total crew result validation failures",
    ["fork_name", "crew_name"],
)

# ── Attachment handling metrics ─────────────────────────────────────────

ATTACHMENTS_INJECTED_TOTAL = Counter(
    "langgraph_attachments_injected_total",
    "Total attachment contexts injected into prompts",
    ["fork_name"],
)

ATTACHMENTS_NONE_TOTAL = Counter(
    "langgraph_attachments_none_total",
    "Total requests with no attachments",
    ["fork_name"],
)

ATTACHMENT_REFUSAL_RETRIES_TOTAL = Counter(
    "langgraph_attachment_refusal_retries_total",
    "Total times model falsely claimed attachments unavailable (retry triggered)",
    ["fork_name"],
)

ATTACHMENT_REFUSAL_RETRIES_SUCCESS_TOTAL = Counter(
    "langgraph_attachment_refusal_retries_success_total",
    "Total successful correction retries after attachment refusal",
    ["fork_name"],
)

# ── Workspace operation metrics ────────────────────────────────────────

WORKSPACE_CREATED_TOTAL = Counter(
    "langgraph_workspace_created_total",
    "Total conversation workspace creations",
    ["fork_name"],
)

WORKSPACE_WRITE_ALLOWED_TOTAL = Counter(
    "langgraph_workspace_write_allowed_total",
    "Total allowed workspace write operations",
    ["fork_name"],
)

WORKSPACE_WRITE_DENIED_TOTAL = Counter(
    "langgraph_workspace_write_denied_total",
    "Total denied workspace write operations",
    ["fork_name"],
)

WORKSPACE_INTENT_DENIED_TOTAL = Counter(
    "langgraph_workspace_intent_denied_total",
    "Total workspace intent denials grouped by operation and reason",
    ["fork_name", "operation", "reason"],
)

WORKSPACE_REVISED_COPY_CREATED_TOTAL = Counter(
    "langgraph_workspace_revised_copy_created_total",
    "Total revised workspace copies created",
    ["fork_name"],
)

WORKSPACE_CLEANUP_DELETED_TOTAL = Counter(
    "langgraph_workspace_cleanup_deleted_total",
    "Total workspace files/directories deleted by cleanup",
    ["fork_name"],
)

PROFILE_ID_MISMATCH_TOTAL = Counter(
    "langgraph_profile_id_mismatch_total",
    "Total chat responses where reported profile_id mismatched active deployment profile",
    ["fork_name", "active_profile_id", "reported_profile_id"],
)

# System info
SYSTEM_INFO = Info(
    "langgraph_system",
    "System information",
)


@contextmanager
def track_graph_request(fork_name: str) -> Generator[dict[str, Any], None, None]:
    """Context manager to track graph request execution.
    
    Usage:
        with track_graph_request("medical-ai-de") as ctx:
            result = graph.invoke(...)
            ctx["status"] = "success"
    """
    start_time = time.time()
    ctx = {"status": "error"}  # Default to error
    
    try:
        yield ctx
    finally:
        duration = time.time() - start_time
        GRAPH_REQUEST_DURATION.labels(fork_name=fork_name).observe(duration)
        GRAPH_REQUESTS_TOTAL.labels(
            fork_name=fork_name,
            status=ctx.get("status", "error")
        ).inc()


@contextmanager
def track_node_execution(fork_name: str, node: str) -> Generator[None, None, None]:
    """Context manager to track individual node execution.
    
    Usage:
        with track_node_execution("medical-ai-de", "load_memory"):
            state = load_memory_node(state)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        NODE_EXECUTION_DURATION.labels(
            fork_name=fork_name,
            node=node
        ).observe(duration)


def track_tokens(fork_name: str, model: str, node: str, tokens: int) -> None:
    """Track token usage.
    
    Args:
        fork_name: Name of the fork
        model: LLM model name
        node: Node that consumed tokens
        tokens: Number of tokens used
    """
    TOKENS_USED_TOTAL.labels(
        fork_name=fork_name,
        model=model,
        node=node
    ).inc(tokens)


def track_llm_call(
    fork_name: str,
    provider: str,
    model: str,
    status: str = "success"
) -> None:
    """Track LLM API call.
    
    Args:
        fork_name: Name of the fork
        provider: LLM provider (ollama, openai, anthropic)
        model: Model name
        status: Call status (success, error, timeout)
    """
    LLM_CALLS_TOTAL.labels(
        fork_name=fork_name,
        provider=provider,
        model=model,
        status=status
    ).inc()


def track_memory_operation(
    fork_name: str,
    operation: str,
    status: str = "success"
) -> None:
    """Track memory operation.
    
    Args:
        fork_name: Name of the fork
        operation: Operation type (load, update, query)
        status: Operation status (success, error)
    """
    MEMORY_OPERATIONS_TOTAL.labels(
        fork_name=fork_name,
        operation=operation,
        status=status
    ).inc()


def track_workspace_created(fork_name: str) -> None:
    WORKSPACE_CREATED_TOTAL.labels(fork_name=fork_name).inc()


def track_workspace_write_allowed(fork_name: str) -> None:
    WORKSPACE_WRITE_ALLOWED_TOTAL.labels(fork_name=fork_name).inc()


def track_workspace_write_denied(fork_name: str) -> None:
    WORKSPACE_WRITE_DENIED_TOTAL.labels(fork_name=fork_name).inc()


def track_workspace_intent_denied(fork_name: str, operation: str, reason: str) -> None:
    WORKSPACE_INTENT_DENIED_TOTAL.labels(
        fork_name=fork_name,
        operation=operation,
        reason=reason,
    ).inc()


def track_workspace_revised_copy_created(fork_name: str) -> None:
    WORKSPACE_REVISED_COPY_CREATED_TOTAL.labels(fork_name=fork_name).inc()


def track_workspace_cleanup_deleted(fork_name: str, deleted_count: int) -> None:
    if deleted_count <= 0:
        return
    WORKSPACE_CLEANUP_DELETED_TOTAL.labels(fork_name=fork_name).inc(deleted_count)


def track_profile_id_mismatch(
    fork_name: str,
    active_profile_id: str,
    reported_profile_id: str,
) -> None:
    PROFILE_ID_MISMATCH_TOTAL.labels(
        fork_name=fork_name,
        active_profile_id=active_profile_id,
        reported_profile_id=reported_profile_id,
    ).inc()


def update_active_sessions(fork_name: str, count: int) -> None:
    """Update active session count.
    
    Args:
        fork_name: Name of the fork
        count: Current number of active sessions
    """
    ACTIVE_SESSIONS.labels(fork_name=fork_name).set(count)


def initialize_system_info(version: str, environment: str) -> None:
    """Initialize system information metric.
    
    Args:
        version: Framework version
        environment: Deployment environment (dev, staging, production)
    """
    SYSTEM_INFO.info({
        "version": version,
        "environment": environment,
    })


def track_memory_importance_ranking(fork_name: str, duration: float) -> None:
    """Track duration of importance-based memory reranking.
    
    Args:
        fork_name: Name of the fork
        duration: Duration in seconds
    """
    MEMORY_IMPORTANCE_RANKING_DURATION.labels(fork_name=fork_name).observe(duration)


def track_memory_graph_statistics(fork_name: str, nodes: int, edges: int) -> None:
    """Track co-occurrence graph statistics.
    
    Args:
        fork_name: Name of the fork
        nodes: Number of nodes (unique memories) in graph
        edges: Number of edges (co-occurrence links) in graph
    """
    MEMORY_CO_OCCURRENCE_GRAPH_NODES.labels(fork_name=fork_name).set(nodes)
    MEMORY_CO_OCCURRENCE_GRAPH_EDGES.labels(fork_name=fork_name).set(edges)


def track_memory_quality(fork_name: str, importance_score: float) -> None:
    """Track individual memory quality score.
    
    Args:
        fork_name: Name of the fork
        importance_score: Computed importance score (0.0-1.0+)
    """
    MEMORY_QUALITY_SCORE.labels(fork_name=fork_name).observe(importance_score)


def track_related_memories(fork_name: str, count: int) -> None:
    """Track number of related memories retrieved.
    
    Args:
        fork_name: Name of the fork
        count: Number of related memories fetched via co-occurrence
    """
    MEMORY_RELATED_RETRIEVED.labels(fork_name=fork_name).observe(count)


def track_memory_age(fork_name: str, age_days: float) -> None:
    """Track age of retrieved memory.
    
    Args:
        fork_name: Name of the fork
        age_days: Age of memory in days
    """
    MEMORY_AGE_DAYS.labels(fork_name=fork_name).observe(age_days)


# Cache tracking functions

def track_cache_hit(cache_type: str, fork_name: str = "default") -> None:
    """Track cache hit.
    
    Args:
        cache_type: Type of cache (llm, memory, crew, semantic)
        fork_name: Name of the fork
    """
    CACHE_HITS_TOTAL.labels(cache_type=cache_type, fork_name=fork_name).inc()


def track_cache_miss(cache_type: str, fork_name: str = "default") -> None:
    """Track cache miss.
    
    Args:
        cache_type: Type of cache (llm, memory, crew, semantic)
        fork_name: Name of the fork
    """
    CACHE_MISSES_TOTAL.labels(cache_type=cache_type, fork_name=fork_name).inc()


def track_cache_error(
    cache_type: str,
    operation: str,
    fork_name: str = "default"
) -> None:
    """Track cache operation error.
    
    Args:
        cache_type: Type of cache (llm, memory, crew, semantic)
        operation: Operation type (get, set, delete)
        fork_name: Name of the fork
    """
    CACHE_ERRORS_TOTAL.labels(
        cache_type=cache_type,
        fork_name=fork_name,
        operation=operation
    ).inc()


def update_cache_hit_rate(
    cache_type: str,
    hit_rate: float,
    fork_name: str = "default"
) -> None:
    """Update cache hit rate gauge.
    
    Args:
        cache_type: Type of cache
        hit_rate: Hit rate as percentage (0-100)
        fork_name: Name of the fork
    """
    CACHE_HIT_RATE.labels(cache_type=cache_type, fork_name=fork_name).set(hit_rate)


@contextmanager
def track_cache_operation(
    cache_type: str,
    operation: str,
    fork_name: str = "default"
) -> Generator[None, None, None]:
    """Context manager to track cache operation duration.
    
    Usage:
        with track_cache_operation("crew", "get", "medical-ai"):
            result = await cache.get_crew_result(...)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        CACHE_OPERATION_DURATION.labels(
            cache_type=cache_type,
            fork_name=fork_name,
            operation=operation
        ).observe(duration)


def track_embedding_generation(fork_name: str, duration: float) -> None:
    """Track embedding generation duration.
    
    Args:
        fork_name: Name of the fork
        duration: Duration in seconds
    """
    EMBEDDING_GENERATION_DURATION.labels(fork_name=fork_name).observe(duration)


def update_embedding_cache_size(fork_name: str, size: int) -> None:
    """Update embedding cache size gauge.
    
    Args:
        fork_name: Name of the fork
        size: Number of embeddings in cache
    """
    EMBEDDING_CACHE_SIZE.labels(fork_name=fork_name).set(size)


def track_semantic_similarity_match(crew_name: str, fork_name: str = "default") -> None:
    """Track semantic similarity cache match.
    
    Args:
        crew_name: Name of the crew that matched
        fork_name: Name of the fork
    """
    SEMANTIC_SIMILARITY_MATCHES.labels(
        fork_name=fork_name,
        crew_name=crew_name
    ).inc()


def update_cache_size(
    cache_type: str,
    size_bytes: int,
    fork_name: str = "default"
) -> None:
    """Update cache size gauge.
    
    Args:
        cache_type: Type of cache
        size_bytes: Size in bytes
        fork_name: Name of the fork
    """
    CACHE_SIZE_BYTES.labels(cache_type=cache_type, fork_name=fork_name).set(size_bytes)


def track_cache_cleanup(
    removed_count: int,
    duration_seconds: float,
    cache_size_before: int,
    cache_size_after: int,
    fork_name: str = "default"
) -> None:
    """Track cache cleanup operation.
    
    Args:
        removed_count: Number of entries removed
        duration_seconds: Cleanup duration in seconds
        cache_size_before: Cache size before cleanup
        cache_size_after: Cache size after cleanup
        fork_name: Name of the fork
    """
    # You can add custom metrics here if needed
    # For now, we'll log the cleanup stats
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Cache cleanup completed: removed={removed_count}, "
        f"duration={duration_seconds:.2f}s, "
        f"size_before={cache_size_before}, size_after={cache_size_after}, "
        f"fork={fork_name}"
    )


# Vector database tracking functions

def track_vector_db_search(fork_name: str, duration: float) -> None:
    """Track vector database search duration.
    
    Args:
        fork_name: Name of the fork
        duration: Search duration in seconds
    """
    VECTOR_DB_SEARCH_DURATION.labels(fork_name=fork_name).observe(duration)


def update_vector_db_collection_size(
    fork_name: str,
    collection_name: str,
    size: int
) -> None:
    """Update vector database collection size.
    
    Args:
        fork_name: Name of the fork
        collection_name: Qdrant collection name
        size: Number of embeddings in collection
    """
    VECTOR_DB_COLLECTION_SIZE.labels(
        fork_name=fork_name,
        collection_name=collection_name
    ).set(size)


def track_vector_db_cleanup(fork_name: str, deleted_count: int) -> None:
    """Track vector database cleanup of expired embeddings.
    
    Args:
        fork_name: Name of the fork
        deleted_count: Number of expired embeddings deleted
    """
    VECTOR_DB_CLEANUP_TOTAL.labels(fork_name=fork_name).inc(deleted_count)


def track_vector_db_fallback(fork_name: str, reason: str) -> None:
    """Track fallback to in-memory search when Qdrant unavailable.
    
    Args:
        fork_name: Name of the fork
        reason: Fallback reason (error, unavailable, timeout)
    """
    VECTOR_DB_FALLBACK_TOTAL.labels(fork_name=fork_name, reason=reason).inc()


# ── Crew tracking helpers ───────────────────────────────────────────────

def track_crew_execution(
    fork_name: str,
    crew_name: str,
    duration: float,
    status: str = "success",
) -> None:
    """Track a single crew execution.

    Args:
        fork_name: Name of the fork.
        crew_name: Crew identifier (e.g. "research").
        duration: Execution duration in seconds.
        status: Outcome – "success", "error", or "timeout".
    """
    CREW_EXECUTION_DURATION.labels(
        fork_name=fork_name, crew_name=crew_name, status=status
    ).observe(duration)
    CREW_EXECUTIONS_TOTAL.labels(
        fork_name=fork_name, crew_name=crew_name, status=status
    ).inc()


def track_crew_retry(fork_name: str, crew_name: str) -> None:
    """Increment crew retry counter."""
    CREW_RETRIES_TOTAL.labels(fork_name=fork_name, crew_name=crew_name).inc()


def track_crew_timeout(fork_name: str, crew_name: str) -> None:
    """Increment crew timeout counter."""
    CREW_TIMEOUTS_TOTAL.labels(fork_name=fork_name, crew_name=crew_name).inc()


@contextmanager
def track_crew_chain(fork_name: str, chain_name: str) -> Generator[None, None, None]:
    """Context manager to track crew-chain execution duration.

    Usage:
        with track_crew_chain("medical-ai", "research_then_analytics"):
            result = chain.execute(initial_input)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        CREW_CHAIN_EXECUTION_DURATION.labels(
            fork_name=fork_name, chain_name=chain_name
        ).observe(duration)


@contextmanager
def track_crew_parallel(fork_name: str) -> Generator[None, None, None]:
    """Context manager to track parallel crew execution duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        CREW_PARALLEL_EXECUTION_DURATION.labels(fork_name=fork_name).observe(duration)


def track_crew_validation_failure(fork_name: str, crew_name: str) -> None:
    """Track crew result validation failure."""
    CREW_VALIDATION_FAILURES.labels(fork_name=fork_name, crew_name=crew_name).inc()
