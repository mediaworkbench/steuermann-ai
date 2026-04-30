"""Caching layer for LLM responses and memory queries.

Implements Redis-backed caching for:
- LLM responses (question-answer pairs)
- Memory query results (semantic search results)
- Conversation summaries

Semantic search uses Qdrant vector database for efficient similarity matching.
"""

import json
import hashlib
import logging
from typing import Optional, Any, Dict, List, Tuple
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from universal_agentic_framework.monitoring.metrics import (
    track_cache_hit,
    track_cache_miss,
    track_cache_error,
    update_cache_hit_rate,
    track_cache_operation,
    track_embedding_generation,
    update_embedding_cache_size,
    track_semantic_similarity_match,
)
from universal_agentic_framework.embeddings import build_embedding_provider, EmbeddingProvider

try:
    from universal_agentic_framework.caching.vector_backend import QdrantCacheVectorBackend
    HAS_QDRANT_BACKEND = True
except ImportError:
    HAS_QDRANT_BACKEND = False

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear entire cache."""
        pass


class RedisCacheBackend(CacheBackend):
    """Redis-backed cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
        """
        import redis.asyncio as redis
        self.redis = None
        self.redis_url = redis_url
        self._initialized = False
        self._loop = None  # Track the event loop the connection belongs to
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established in the current event loop."""
        import asyncio
        current_loop = asyncio.get_running_loop()
        # If initialized but in a different event loop, reset the connection
        if self._initialized and self._loop is not current_loop:
            self._initialized = False
            self.redis = None
        if not self._initialized:
            try:
                try:
                    import redis.asyncio as redis
                except ImportError:
                    import redis
                self.redis = await redis.from_url(self.redis_url, decode_responses=True)
                await self.redis.ping()
                self._initialized = True
                self._loop = current_loop
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            await self._ensure_connected()
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in Redis with TTL."""
        try:
            await self._ensure_connected()
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl_seconds, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from Redis."""
        try:
            await self._ensure_connected()
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Redis database."""
        try:
            await self._ensure_connected()
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False
    
    async def cleanup(self) -> int:
        """Cleanup expired keys from Redis.
        
        Note: Redis automatically removes expired keys, so this is
        mainly for monitoring purposes.
        
        Returns:
            Number of keys cleaned up (always 0 for Redis)
        """
        # Redis handles TTL cleanup automatically via background threads
        # This method is here for interface consistency
        return 0


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: str = "LRU"
    ):
        """Initialize memory cache backend.
        
        Args:
            max_size: Maximum number of entries
            eviction_policy: Eviction policy name (LRU, LFU, FIFO, TTL, Random)
        """
        from universal_agentic_framework.caching.eviction import (
            create_eviction_policy,
            CacheEntry
        )
        
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.eviction_policy = create_eviction_policy(eviction_policy)
        self.stats = {
            "evictions": 0,
            "policy": self.eviction_policy.get_name()
        }
        
        logger.info(
            f"MemoryCacheBackend initialized: max_size={max_size}, "
            f"policy={self.eviction_policy.get_name()}"
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiry
            if entry.expiry is not None and time.time() >= entry.expiry:
                # Expired
                del self.cache[key]
                return None
            
            # Update policy metadata
            self.eviction_policy.on_get(key, entry)
            
            # Return value
            return entry.value
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in memory cache."""
        from universal_agentic_framework.caching.eviction import CacheEntry
        
        current_time = time.time()
        expiry = current_time + ttl_seconds if ttl_seconds else None
        
        # Create or update entry
        if key in self.cache:
            # Update existing entry
            entry = self.cache[key]
            entry.value = value
            entry.expiry = expiry
            entry.last_accessed = current_time
        else:
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                expiry=expiry,
                inserted_at=current_time,
                last_accessed=current_time,
                access_count=0
            )
            self.cache[key] = entry
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                await self._evict(1)
        
        # Notify policy
        self.eviction_policy.on_set(key, entry)
        
        return True
    
    async def _evict(self, count: int) -> int:
        """Evict entries using configured policy.
        
        Args:
            count: Number of entries to evict
            
        Returns:
            Number of entries actually evicted
        """
        if not self.cache:
            return 0
        
        # Select victims
        victims = self.eviction_policy.select_victims(self.cache, count)
        
        # Remove victims
        evicted = 0
        for key in victims:
            if key in self.cache:
                del self.cache[key]
                evicted += 1
        
        self.stats["evictions"] += evicted
        
        logger.debug(
            f"Evicted {evicted} entries using {self.eviction_policy.get_name()} policy"
        )
        
        return evicted
    
    async def delete(self, key: str) -> bool:
        """Delete from memory cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self) -> bool:
        """Clear memory cache."""
        self.cache.clear()
        return True
    
    async def cleanup(self) -> int:
        """Remove expired entries from memory cache.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        current_time = time.time()
        expired_keys = []
        
        for key, entry in list(self.cache.items()):
            if entry.expiry is not None and current_time >= entry.expiry:
                expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
                removed += 1
        
        return removed
    
    def get_eviction_stats(self) -> Dict[str, Any]:
        """Get eviction statistics.
        
        Returns:
            Dict with eviction stats
        """
        return {
            **self.stats,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0
        }


class CacheManager:
    """High-level cache manager with key generation and stats."""
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        fork_name: str = "default",
        use_vector_db: bool = True,
        qdrant_host: Optional[str] = None,
        qdrant_port: int = 6333,
        similarity_threshold: float = 0.85,
        enable_compression: bool = False,
        compression_threshold_kb: int = 1,
        embedding_model: str = "text-embedding-granite-embedding-278m-multilingual",
        embedding_dimension: int = 768,
        embedding_provider_type: str = "remote",
        embedding_remote_endpoint: Optional[str] = None,
    ):
        """Initialize cache manager.
        
        Args:
            backend: Cache backend (defaults to MemoryCacheBackend)
            fork_name: Name of the fork for metrics tracking
            use_vector_db: Whether to use Qdrant for semantic search
            qdrant_host: Qdrant server host (default: QDRANT_HOST env var or "localhost")
            qdrant_port: Qdrant server port
            similarity_threshold: Minimum cosine similarity for semantic matches
            enable_compression: Whether to enable cache compression (default: False)
            compression_threshold_kb: Minimum size to compress in KB (default: 1KB)
            embedding_model: Embedding model name
            embedding_dimension: Embedding dimension
            embedding_provider_type: Embedding provider type (remote-only)
            embedding_remote_endpoint: Required remote endpoint for embedding requests
        """
        import os
        self.backend = backend or MemoryCacheBackend()
        self.fork_name = fork_name
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
        self.similarity_threshold = similarity_threshold
        # Resolve Qdrant host: explicit arg > QDRANT_HOST env var > localhost
        qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", str(qdrant_port)))
        # Resolve embedding endpoint: explicit arg > EMBEDDING_SERVER env var
        embedding_remote_endpoint = embedding_remote_endpoint or os.getenv("EMBEDDING_SERVER")
        
        # Store embedding config for provider initialization
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.embedding_provider_type = embedding_provider_type
        self.embedding_remote_endpoint = embedding_remote_endpoint
        
        # Initialize vector backend for semantic search
        self.vector_backend: Optional[QdrantCacheVectorBackend] = None
        if use_vector_db and HAS_QDRANT_BACKEND:
            try:
                self.vector_backend = QdrantCacheVectorBackend(
                    collection_prefix="cache",
                    host=qdrant_host,
                    port=qdrant_port,
                    embedding_model=embedding_model,
                    dimension=embedding_dimension,
                    fork_name=fork_name,
                    similarity_threshold=similarity_threshold,
                    embedding_provider_type=embedding_provider_type,
                    embedding_remote_endpoint=embedding_remote_endpoint,
                )
                logger.info(f"Initialized Qdrant vector backend for cache (fork={fork_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant vector backend: {e}. Falling back to in-memory search.")
                self.vector_backend = None
        
        # Semantic matching cache
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_provider: Optional[EmbeddingProvider] = None
        
        # Initialize compressor for cache compression
        self.compressor: Optional[Any] = None
        if enable_compression:
            try:
                from universal_agentic_framework.caching.compression import create_compressor
                self.compressor = create_compressor(
                    threshold_kb=compression_threshold_kb,
                    level=6,
                    enabled=True
                )
                logger.info(f"Initialized cache compression: threshold={compression_threshold_kb}KB")
            except Exception as e:
                logger.warning(f"Failed to initialize cache compression: {e}")
                self.compressor = None
        
        # Update embedding cache size metric
        update_embedding_cache_size(fork_name, 0)
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        content = f"{prefix}:" + ":".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_embedding_provider(self) -> Optional[EmbeddingProvider]:
        """Lazy-load embedding provider for semantic matching."""
        if self._embedding_provider is None:
            try:
                self._embedding_provider = build_embedding_provider(
                    model_name=self.embedding_model,
                    dimension=self.embedding_dimension,
                    provider_type=self.embedding_provider_type,
                    remote_endpoint=self.embedding_remote_endpoint,
                )
            except Exception as e:
                logger.warning(f"Failed to load embedding provider: {e}")
                return None
        
        return self._embedding_provider
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for a query (with caching)."""
        # Check cache first
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        
        provider = self._get_embedding_provider()
        if provider is None:
            return None
        
        try:
            # Track embedding generation
            start_time = time.time()
            
            # Generate embedding
            embedding = provider.encode(query)
            
            # Record duration
            duration = time.time() - start_time
            track_embedding_generation(self.fork_name, duration)
            
            # Ensure it's a list (remote API returns list, local might return numpy)
            if not isinstance(embedding, list):
                embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Cache it
            self._embedding_cache[query] = embedding
            
            # Update cache size metric
            update_embedding_cache_size(self.fork_name, len(self._embedding_cache))
            
            # Don't cache more than 1000 embeddings to avoid memory bloat
            if len(self._embedding_cache) > 1000:
                # Remove oldest (first) entry (simple FIFO)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
                # Update metric after eviction
                update_embedding_cache_size(self.fork_name, len(self._embedding_cache))
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for query: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not HAS_NUMPY:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a ** 2 for a in vec1) ** 0.5
            norm2 = sum(b ** 2 for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        else:
            # Use numpy for better performance
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    async def get_llm_response(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 2048
    ) -> Optional[str]:
        """Get cached LLM response."""
        key = self._make_key("llm", model, prompt[:100], max_tokens)
        try:
            result = await self.backend.get(key)
            if result:
                # Decompress if compressed
                if self.compressor and isinstance(result, dict) and "is_compressed" in result:
                    result = self.compressor.decompress_from_metadata(result)
                
                self.stats["hits"] += 1
                logger.debug(f"LLM cache hit for {model}")
            else:
                self.stats["misses"] += 1
            return result
        except Exception as e:
            logger.warning(f"LLM cache get error: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_llm_response(
        self,
        model: str,
        prompt: str,
        response: str,
        max_tokens: int = 2048,
        ttl_seconds: int = 86400  # 24 hours
    ) -> bool:
        """Cache LLM response."""
        key = self._make_key("llm", model, prompt[:100], max_tokens)
        try:
            # Compress if compressor enabled
            data_to_store = response
            if self.compressor:
                data_to_store = self.compressor.compress_with_metadata(response)
            
            return await self.backend.set(key, data_to_store, ttl_seconds)
        except Exception as e:
            logger.warning(f"LLM cache set error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def get_memory_query(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> Optional[List[Dict]]:
        """Get cached memory query results."""
        key = self._make_key("memory", user_id, query[:100], top_k)
        try:
            result = await self.backend.get(key)
            if result:
                self.stats["hits"] += 1
                logger.debug(f"Memory cache hit for user {user_id}")
            else:
                self.stats["misses"] += 1
            return result
        except Exception as e:
            logger.warning(f"Memory cache get error: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_memory_query(
        self,
        user_id: str,
        query: str,
        results: List[Dict],
        top_k: int = 5,
        ttl_seconds: int = 3600  # 1 hour
    ) -> bool:
        """Cache memory query results."""
        key = self._make_key("memory", user_id, query[:100], top_k)
        try:
            return await self.backend.set(key, results, ttl_seconds)
        except Exception as e:
            logger.warning(f"Memory cache set error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def get_conversation_summary(
        self,
        user_id: str
    ) -> Optional[str]:
        """Get cached conversation summary."""
        key = self._make_key("summary", user_id)
        try:
            result = await self.backend.get(key)
            if result:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return result
        except Exception as e:
            logger.warning(f"Summary cache get error: {e}")
            self.stats["errors"] += 1
            return None

    async def _find_semantic_match(
        self,
        crew_name: str,
        user_id: str,
        query: str,
        language: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Find cached result for semantically similar query.
        
        Uses Qdrant vector database if available for efficient O(log n) search.
        Falls back to in-memory O(n) iteration if Qdrant unavailable.
        
        Args:
            crew_name: Name of the crew
            user_id: User ID
            query: Query string to match semantically
            language: Optional language code
            similarity_threshold: Minimum cosine similarity (0.0-1.0)
        
        Returns:
            Cached result if semantic match found above threshold, else None
        """
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return None
        
        # Try vector database (Qdrant) first - O(log n)
        if self.vector_backend:
            try:
                matches = self.vector_backend.search_similar(
                    embedding=query_embedding,
                    crew_name=crew_name,
                    user_id=user_id,
                    language=language or "unknown",
                    top_k=5,
                    similarity_threshold=similarity_threshold,
                )
                
                if matches:
                    best_match = matches[0]  # Highest similarity score
                    logger.debug(
                        f"Semantic match found via Qdrant for {crew_name} "
                        f"({user_id}): similarity={best_match['score']:.3f}"
                    )
                    track_semantic_similarity_match(crew_name, self.fork_name)
                    
                    # Resolve actual cached result using the original query
                    original_query = best_match.get("query", "")
                    if original_query:
                        match_key = self._make_key(
                            "crew",
                            crew_name,
                            user_id,
                            language or "",
                            original_query[:100],
                        )
                        cached_value = await self.backend.get(match_key)
                        if cached_value:
                            if isinstance(cached_value, dict) and "_embedding" in cached_value:
                                return cached_value.get("result")
                            return cached_value
                    
                    # If Qdrant match could not be resolved from backend
                    # (e.g. stale Qdrant entry), fall through to in-memory search
                    logger.debug(
                        f"Qdrant match for {crew_name} could not be resolved "
                        f"from cache backend, falling through to in-memory search"
                    )
            except Exception as e:
                logger.warning(f"Vector database search failed, falling back to memory: {e}")
                # Track fallback event
                from universal_agentic_framework.monitoring import metrics
                metrics.track_vector_db_fallback(self.fork_name, reason="error")
        
        # Fallback: In-memory O(n) iteration
        prefix = f"crew:{crew_name}:{user_id}:{language or ''}:"
        
        if isinstance(self.backend, MemoryCacheBackend):
            cache_dict = getattr(self.backend, "cache", {})
            for key, cached_data in cache_dict.items():
                if not key.startswith(prefix):
                    continue
                
                # Extract cached value
                value, _ = cached_data if isinstance(cached_data, tuple) else (cached_data, None)
                
                # Extract stored embedding if available
                if isinstance(value, dict) and "_embedding" in value:
                    stored_embedding = value.get("_embedding")
                    if stored_embedding:
                        similarity = self._cosine_similarity(
                            query_embedding,
                            stored_embedding
                        )
                        if similarity >= similarity_threshold:
                            logger.debug(
                                f"Semantic match found (in-memory) for {crew_name} "
                                f"({user_id}): similarity={similarity:.3f}"
                            )
                            track_semantic_similarity_match(crew_name, self.fork_name)
                            return value.get("result")
        
        return None

    async def get_crew_result(
        self,
        crew_name: str,
        user_id: str,
        query: str,
        language: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Get cached crew result with semantic matching fallback.
        
        Three-tier lookup:
        1. Exact key match (fast, specific)
        2. Semantic similarity search (slower, catches paraphrasing)
        3. Miss (return None)
        """
        key = self._make_key(
            "crew",
            crew_name,
            user_id,
            language or "",
            query[:100],
        )
        try:
            # Tier 1: Exact match
            result = await self.backend.get(key)
            if result:
                self.stats["hits"] += 1
                track_cache_hit("crew", self.fork_name)
                logger.debug(f"Crew cache HIT (exact) for {crew_name} ({user_id})")
                # Extract actual result if stored with metadata
                if isinstance(result, dict) and "_embedding" in result:
                    return result.get("result")
                return result
            
            # Tier 2: Semantic similarity
            semantic_result = await self._find_semantic_match(
                crew_name,
                user_id,
                query,
                language,
                similarity_threshold=similarity_threshold,
            )
            if semantic_result:
                self.stats["hits"] += 1
                track_cache_hit("semantic", self.fork_name)
                logger.debug(f"Crew cache HIT (semantic) for {crew_name} ({user_id})")
                return semantic_result
            
            # Tier 3: Miss
            self.stats["misses"] += 1
            track_cache_miss("crew", self.fork_name)
            return None
        except Exception as e:
            logger.warning(f"Crew cache get error: {e}")
            self.stats["errors"] += 1
            track_cache_error("crew", "get", self.fork_name)
            return None

    async def set_crew_result(
        self,
        crew_name: str,
        user_id: str,
        query: str,
        result: Dict[str, Any],
        language: Optional[str] = None,
        ttl_seconds: int = 3600,
    ) -> bool:
        """Cache crew result for a user and query.
        
        Stores result along with query embedding for semantic matching.
        If vector database available, also indexes embedding for fast lookup.
        """
        key = self._make_key(
            "crew",
            crew_name,
            user_id,
            language or "",
            query[:100],
        )
        try:
            # Get query embedding for semantic matching
            embedding = self._get_query_embedding(query)
            
            # Store in vector backend if available
            result_hash = self._make_key("result", crew_name, user_id, str(result))
            if embedding and self.vector_backend:
                try:
                    self.vector_backend.store_embedding(
                        query=query,
                        embedding=embedding,
                        crew_name=crew_name,
                        user_id=user_id,
                        language=language or "unknown",
                        result_hash=result_hash,
                        ttl_seconds=ttl_seconds,
                    )
                except Exception as e:
                    logger.warning(f"Failed to store embedding in vector backend: {e}")
                    # Continue with regular caching even if vector storage fails
            
            # Wrap result with metadata if embedding available
            if embedding:
                cache_value = {
                    "result": result,
                    "_embedding": embedding,
                    "_query": query,
                    "_result_hash": result_hash,
                }
            else:
                # Store as-is if no embedding capability
                cache_value = result
            
            success = await self.backend.set(key, cache_value, ttl_seconds)
            if success:
                logger.debug(
                    f"Crew result cached for {crew_name} ({user_id}) "
                    f"with embedding={'yes' if embedding else 'no'} "
                    f"and vector_db={'yes' if self.vector_backend else 'no'}"
                )
            else:
                track_cache_error("crew", "set", self.fork_name)
            return success
        except Exception as e:
            logger.warning(f"Crew cache set error: {e}")
            self.stats["errors"] += 1
            track_cache_error("crew", "set", self.fork_name)
            return False
    
    async def set_conversation_summary(
        self,
        user_id: str,
        summary: str,
        ttl_seconds: int = 604800  # 7 days
    ) -> bool:
        """Cache conversation summary."""
        key = self._make_key("summary", user_id)
        try:
            return await self.backend.set(key, summary, ttl_seconds)
        except Exception as e:
            logger.warning(f"Summary cache set error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def clear_user_cache(self, user_id: str) -> bool:
        """Clear all cached data for a user."""
        # Note: In a real implementation, we'd need to track user-specific keys
        logger.info(f"Clearing cache for user {user_id}")
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        # Update hit rate metrics
        update_cache_hit_rate("crew", hit_rate, self.fork_name)
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_percent": hit_rate
        }
    
    def cleanup_expired_entries(self, current_time: Optional[float] = None) -> int:
        """Clean up expired cache entries from vector store.
        
        This should be called periodically (e.g., hourly) to remove stale embeddings.
        
        Args:
            current_time: Current timestamp (defaults to time.time())
        
        Returns:
            Number of expired entries deleted
        """
        if self.vector_backend:
            try:
                deleted = self.vector_backend.cleanup_expired(current_time)
                logger.info(f"Cleaned up {deleted} expired cache entries for fork {self.fork_name}")
                return deleted
            except Exception as e:
                logger.error(f"Failed to cleanup expired entries: {e}")
                return 0
        return 0
