"""Caching layer for LLM responses and memory queries.

Implements Redis-backed caching for:
- LLM responses (question-answer pairs)
- Memory query results (semantic search results)
- Conversation summaries
"""

import json
import hashlib
import logging
from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod
import time

from universal_agentic_framework.monitoring.metrics import update_cache_hit_rate

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
        import threading

        self.redis = None  # client of the most recently used loop (introspection only)
        self.redis_url = redis_url
        # One client per event loop. redis.asyncio clients are bound to the loop they
        # were created on, and callers may run cache calls on short-lived loops in
        # worker threads — swapping a single shared client back and forth broke the
        # main loop's in-flight calls and leaked the replaced connection pools.
        self._clients: Dict[Any, Any] = {}
        self._clients_lock = threading.Lock()

    async def _client(self):
        """Return the Redis client for the running event loop (created on first use)."""
        import asyncio
        current_loop = asyncio.get_running_loop()
        with self._clients_lock:
            # Forget clients whose loop has finished (a worker thread's asyncio.run).
            for loop in [lp for lp in self._clients if lp.is_closed()]:
                self._clients.pop(loop, None)
            client = self._clients.get(current_loop)
        if client is not None:
            return client
        try:
            import redis.asyncio as redis
        except ImportError:  # pragma: no cover - redis<4.2
            import redis
        try:
            client = await redis.from_url(self.redis_url, decode_responses=True)
            await client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        with self._clients_lock:
            self._clients[current_loop] = client
        self.redis = client
        logger.info(f"Connected to Redis at {self.redis_url}")
        return client

    async def _ensure_connected(self):
        """Back-compat shim: ensure a client exists for the running loop."""
        await self._client()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            client = await self._client()
            value = await client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in Redis with TTL."""
        try:
            client = await self._client()
            serialized = json.dumps(value)
            await client.setex(key, ttl_seconds, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from Redis."""
        try:
            client = await self._client()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Redis database."""
        try:
            client = await self._client()
            await client.flushdb()
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
        profile_name: str = "default",
        enable_compression: bool = False,
        compression_threshold_kb: int = 1,
    ):
        """Initialize cache manager.
        
        Args:
            backend: Cache backend (defaults to MemoryCacheBackend)
            profile_name: Name of the profile for metrics tracking
            enable_compression: Whether to enable cache compression (default: False)
            compression_threshold_kb: Minimum size to compress in KB (default: 1KB)
        """
        self.backend = backend or MemoryCacheBackend()
        self.profile_name = profile_name
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

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

    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        content = f"{prefix}:" + ":".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_llm_response(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 2048
    ) -> Optional[str]:
        """Get cached LLM response."""
        key = self._make_key("llm", model, prompt, max_tokens)
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
        key = self._make_key("llm", model, prompt, max_tokens)
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
        key = self._make_key("memory", user_id, query, top_k)
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
        key = self._make_key("memory", user_id, query, top_k)
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
        update_cache_hit_rate("all", hit_rate, self.profile_name)
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_percent": hit_rate
        }
