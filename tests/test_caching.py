"""Tests for caching infrastructure."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from universal_agentic_framework.caching import (
    CacheManager,
    RedisCacheBackend,
    MemoryCacheBackend,
)


class TestMemoryCacheBackend:
    """Test in-memory cache backend."""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set/get operations."""
        backend = MemoryCacheBackend()
        
        await backend.set("key1", "value1")
        result = await backend.get("key1")
        
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        backend = MemoryCacheBackend()
        
        await backend.set("expiring", "data", ttl_seconds=1)
        
        # Should exist immediately
        assert await backend.get("expiring") == "data"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await backend.get("expiring") is None
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        backend = MemoryCacheBackend()
        
        await backend.set("key", "value")
        assert await backend.delete("key") is True
        assert await backend.get("key") is None
        assert await backend.delete("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test cache clearing."""
        backend = MemoryCacheBackend()
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        
        assert await backend.clear() is True
        assert await backend.get("key1") is None
        assert await backend.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test eviction when max size exceeded."""
        backend = MemoryCacheBackend(max_size=3)
        
        await backend.set("key1", "value1", ttl_seconds=3600)
        await backend.set("key2", "value2", ttl_seconds=3600)
        await backend.set("key3", "value3", ttl_seconds=3600)
        
        # Cache is now full with 3 items
        assert len(backend.cache) == 3
        
        # Adding 4th should trigger eviction
        await backend.set("key4", "value4", ttl_seconds=3600)
        
        # Cache should still be max_size (3) after eviction
        assert len(backend.cache) == 3


class TestCacheManager:
    """Test cache manager."""
    
    @pytest.mark.asyncio
    async def test_key_generation(self):
        """Test cache key generation."""
        manager = CacheManager()
        
        key1 = manager._make_key("llm", "gpt-4", "hello", 2048)
        key2 = manager._make_key("llm", "gpt-4", "hello", 2048)
        key3 = manager._make_key("llm", "gpt-4", "world", 2048)
        
        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different key
        assert len(key1) == 32  # MD5 hash length
    
    @pytest.mark.asyncio
    async def test_llm_response_caching(self):
        """Test LLM response caching."""
        manager = CacheManager()
        
        # Cache a response
        success = await manager.set_llm_response(
            "gpt-4",
            "What is AI?",
            "AI is artificial intelligence.",
            max_tokens=2048
        )
        assert success is True
        
        # Retrieve from cache
        result = await manager.get_llm_response("gpt-4", "What is AI?", 2048)
        assert result == "AI is artificial intelligence."
        
        # Check stats
        stats = manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
    
    @pytest.mark.asyncio
    async def test_memory_query_caching(self):
        """Test memory query result caching."""
        backend = MemoryCacheBackend()  # Use memory backend for consistent test behavior
        manager = CacheManager(backend)
        
        results = [
            {"id": "1", "content": "First result"},
            {"id": "2", "content": "Second result"}
        ]
        
        # Cache results using backend directly
        key = manager._make_key("memory", "user123", "search term", 2)
        success = await backend.set(key, results, ttl_seconds=3600)
        assert success is True
        
        # Retrieve from cache
        cached = await backend.get(key)
        assert cached == results
    
    @pytest.mark.asyncio
    async def test_conversation_summary_caching(self):
        """Test summary caching."""
        manager = CacheManager()
        
        summary = "User asked about Python, answered with resources."
        
        # Cache summary
        success = await manager.set_conversation_summary("user123", summary)
        assert success is True
        
        # Retrieve from cache
        cached = await manager.get_conversation_summary("user123")
        assert cached == summary
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        manager = CacheManager()
        
        # Generate some cache activity
        await manager.set_llm_response("gpt-4", "q1", "a1")
        await manager.get_llm_response("gpt-4", "q1", 2048)  # Hit
        await manager.get_llm_response("gpt-4", "q2", 2048)  # Miss
        await manager.get_llm_response("gpt-4", "q1", 2048)  # Hit
        
        stats = manager.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate_percent"] == pytest.approx(66.67, rel=0.1)


class TestRedisCacheBackend:
    """Test Redis cache backend (requires Redis running)."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="Requires Redis running")
    async def test_redis_connection(self):
        """Test Redis connection."""
        backend = RedisCacheBackend("redis://localhost:6379/0")
        
        # Should initialize without error
        await backend._ensure_connected()
        assert backend._initialized is True
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="Requires Redis running")
    async def test_redis_operations(self):
        """Test Redis set/get/delete."""
        backend = RedisCacheBackend()
        
        # Set and get
        success = await backend.set("test_key", {"data": "value"})
        assert success is True
        
        result = await backend.get("test_key")
        assert result == {"data": "value"}
        
        # Delete
        deleted = await backend.delete("test_key")
        assert deleted is True


class TestCacheDecorator:
    """Test caching decorator."""
    
    @pytest.mark.asyncio
    async def test_cache_llm_response_decorator(self):
        """Test @cache_llm_response decorator."""
        from universal_agentic_framework.caching.decorators import (
            cache_llm_response,
            initialize_cache
        )
        
        # Initialize with memory backend for testing
        initialize_cache(use_redis=False)
        
        call_count = 0
        
        @cache_llm_response(ttl_seconds=3600)
        async def mock_llm(model: str, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"
        
        # First call - should execute function
        result1 = await mock_llm(model="gpt-4", prompt="Hello")
        assert result1 == "Response to: Hello"
        assert call_count == 1
        
        # Second call - should hit cache
        result2 = await mock_llm(model="gpt-4", prompt="Hello")
        assert result2 == "Response to: Hello"
        assert call_count == 1  # Function not called again
