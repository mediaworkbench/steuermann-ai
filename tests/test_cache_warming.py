"""Tests for cache warming strategies."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from universal_agentic_framework.caching.manager import CacheManager, MemoryCacheBackend
from universal_agentic_framework.caching.warming import CacheWarmer, create_cache_warmer


@pytest.fixture
def memory_backend():
    """Create memory cache backend."""
    return MemoryCacheBackend(max_size=100)


@pytest.fixture
def cache_manager(memory_backend):
    """Create cache manager with memory backend."""
    return CacheManager(backend=memory_backend, use_vector_db=False)


@pytest.fixture
def warming_config():
    """Create warming configuration."""
    return {
        "warming_queries": [
            {
                "crew_name": "research",
                "user_id": "system",
                "query": "What is AI?",
                "result": {"output": "AI is artificial intelligence"},
                "language": "en",
                "ttl_seconds": 3600
            },
            {
                "crew_name": "analytics",
                "user_id": "system",
                "query": "Show latest metrics",
                "result": {"metrics": {"cpu": 50, "memory": 75}},
                "language": "en",
                "ttl_seconds": 3600
            }
        ],
        "warm_on_startup": True,
        "warming_interval_hours": 24
    }


@pytest.mark.asyncio
async def test_cache_warmer_initialization(cache_manager, warming_config):
    """Test cache warmer initializes correctly."""
    warmer = CacheWarmer(cache_manager, warming_config)
    
    assert warmer.cache_manager == cache_manager
    assert len(warmer.warming_queries) == 2
    assert warmer.warm_on_startup is True
    assert warmer.warming_interval_hours == 24


@pytest.mark.asyncio
async def test_warm_crew_cache(cache_manager):
    """Test warming cache for specific crew query."""
    warmer = CacheWarmer(cache_manager)
    
    # Result generator
    call_count = []
    async def generate_result():
        call_count.append(1)
        return {"output": "Test result"}
    
    # Warm cache
    success = await warmer.warm_crew_cache(
        crew_name="research",
        user_id="user123",
        query="Test query",
        result_generator=generate_result,
        language="en"
    )
    
    assert success is True
    assert len(call_count) == 1  # Generator was called


@pytest.mark.asyncio
async def test_warm_crew_cache_skip_existing(cache_manager):
    """Test warming skips already cached queries."""
    warmer = CacheWarmer(cache_manager)
    
    # Pre-populate cache
    await cache_manager.set_crew_result(
        "research", "user123", "Test query",
        {"output": "Existing result"}, "en"
    )
    
    # Result generator (should not be called)
    call_count = []
    async def generate_result():
        call_count.append(1)
        return {"output": "New result"}
    
    # Warm cache (should skip)
    success = await warmer.warm_crew_cache(
        crew_name="research",
        user_id="user123",
        query="Test query",
        result_generator=generate_result,
        language="en"
    )
    
    assert success is True
    assert len(call_count) == 0  # Generator not called
    
    # Verify cache unchanged
    cached = await cache_manager.get_crew_result(
        "research", "user123", "Test query", "en"
    )
    assert cached["output"] == "Existing result"


@pytest.mark.asyncio
async def test_warm_from_config(cache_manager, warming_config):
    """Test warming cache from configuration."""
    warmer = CacheWarmer(cache_manager, warming_config)
    
    stats = await warmer.warm_from_config()
    
    assert stats["total"] == 2
    assert stats["success"] == 2
    assert stats["failed"] == 0
    assert stats["skipped"] == 0
    assert stats["duration_seconds"] > 0
    
    # Verify cache manager stats show activity
    manager_stats = cache_manager.get_stats()
    assert manager_stats["total_requests"] > 0


@pytest.mark.asyncio
async def test_warm_from_config_skip_existing(cache_manager, warming_config):
    """Test warming skips already cached queries."""
    warmer = CacheWarmer(cache_manager, warming_config)
    
    # Pre-populate one query
    await cache_manager.set_crew_result(
        "research", "system", "What is AI?",
        {"output": "Existing answer"}, "en"
    )
    
    stats = await warmer.warm_from_config()
    
    assert stats["total"] == 2
    assert stats["success"] == 1  # Only the new one
    assert stats["skipped"] == 1  # The existing one


@pytest.mark.asyncio
async def test_warm_from_config_invalid_queries(cache_manager):
    """Test warming handles invalid query configurations."""
    config = {
        "warming_queries": [
            {"crew_name": "research"},  # Missing query and result
            {"query": "Test"},  # Missing crew_name
            {
                "crew_name": "valid",
                "query": "Valid query",
                "result": {"output": "Valid result"}
            }
        ]
    }
    
    warmer = CacheWarmer(cache_manager, config)
    stats = await warmer.warm_from_config()
    
    assert stats["total"] == 3
    assert stats["success"] == 1
    assert stats["skipped"] == 2


@pytest.mark.asyncio
async def test_warm_top_queries(cache_manager):
    """Test warming with top queries."""
    warmer = CacheWarmer(cache_manager)
    
    # Mock query generator
    async def top_query_generator(n):
        return [
            ("research", "user1", "Query 1", {"output": "Result 1"}, "en"),
            ("analytics", "user2", "Query 2", {"output": "Result 2"}, "en"),
        ]
    
    stats = await warmer.warm_top_queries(
        top_n=10,
        query_generator=top_query_generator
    )
    
    assert stats["total"] == 2
    assert stats["success"] == 2
    assert stats["failed"] == 0
    
    # Cache manager stats should show activity
    manager_stats = cache_manager.get_stats()
    assert manager_stats["total_requests"] > 0


@pytest.mark.asyncio
async def test_warm_top_queries_no_generator(cache_manager):
    """Test warming with no query generator."""
    warmer = CacheWarmer(cache_manager)
    
    stats = await warmer.warm_top_queries(top_n=10)
    
    assert stats["total"] == 0
    assert stats["success"] == 0


@pytest.mark.asyncio
async def test_warm_scheduled(cache_manager, warming_config):
    """Test scheduled warming."""
    warmer = CacheWarmer(cache_manager, warming_config)
    
    stats = await warmer.warm_scheduled()
    
    assert stats["total"] == 2
    assert stats["success"] == 2


@pytest.mark.asyncio
async def test_warm_on_demand(cache_manager):
    """Test on-demand cache warming."""
    warmer = CacheWarmer(cache_manager)
    
    queries = [
        {
            "query": "On-demand query 1",
            "result": {"output": "Result 1"},
            "user_id": "user123",
            "language": "en"
        },
        {
            "query": "On-demand query 2",
            "result": {"output": "Result 2"},
            "language": "de"
        }
    ]
    
    stats = await warmer.warm_on_demand("research", queries)
    
    assert stats["total"] == 2
    assert stats["success"] == 2
    assert stats["failed"] == 0


@pytest.mark.asyncio
async def test_warm_on_demand_invalid_queries(cache_manager):
    """Test on-demand warming with invalid queries."""
    warmer = CacheWarmer(cache_manager)
    
    queries = [
        {"query": "Valid", "result": {"output": "Result"}},
        {"query": "No result"},  # Missing result
        {"result": {"output": "Result"}},  # Missing query
    ]
    
    stats = await warmer.warm_on_demand("research", queries)
    
    assert stats["total"] == 3
    assert stats["success"] == 1


@pytest.mark.asyncio
async def test_create_cache_warmer(cache_manager, warming_config):
    """Test create_cache_warmer helper."""
    warmer = await create_cache_warmer(
        cache_manager,
        warming_config,
        warm_on_create=False
    )
    
    assert isinstance(warmer, CacheWarmer)


@pytest.mark.asyncio
async def test_create_cache_warmer_with_warming(cache_manager, warming_config):
    """Test create_cache_warmer with immediate warming."""
    warmer = await create_cache_warmer(
        cache_manager,
        warming_config,
        warm_on_create=True
    )
    
    assert isinstance(warmer, CacheWarmer)


@pytest.mark.asyncio
async def test_warm_crew_cache_error_handling(cache_manager):
    """Test error handling in cache warming."""
    warmer = CacheWarmer(cache_manager)
    
    # Result generator that raises error
    async def failing_generator():
        raise Exception("Test error")
    
    success = await warmer.warm_crew_cache(
        crew_name="research",
        user_id="user123",
        query="Test query",
        result_generator=failing_generator
    )
    
    assert success is False


@pytest.mark.asyncio
async def test_warm_from_config_empty(cache_manager):
    """Test warming with empty configuration."""
    config = {"warming_queries": []}
    warmer = CacheWarmer(cache_manager, config)
    
    stats = await warmer.warm_from_config()
    
    assert stats["total"] == 0
    assert stats["success"] == 0


@pytest.mark.asyncio
async def test_warm_top_queries_error_in_generator(cache_manager):
    """Test warming handles errors from query generator."""
    warmer = CacheWarmer(cache_manager)
    
    async def failing_generator(n):
        raise Exception("Generator failed")
    
    stats = await warmer.warm_top_queries(
        top_n=10,
        query_generator=failing_generator
    )
    
    assert stats["total"] == 0
    assert stats["failed"] == 0
