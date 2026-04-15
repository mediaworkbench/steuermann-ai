"""Tests for cache scheduler and cleanup functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from universal_agentic_framework.caching.manager import CacheManager, MemoryCacheBackend
from universal_agentic_framework.caching.scheduler import CacheScheduler, create_scheduler
from universal_agentic_framework.caching.eviction import CacheEntry


@pytest.fixture
def memory_backend():
    """Create memory cache backend."""
    return MemoryCacheBackend(max_size=100)


@pytest.fixture
def cache_manager(memory_backend):
    """Create cache manager with memory backend."""
    return CacheManager(backend=memory_backend, use_vector_db=False)


@pytest.fixture
def scheduler_config():
    """Create scheduler configuration."""
    return {
        "cleanup_interval_minutes": 1,
        "stats_interval_minutes": 1,
        "enable_cleanup": True,
        "enable_stats": True,
        "max_cache_age_hours": 24,
    }


@pytest.mark.asyncio
async def test_scheduler_initialization(cache_manager, scheduler_config):
    """Test scheduler initializes correctly."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    assert scheduler.cache_manager == cache_manager
    assert scheduler.cleanup_interval == 1
    assert scheduler.stats_interval == 1
    assert scheduler.enable_cleanup is True
    assert scheduler.enable_stats is True
    assert not scheduler.running


@pytest.mark.asyncio
async def test_scheduler_start_stop(cache_manager, scheduler_config):
    """Test scheduler can start and stop."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Start scheduler
    scheduler.start()
    assert scheduler.running
    
    # Get jobs
    jobs = scheduler.get_jobs()
    assert len(jobs) >= 2  # cleanup + stats jobs
    
    # Stop scheduler
    scheduler.stop()
    assert not scheduler.running


@pytest.mark.asyncio
async def test_cleanup_expired_entries(cache_manager):
    """Test cleanup removes expired entries."""
    backend = cache_manager.backend
    
    # Add some entries with different expiry times
    import time
    current_time = time.time()
    
    # Add expired entries by creating CacheEntry objects with past expiry times
    backend.cache["expired1"] = CacheEntry(
        key="expired1",
        value="value1",
        expiry=current_time - 10,  # Expired 10 seconds ago
        inserted_at=current_time - 20,
        last_accessed=current_time - 20,
        access_count=0
    )
    backend.cache["expired2"] = CacheEntry(
        key="expired2",
        value="value2",
        expiry=current_time - 5,  # Expired 5 seconds ago
        inserted_at=current_time - 15,
        last_accessed=current_time - 15,
        access_count=0
    )
    
    # Valid entry
    await backend.set("valid1", "value3", ttl_seconds=3600)
    
    # Cleanup
    removed = await backend.cleanup()
    
    # Check expired entries are gone
    assert await backend.get("expired1") is None
    assert await backend.get("expired2") is None
    assert await backend.get("valid1") == "value3"
    assert removed == 2


@pytest.mark.asyncio
async def test_cleanup_task_execution(cache_manager, scheduler_config):
    """Test cleanup task executes correctly."""
    backend = cache_manager.backend
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Add expired entries by creating CacheEntry objects with past expiry times
    import time
    current_time = time.time()
    backend.cache["expired1"] = CacheEntry(
        key="expired1",
        value="value1",
        expiry=current_time - 10,
        inserted_at=current_time - 20,
        last_accessed=current_time - 20,
        access_count=0
    )
    backend.cache["expired2"] = CacheEntry(
        key="expired2",
        value="value2",
        expiry=current_time - 5,
        inserted_at=current_time - 15,
        last_accessed=current_time - 15,
        access_count=0
    )
    
    # Run cleanup task manually
    await scheduler._cleanup_task()
    
    # Verify cleanup occurred
    assert await backend.get("expired1") is None
    assert await backend.get("expired2") is None


@pytest.mark.asyncio
async def test_stats_collection_task(cache_manager, scheduler_config):
    """Test stats collection task."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Add some cache operations
    await cache_manager.backend.set("key1", "value1", ttl_seconds=3600)
    await cache_manager.backend.get("key1")
    
    # Manually trigger stats collection
    await scheduler._stats_collection_task()
    
    # Should not raise exceptions
    assert True


@pytest.mark.asyncio
async def test_scheduler_with_cron_expression(cache_manager):
    """Test scheduler with cron expression."""
    config = {
        "cleanup_cron": "0 * * * *",  # Every hour
        "enable_cleanup": True,
        "enable_stats": False,
    }
    scheduler = CacheScheduler(cache_manager, config)
    
    scheduler.start()
    jobs = scheduler.get_jobs()
    
    # Should have cleanup job with cron trigger
    assert len(jobs) >= 1
    assert any("Cron" in job["name"] for job in jobs)
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_create_scheduler_helper(cache_manager, scheduler_config):
    """Test create_scheduler helper function."""
    scheduler = await create_scheduler(cache_manager, scheduler_config, start=False)
    
    assert isinstance(scheduler, CacheScheduler)
    assert not scheduler.running
    
    # Start it
    scheduler.start()
    assert scheduler.running
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_custom_job(cache_manager, scheduler_config):
    """Test adding custom scheduled jobs."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    scheduler.start()
    
    # Define custom job
    custom_called = []
    
    async def custom_job():
        custom_called.append(True)
    
    # Add custom job
    scheduler.add_custom_job(custom_job, trigger="interval", seconds=1)
    
    # Check job was added
    jobs = scheduler.get_jobs()
    assert any("custom_job" in job["name"] for job in jobs)
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_cleanup_task_error_handling(cache_manager, scheduler_config):
    """Test cleanup task handles errors gracefully."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Mock backend.cleanup to raise exception
    with patch.object(
        cache_manager.backend,
        "cleanup",
        side_effect=Exception("Test error")
    ):
        # Should not raise exception
        await scheduler._cleanup_task()


@pytest.mark.asyncio
async def test_memory_backend_cleanup_method():
    """Test MemoryCacheBackend cleanup method."""
    backend = MemoryCacheBackend()
    import time
    current_time = time.time()
    
    # Add entries - set expired ones with CacheEntry objects
    backend.cache["key1"] = CacheEntry(
        key="key1",
        value="value1",
        expiry=current_time - 10,
        inserted_at=current_time - 20,
        last_accessed=current_time - 20,
        access_count=0
    )
    await backend.set("key2", "value2", ttl_seconds=3600)
    backend.cache["key3"] = CacheEntry(
        key="key3",
        value="value3",
        expiry=current_time - 5,
        inserted_at=current_time - 15,
        last_accessed=current_time - 15,
        access_count=0
    )
    
    # Cleanup
    removed = await backend.cleanup()
    
    assert removed == 2
    assert await backend.get("key2") == "value2"


@pytest.mark.asyncio
async def test_scheduler_disabled_jobs(cache_manager):
    """Test scheduler with disabled jobs."""
    config = {
        "enable_cleanup": False,
        "enable_stats": False,
    }
    scheduler = CacheScheduler(cache_manager, config)
    
    scheduler.start()
    jobs = scheduler.get_jobs()
    
    # Should have no jobs
    assert len(jobs) == 0
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_get_jobs(cache_manager, scheduler_config):
    """Test get_jobs returns correct job information."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    scheduler.start()
    
    jobs = scheduler.get_jobs()
    
    # Should have at least cleanup and stats jobs
    assert len(jobs) >= 2
    
    # Check job structure
    for job in jobs:
        assert "id" in job
        assert "name" in job
        assert "next_run_time" in job
        assert "trigger" in job
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_cleanup_empty_cache(cache_manager, scheduler_config):
    """Test cleanup on empty cache."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Run cleanup on empty cache
    await scheduler._cleanup_task()
    
    # Should complete without errors
    assert True


@pytest.mark.asyncio
async def test_scheduler_double_start(cache_manager, scheduler_config):
    """Test starting scheduler twice logs warning."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    scheduler.start()
    assert scheduler.running
    
    # Try to start again (should log warning but not crash)
    scheduler.start()
    assert scheduler.running
    
    scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_stop_not_running(cache_manager, scheduler_config):
    """Test stopping scheduler that's not running."""
    scheduler = CacheScheduler(cache_manager, scheduler_config)
    
    # Stop without starting (should log warning but not crash)
    scheduler.stop()
    assert not scheduler.running
