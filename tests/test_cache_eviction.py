"""Tests for cache eviction policies."""

import pytest
import asyncio
import time
from universal_agentic_framework.caching.eviction import (
    LRUPolicy,
    LFUPolicy,
    FIFOPolicy,
    TTLPolicy,
    RandomPolicy,
    CacheEntry,
    create_eviction_policy
)
from universal_agentic_framework.caching.manager import MemoryCacheBackend


@pytest.fixture
def sample_cache():
    """Create sample cache with entries."""
    current_time = time.time()
    return {
        "key1": CacheEntry("key1", "val1", None, current_time - 10, current_time - 5, 10),
        "key2": CacheEntry("key2", "val2", None, current_time - 8, current_time - 2, 5),
        "key3": CacheEntry("key3", "val3", None, current_time - 6, current_time - 1, 15),
        "key4": CacheEntry("key4", "val4", None, current_time - 4, current_time - 3, 2),
    }


def test_lru_policy():
    """Test LRU eviction policy."""
    policy = LRUPolicy()
    current_time = time.time()
    
    cache = {
        "key1": CacheEntry("key1", "val1", None, current_time, current_time - 10, 0),
        "key2": CacheEntry("key2", "val2", None, current_time, current_time - 5, 0),
        "key3": CacheEntry("key3", "val3", None, current_time, current_time - 1, 0),
    }
    
    # Select victims - should get least recently accessed
    victims = policy.select_victims(cache, 2)
    
    assert len(victims) == 2
    assert "key1" in victims  # Oldest access
    assert "key2" in victims  # Second oldest


def test_lru_policy_on_get():
    """Test LRU updates on access."""
    policy = LRUPolicy()
    current_time = time.time()
    
    entry = CacheEntry("key1", "val1", None, current_time, current_time - 10, 0)
    old_access_time = entry.last_accessed
    
    # Access entry
    time.sleep(0.01)  # Small delay to ensure time difference
    policy.on_get("key1", entry)
    
    assert entry.last_accessed > old_access_time


def test_lfu_policy():
    """Test LFU eviction policy."""
    policy = LFUPolicy()
    current_time = time.time()
    
    cache = {
        "key1": CacheEntry("key1", "val1", None, current_time, current_time, 10),
        "key2": CacheEntry("key2", "val2", None, current_time, current_time, 5),
        "key3": CacheEntry("key3", "val3", None, current_time, current_time, 15),
    }
    
    # Select victims - should get least frequently accessed
    victims = policy.select_victims(cache, 2)
    
    assert len(victims) == 2
    assert "key2" in victims  # Lowest access count (5)
    assert "key1" in victims  # Second lowest (10)


def test_lfu_policy_on_get():
    """Test LFU increments access count."""
    policy = LFUPolicy()
    current_time = time.time()
    
    entry = CacheEntry("key1", "val1", None, current_time, current_time, 5)
    
    # Access entry
    policy.on_get("key1", entry)
    
    assert entry.access_count == 6  # Incremented


def test_fifo_policy():
    """Test FIFO eviction policy."""
    policy = FIFOPolicy()
    current_time = time.time()
    
    cache = {
        "key1": CacheEntry("key1", "val1", None, current_time - 10, current_time, 0),
        "key2": CacheEntry("key2", "val2", None, current_time - 5, current_time, 0),
        "key3": CacheEntry("key3", "val3", None, current_time - 1, current_time, 0),
    }
    
    # Select victims - should get oldest inserted
    victims = policy.select_victims(cache, 2)
    
    assert len(victims) == 2
    assert "key1" in victims  # Oldest insertion
    assert "key2" in victims  # Second oldest


def test_ttl_policy_expired_first():
    """Test TTL policy evicts expired entries first."""
    policy = TTLPolicy()
    current_time = time.time()
    
    cache = {
        "key1": CacheEntry("key1", "val1", current_time - 10, current_time - 5, current_time - 5, 0),  # Expired
        "key2": CacheEntry("key2", "val2", current_time + 100, current_time - 3, current_time - 3, 0),  # Not expired
        "key3": CacheEntry("key3", "val3", current_time - 5, current_time - 8, current_time - 8, 0),  # Expired
    }
    
    # Select victims - should get expired entries first
    victims = policy.select_victims(cache, 2)
    
    assert len(victims) == 2
    assert "key1" in victims  # Expired
    assert "key3" in victims  # Expired


def test_ttl_policy_fallback_to_lru():
    """Test TTL policy falls back to LRU when no expired entries."""
    policy = TTLPolicy()
    current_time = time.time()
    
    cache = {
        "key1": CacheEntry("key1", "val1", None, current_time, current_time - 10, 0),
        "key2": CacheEntry("key2", "val2", None, current_time, current_time - 5, 0),
        "key3": CacheEntry("key3", "val3", None, current_time, current_time - 1, 0),
    }
    
    # Select victims - should fall back to LRU (no expired entries)
    victims = policy.select_victims(cache, 2)
    
    assert len(victims) == 2
    assert "key1" in victims  # LRU: oldest access
    assert "key2" in victims  # LRU: second oldest


def test_random_policy():
    """Test random eviction policy."""
    policy = RandomPolicy()
    current_time = time.time()
    
    cache = {
        f"key{i}": CacheEntry(f"key{i}", f"val{i}", None, current_time, current_time, 0)
        for i in range(10)
    }
    
    # Select victims - should return requested count
    victims = policy.select_victims(cache, 3)
    
    assert len(victims) == 3
    assert all(key in cache for key in victims)


def test_create_eviction_policy():
    """Test eviction policy factory."""
    # Test valid policies
    lru = create_eviction_policy("LRU")
    assert isinstance(lru, LRUPolicy)
    
    lfu = create_eviction_policy("LFU")
    assert isinstance(lfu, LFUPolicy)
    
    fifo = create_eviction_policy("FIFO")
    assert isinstance(fifo, FIFOPolicy)
    
    ttl = create_eviction_policy("TTL")
    assert isinstance(ttl, TTLPolicy)
    
    # Test case insensitive
    lru_lower = create_eviction_policy("lru")
    assert isinstance(lru_lower, LRUPolicy)


def test_create_eviction_policy_invalid():
    """Test factory with invalid policy name."""
    with pytest.raises(ValueError, match="Unknown eviction policy"):
        create_eviction_policy("INVALID")


@pytest.mark.asyncio
async def test_memory_backend_with_lru():
    """Test memory backend with LRU eviction."""
    backend = MemoryCacheBackend(max_size=3, eviction_policy="LRU")
    
    # Fill cache
    await backend.set("key1", "val1", ttl_seconds=3600)
    await backend.set("key2", "val2", ttl_seconds=3600)
    await backend.set("key3", "val3", ttl_seconds=3600)
    
    # Access key1 and key2
    await backend.get("key1")
    await backend.get("key2")
    
    # Add key4 - should evict key3 (least recently used)
    await backend.set("key4", "val4", ttl_seconds=3600)
    
    assert await backend.get("key1") is not None
    assert await backend.get("key2") is not None
    assert await backend.get("key3") is None  # Evicted
    assert await backend.get("key4") is not None


@pytest.mark.asyncio
async def test_memory_backend_with_lru_when_time_frozen(monkeypatch):
    """LRU should still evict untouched entries under frozen/coarse clocks."""
    frozen_time = 1000.0
    monkeypatch.setattr(
        "universal_agentic_framework.caching.manager.time.time",
        lambda: frozen_time,
    )
    monkeypatch.setattr(
        "universal_agentic_framework.caching.eviction.time.time",
        lambda: frozen_time,
    )

    backend = MemoryCacheBackend(max_size=3, eviction_policy="LRU")

    await backend.set("key1", "val1", ttl_seconds=3600)
    await backend.set("key2", "val2", ttl_seconds=3600)
    await backend.set("key3", "val3", ttl_seconds=3600)

    await backend.get("key1")
    await backend.get("key2")

    await backend.set("key4", "val4", ttl_seconds=3600)

    assert await backend.get("key1") is not None
    assert await backend.get("key2") is not None
    assert await backend.get("key3") is None
    assert await backend.get("key4") is not None


@pytest.mark.asyncio
async def test_memory_backend_with_lfu():
    """Test memory backend with LFU eviction."""
    backend = MemoryCacheBackend(max_size=3, eviction_policy="LFU")
    
    # Fill cache
    await backend.set("key1", "val1", ttl_seconds=3600)
    await backend.set("key2", "val2", ttl_seconds=3600)
    await backend.set("key3", "val3", ttl_seconds=3600)
    
    # Access key1 multiple times
    for _ in range(5):
        await backend.get("key1")
    
    # Access key2 twice
    await backend.get("key2")
    await backend.get("key2")
    
    # key3 has 0 accesses (least frequent)
    # Add key4 - should evict key3
    await backend.set("key4", "val4", ttl_seconds=3600)
    
    assert await backend.get("key1") is not None
    assert await backend.get("key2") is not None
    assert await backend.get("key3") is None  # Evicted (LFU)
    assert await backend.get("key4") is not None


@pytest.mark.asyncio
async def test_memory_backend_with_fifo():
    """Test memory backend with FIFO eviction."""
    backend = MemoryCacheBackend(max_size=3, eviction_policy="FIFO")
    
    # Fill cache in order
    await backend.set("key1", "val1", ttl_seconds=3600)
    await asyncio.sleep(0.01)  # Small delay to ensure order
    await backend.set("key2", "val2", ttl_seconds=3600)
    await asyncio.sleep(0.01)
    await backend.set("key3", "val3", ttl_seconds=3600)
    
    # Access key1 many times (shouldn't matter for FIFO)
    for _ in range(10):
        await backend.get("key1")
    
    # Add key4 - should evict key1 (first in)
    await backend.set("key4", "val4", ttl_seconds=3600)
    
    assert await backend.get("key1") is None  # Evicted (FIFO)
    assert await backend.get("key2") is not None
    assert await backend.get("key3") is not None
    assert await backend.get("key4") is not None


@pytest.mark.asyncio
async def test_memory_backend_with_ttl():
    """Test memory backend with TTL eviction."""
    backend = MemoryCacheBackend(max_size=3, eviction_policy="TTL")
    
    current_time = time.time()
    
    # Add entries with different TTLs
    await backend.set("key1", "val1", ttl_seconds=1)  # Expires soon
    await asyncio.sleep(0.01)
    await backend.set("key2", "val2", ttl_seconds=3600)
    await asyncio.sleep(0.01)
    await backend.set("key3", "val3", ttl_seconds=3600)
    
    # Wait for key1 to expire
    await asyncio.sleep(1.5)
    
    # Add key4 - should evict expired key1
    await backend.set("key4", "val4", ttl_seconds=3600)
    
    assert await backend.get("key1") is None  # Expired/evicted
    assert await backend.get("key2") is not None
    assert await backend.get("key3") is not None
    assert await backend.get("key4") is not None


@pytest.mark.asyncio
async def test_memory_backend_eviction_stats():
    """Test eviction statistics tracking."""
    backend = MemoryCacheBackend(max_size=2, eviction_policy="LRU")
    
    # Fill cache
    await backend.set("key1", "val1", ttl_seconds=3600)
    await backend.set("key2", "val2", ttl_seconds=3600)
    
    # Trigger eviction
    await backend.set("key3", "val3", ttl_seconds=3600)
    
    stats = backend.get_eviction_stats()
    
    assert stats["evictions"] == 1
    assert stats["policy"] == "LRU"
    assert stats["current_size"] == 2
    assert stats["max_size"] == 2
    assert stats["utilization"] == 1.0


@pytest.mark.asyncio
async def test_memory_backend_multiple_evictions():
    """Test multiple evictions in series."""
    backend = MemoryCacheBackend(max_size=3, eviction_policy="LRU")
    
    # Add 6 items (should trigger 3 evictions)
    for i in range(6):
        await backend.set(f"key{i}", f"val{i}", ttl_seconds=3600)
    
    stats = backend.get_eviction_stats()
    
    assert stats["evictions"] == 3
    assert stats["current_size"] == 3


@pytest.mark.asyncio
async def test_memory_backend_update_existing():
    """Test updating existing key doesn't trigger eviction."""
    backend = MemoryCacheBackend(max_size=2, eviction_policy="LRU")
    
    await backend.set("key1", "val1", ttl_seconds=3600)
    await backend.set("key2", "val2", ttl_seconds=3600)
    
    # Update key1 (shouldn't evict)
    await backend.set("key1", "val1_updated", ttl_seconds=3600)
    
    stats = backend.get_eviction_stats()
    
    assert stats["evictions"] == 0
    assert stats["current_size"] == 2


@pytest.mark.asyncio
async def test_memory_backend_cleanup_doesnt_affect_eviction_stats():
    """Test cleanup vs eviction are tracked separately."""
    backend = MemoryCacheBackend(max_size=5, eviction_policy="LRU")
    
    # Add entries with short TTL
    await backend.set("key1", "val1", ttl_seconds=1)
    await backend.set("key2", "val2", ttl_seconds=3600)
    
    # Wait for expiry
    await asyncio.sleep(1.5)
    
    # Cleanup
    removed = await backend.cleanup()
    
    stats = backend.get_eviction_stats()
    
    assert removed == 1  # Cleanup removed 1
    assert stats["evictions"] == 0  # But eviction count is 0


def test_policy_names():
    """Test policy name methods."""
    assert LRUPolicy().get_name() == "LRU"
    assert LFUPolicy().get_name() == "LFU"
    assert FIFOPolicy().get_name() == "FIFO"
    assert TTLPolicy().get_name() == "TTL"
    assert RandomPolicy().get_name() == "Random"


def test_select_victims_edge_cases():
    """Test selecting victims with edge cases."""
    policy = LRUPolicy()
    
    # Empty cache
    victims = policy.select_victims({}, 5)
    assert len(victims) == 0
    
    # Request more than available
    cache = {
        "key1": CacheEntry("key1", "val1", None, time.time(), time.time(), 0),
        "key2": CacheEntry("key2", "val2", None, time.time(), time.time(), 0),
    }
    victims = policy.select_victims(cache, 10)
    assert len(victims) == 2  # Only 2 available
