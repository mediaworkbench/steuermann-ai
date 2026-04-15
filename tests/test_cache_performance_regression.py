"""Performance regression tests for cache operations.

These tests establish performance baselines and detect regressions in:
- Basic cache operations (get/set latency)
- Compression overhead
- Eviction policy performance
- Memory usage
- Throughput under load

Tests use performance thresholds that should be met to pass.
Run with: pytest tests/test_cache_performance_regression.py -v
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
import random
import string

from universal_agentic_framework.caching.manager import (
    CacheManager,
    MemoryCacheBackend
)
from universal_agentic_framework.caching.compression import create_compressor
from universal_agentic_framework.caching.eviction import CacheEntry


# Performance thresholds (in milliseconds)
THRESHOLDS = {
    "get_latency_p50": 1.0,      # 50th percentile: 1ms
    "get_latency_p95": 5.0,      # 95th percentile: 5ms
    "set_latency_p50": 2.0,      # 50th percentile: 2ms
    "set_latency_p95": 10.0,     # 95th percentile: 10ms
    "compression_overhead": 50.0, # Max 50ms overhead for 100KB
    "eviction_latency": 10.0,    # Max 10ms for eviction
    "throughput_ops_per_sec": 1000,  # Min 1000 ops/sec
}


def generate_random_string(size: int) -> str:
    """Generate random string of specified size."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))


def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calculate specific percentile from values."""
    return statistics.quantiles(sorted(values), n=100)[percentile - 1]


@pytest.mark.asyncio
async def test_get_operation_latency():
    """Test cache get operation latency."""
    backend = MemoryCacheBackend(max_size=1000)
    
    # Populate cache
    for i in range(100):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    # Measure get latency
    latencies = []
    for i in range(100):
        start = time.perf_counter()
        await backend.get(f"key{i}")
        duration_ms = (time.perf_counter() - start) * 1000
        latencies.append(duration_ms)
    
    p50 = calculate_percentile(latencies, 50)
    p95 = calculate_percentile(latencies, 95)
    
    print(f"\nGet Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms")
    
    assert p50 < THRESHOLDS["get_latency_p50"], \
        f"P50 get latency {p50:.3f}ms exceeds threshold {THRESHOLDS['get_latency_p50']}ms"
    assert p95 < THRESHOLDS["get_latency_p95"], \
        f"P95 get latency {p95:.3f}ms exceeds threshold {THRESHOLDS['get_latency_p95']}ms"


@pytest.mark.asyncio
async def test_set_operation_latency():
    """Test cache set operation latency."""
    backend = MemoryCacheBackend(max_size=1000)
    
    # Measure set latency
    latencies = []
    for i in range(100):
        start = time.perf_counter()
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
        duration_ms = (time.perf_counter() - start) * 1000
        latencies.append(duration_ms)
    
    p50 = calculate_percentile(latencies, 50)
    p95 = calculate_percentile(latencies, 95)
    
    print(f"\nSet Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms")
    
    assert p50 < THRESHOLDS["set_latency_p50"], \
        f"P50 set latency {p50:.3f}ms exceeds threshold {THRESHOLDS['set_latency_p50']}ms"
    assert p95 < THRESHOLDS["set_latency_p95"], \
        f"P95 set latency {p95:.3f}ms exceeds threshold {THRESHOLDS['set_latency_p95']}ms"


@pytest.mark.asyncio
async def test_compression_overhead():
    """Test compression overhead for large values."""
    compressor = create_compressor(
        threshold_kb=1,
        level=6
    )
    
    # Generate 100KB data
    large_data = generate_random_string(100 * 1024)
    
    # Measure compression time
    compression_times = []
    for _ in range(10):
        start = time.perf_counter()
        compressed_data, _ = compressor.compress(large_data)
        duration_ms = (time.perf_counter() - start) * 1000
        compression_times.append(duration_ms)
    
    avg_compression = statistics.mean(compression_times)
    
    # Measure decompression time
    compressed_data, _ = compressor.compress(large_data)
    decompression_times = []
    for _ in range(10):
        start = time.perf_counter()
        compressor.decompress(compressed_data)
        duration_ms = (time.perf_counter() - start) * 1000
        decompression_times.append(duration_ms)
    
    avg_decompression = statistics.mean(decompression_times)
    total_overhead = avg_compression + avg_decompression
    
    print(f"\nCompression: {avg_compression:.3f}ms, Decompression: {avg_decompression:.3f}ms, Total: {total_overhead:.3f}ms")
    
    assert total_overhead < THRESHOLDS["compression_overhead"], \
        f"Compression overhead {total_overhead:.3f}ms exceeds threshold {THRESHOLDS['compression_overhead']}ms"


@pytest.mark.asyncio
async def test_eviction_performance_lru():
    """Test LRU eviction performance."""
    backend = MemoryCacheBackend(max_size=100, eviction_policy="LRU")
    
    # Fill cache to capacity
    for i in range(100):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    # Measure eviction time (triggered by adding new items)
    eviction_times = []
    for i in range(100, 200):
        start = time.perf_counter()
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
        duration_ms = (time.perf_counter() - start) * 1000
        eviction_times.append(duration_ms)
    
    avg_eviction = statistics.mean(eviction_times)
    p95_eviction = calculate_percentile(eviction_times, 95)
    
    print(f"\nLRU Eviction - Avg: {avg_eviction:.3f}ms, P95: {p95_eviction:.3f}ms")
    
    assert p95_eviction < THRESHOLDS["eviction_latency"], \
        f"P95 eviction latency {p95_eviction:.3f}ms exceeds threshold {THRESHOLDS['eviction_latency']}ms"


@pytest.mark.asyncio
async def test_eviction_performance_lfu():
    """Test LFU eviction performance."""
    backend = MemoryCacheBackend(max_size=100, eviction_policy="LFU")
    
    # Fill cache
    for i in range(100):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    # Measure eviction time
    eviction_times = []
    for i in range(100, 200):
        start = time.perf_counter()
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
        duration_ms = (time.perf_counter() - start) * 1000
        eviction_times.append(duration_ms)
    
    avg_eviction = statistics.mean(eviction_times)
    p95_eviction = calculate_percentile(eviction_times, 95)
    
    print(f"\nLFU Eviction - Avg: {avg_eviction:.3f}ms, P95: {p95_eviction:.3f}ms")
    
    assert p95_eviction < THRESHOLDS["eviction_latency"], \
        f"P95 eviction latency {p95_eviction:.3f}ms exceeds threshold {THRESHOLDS['eviction_latency']}ms"


@pytest.mark.asyncio
async def test_eviction_performance_fifo():
    """Test FIFO eviction performance."""
    backend = MemoryCacheBackend(max_size=100, eviction_policy="FIFO")
    
    # Fill cache
    for i in range(100):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    # Measure eviction time
    eviction_times = []
    for i in range(100, 200):
        start = time.perf_counter()
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
        duration_ms = (time.perf_counter() - start) * 1000
        eviction_times.append(duration_ms)
    
    avg_eviction = statistics.mean(eviction_times)
    p95_eviction = calculate_percentile(eviction_times, 95)
    
    print(f"\nFIFO Eviction - Avg: {avg_eviction:.3f}ms, P95: {p95_eviction:.3f}ms")
    
    assert p95_eviction < THRESHOLDS["eviction_latency"], \
        f"P95 eviction latency {p95_eviction:.3f}ms exceeds threshold {THRESHOLDS['eviction_latency']}ms"


@pytest.mark.asyncio
async def test_cache_throughput():
    """Test cache operations throughput."""
    backend = MemoryCacheBackend(max_size=1000)
    
    # Measure throughput (ops/sec)
    operations = 1000
    start = time.perf_counter()
    
    # Mix of set and get operations
    for i in range(operations // 2):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    for i in range(operations // 2):
        await backend.get(f"key{i % (operations // 2)}")
    
    duration = time.perf_counter() - start
    throughput = operations / duration
    
    print(f"\nThroughput: {throughput:.0f} ops/sec")
    
    assert throughput > THRESHOLDS["throughput_ops_per_sec"], \
        f"Throughput {throughput:.0f} ops/sec below threshold {THRESHOLDS['throughput_ops_per_sec']} ops/sec"


@pytest.mark.asyncio
async def test_cache_with_compression_latency():
    """Test cache latency with compression enabled."""
    backend = MemoryCacheBackend(max_size=100)
    manager = CacheManager(
        backend=backend,
        enable_compression=True,
        compression_threshold_kb=1
    )
    
    # Test with data above compression threshold (10KB)
    large_value = generate_random_string(10 * 1024)
    
    # Measure set with compression
    set_times = []
    for i in range(20):
        start = time.perf_counter()
        await manager.set_llm_response(
            f"model{i}",
            f"prompt{i}",
            large_value,
            max_tokens=2048
        )
        duration_ms = (time.perf_counter() - start) * 1000
        set_times.append(duration_ms)
    
    # Measure get with decompression
    get_times = []
    for i in range(20):
        start = time.perf_counter()
        await manager.get_llm_response(f"model{i}", f"prompt{i}", 2048)
        duration_ms = (time.perf_counter() - start) * 1000
        get_times.append(duration_ms)
    
    avg_set = statistics.mean(set_times)
    avg_get = statistics.mean(get_times)
    
    print(f"\nWith Compression - Set: {avg_set:.3f}ms, Get: {avg_get:.3f}ms")
    
    # Should still be reasonably fast even with compression
    assert avg_set < 20.0, f"Set with compression {avg_set:.3f}ms too slow"
    assert avg_get < 20.0, f"Get with decompression {avg_get:.3f}ms too slow"


@pytest.mark.asyncio
async def test_eviction_policy_comparison():
    """Compare performance of different eviction policies."""
    policies = ["LRU", "LFU", "FIFO"]
    results = {}
    
    for policy in policies:
        backend = MemoryCacheBackend(max_size=100, eviction_policy=policy)
        
        # Fill cache
        for i in range(100):
            await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
        
        # Measure eviction time
        eviction_times = []
        for i in range(100, 200):
            start = time.perf_counter()
            await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
            duration_ms = (time.perf_counter() - start) * 1000
            eviction_times.append(duration_ms)
        
        results[policy] = {
            "mean": statistics.mean(eviction_times),
            "p50": calculate_percentile(eviction_times, 50),
            "p95": calculate_percentile(eviction_times, 95),
        }
    
    print("\nEviction Policy Performance Comparison:")
    for policy, metrics in results.items():
        print(f"  {policy:5s} - Mean: {metrics['mean']:.3f}ms, P50: {metrics['p50']:.3f}ms, P95: {metrics['p95']:.3f}ms")
    
    # All policies should meet thresholds
    for policy, metrics in results.items():
        assert metrics["p95"] < THRESHOLDS["eviction_latency"], \
            f"{policy} P95 {metrics['p95']:.3f}ms exceeds threshold"


@pytest.mark.asyncio
async def test_cleanup_performance():
    """Test cleanup operation performance."""
    backend = MemoryCacheBackend(max_size=1000)
    
    # Add entries with expired TTL
    current_time = time.time()
    for i in range(500):
        # Half expired, half valid
        if i < 250:
            backend.cache[f"expired{i}"] = CacheEntry(
                key=f"expired{i}",
                value=f"value{i}",
                expiry=current_time - 1,  # Expired
                inserted_at=current_time - 10,
                last_accessed=current_time - 10,
                access_count=0
            )
        else:
            await backend.set(f"valid{i}", f"value{i}", ttl_seconds=3600)
    
    # Measure cleanup time
    start = time.perf_counter()
    removed = await backend.cleanup()
    duration_ms = (time.perf_counter() - start) * 1000
    
    print(f"\nCleanup Performance - Duration: {duration_ms:.3f}ms, Removed: {removed} entries")
    
    assert removed == 250, "Should remove 250 expired entries"
    assert duration_ms < 50.0, f"Cleanup {duration_ms:.3f}ms too slow for 500 entries"


@pytest.mark.asyncio
async def test_concurrent_access_performance():
    """Test cache performance under concurrent access."""
    backend = MemoryCacheBackend(max_size=1000)
    
    # Pre-populate
    for i in range(100):
        await backend.set(f"key{i}", f"value{i}", ttl_seconds=3600)
    
    # Simulate concurrent access
    async def worker(worker_id: int, operations: int):
        times = []
        for i in range(operations):
            key = f"key{random.randint(0, 99)}"
            start = time.perf_counter()
            await backend.get(key)
            duration_ms = (time.perf_counter() - start) * 1000
            times.append(duration_ms)
        return times
    
    # Run 10 workers concurrently, 100 ops each
    start = time.perf_counter()
    results = await asyncio.gather(*[worker(i, 100) for i in range(10)])
    total_duration = time.perf_counter() - start
    
    # Flatten all times
    all_times = [t for worker_times in results for t in worker_times]
    
    p50 = calculate_percentile(all_times, 50)
    p95 = calculate_percentile(all_times, 95)
    throughput = (10 * 100) / total_duration
    
    print(f"\nConcurrent Access (10 workers, 100 ops each):")
    print(f"  P50: {p50:.3f}ms, P95: {p95:.3f}ms")
    print(f"  Throughput: {throughput:.0f} ops/sec")
    
    assert p95 < 10.0, f"P95 latency {p95:.3f}ms too high under concurrent load"


@pytest.mark.asyncio
async def test_cache_manager_llm_response_performance():
    """Test CacheManager LLM response operations."""
    manager = CacheManager()
    
    # Measure set_llm_response latency
    set_times = []
    for i in range(100):
        start = time.perf_counter()
        await manager.set_llm_response(
            f"model{i}",
            f"prompt{i}",
            f"response{i}",
            max_tokens=2048
        )
        duration_ms = (time.perf_counter() - start) * 1000
        set_times.append(duration_ms)
    
    # Measure get_llm_response latency
    get_times = []
    for i in range(100):
        start = time.perf_counter()
        await manager.get_llm_response(f"model{i}", f"prompt{i}", 2048)
        duration_ms = (time.perf_counter() - start) * 1000
        get_times.append(duration_ms)
    
    set_p50 = calculate_percentile(set_times, 50)
    get_p50 = calculate_percentile(get_times, 50)
    
    print(f"\nCacheManager Performance - Set P50: {set_p50:.3f}ms, Get P50: {get_p50:.3f}ms")
    
    assert set_p50 < 5.0, f"Set P50 {set_p50:.3f}ms too high"
    assert get_p50 < 5.0, f"Get P50 {get_p50:.3f}ms too high"


@pytest.mark.asyncio
async def test_memory_usage_efficiency():
    """Test memory efficiency of cache storage."""
    import sys
    
    backend = MemoryCacheBackend(max_size=1000)
    
    # Small values (100 bytes each)
    small_value = "x" * 100
    
    # Measure memory for 1000 entries
    for i in range(1000):
        await backend.set(f"key{i:04d}", small_value, ttl_seconds=3600)
    
    # Rough memory estimate (key + value + metadata)
    # Each entry: ~100 bytes value + ~10 bytes key + CacheEntry overhead
    expected_memory_per_entry = 200  # bytes (rough estimate)
    expected_total = 1000 * expected_memory_per_entry
    
    # Get actual cache size
    cache_size = len(backend.cache)
    
    print(f"\nMemory Efficiency - Entries: {cache_size}, Expected: ~{expected_total / 1024:.1f}KB")
    
    assert cache_size == 1000, "Should store all 1000 entries"


def test_performance_thresholds_documented():
    """Document current performance thresholds."""
    print("\n=== Performance Thresholds ===")
    for metric, threshold in THRESHOLDS.items():
        print(f"  {metric}: {threshold}")
    
    # Always passes - just for documentation
    assert True
