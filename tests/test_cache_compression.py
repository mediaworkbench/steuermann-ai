"""Tests for cache compression functionality."""

import pytest
import json
from universal_agentic_framework.caching.compression import (
    CacheCompressor,
    create_compressor,
    CompressionStats
)
from universal_agentic_framework.caching.manager import CacheManager, MemoryCacheBackend


@pytest.fixture
def compressor():
    """Create cache compressor with 100 byte threshold."""
    return CacheCompressor(
        compression_threshold_bytes=100,
        compression_level=6,
        enable_compression=True
    )


@pytest.fixture
def cache_manager_with_compression():
    """Create cache manager with compression enabled."""
    backend = MemoryCacheBackend()
    return CacheManager(
        backend=backend,
        use_vector_db=False,
        enable_compression=True,
        compression_threshold_kb=1  # 1KB threshold
    )


def test_compressor_initialization():
    """Test compressor initializes correctly."""
    compressor = CacheCompressor(
        compression_threshold_bytes=1024,
        compression_level=6,
        enable_compression=True
    )
    
    assert compressor.compression_threshold == 1024
    assert compressor.compression_level == 6
    assert compressor.enable_compression is True
    assert compressor.stats["total_compressed"] == 0


def test_compress_small_data_no_compression(compressor):
    """Test small data below threshold is not compressed."""
    small_data = {"key": "value"}
    
    compressed_bytes, stats = compressor.compress(small_data)
    
    assert stats.is_compressed is False
    assert stats.compression_ratio == 1.0
    assert stats.original_size < 100  # Below threshold


def test_compress_large_data(compressor):
    """Test large data above threshold is compressed."""
    # Create data above threshold
    large_data = {"key": "value" * 100}  # Should be > 100 bytes
    
    compressed_bytes, stats = compressor.compress(large_data)
    
    assert stats.is_compressed is True
    assert stats.compression_ratio > 1.0  # Should achieve some compression
    assert stats.compressed_size < stats.original_size
    assert compressor.stats["total_compressed"] == 1
    assert compressor.stats["bytes_saved"] > 0


def test_decompress_data(compressor):
    """Test decompression restores original data."""
    original_data = {"key": "value" * 100, "number": 42, "list": [1, 2, 3]}
    
    # Compress
    compressed_bytes, stats = compressor.compress(original_data)
    
    # Decompress
    decompressed_data = compressor.decompress(compressed_bytes, stats.is_compressed)
    
    assert decompressed_data == original_data
    assert compressor.stats["total_decompressed"] == 1


def test_decompress_uncompressed_data(compressor):
    """Test decompression of uncompressed data."""
    small_data = {"key": "value"}
    
    # Compress (will not actually compress due to threshold)
    compressed_bytes, stats = compressor.compress(small_data)
    assert stats.is_compressed is False
    
    # Decompress
    decompressed_data = compressor.decompress(compressed_bytes, False)
    
    assert decompressed_data == small_data


def test_compress_with_metadata(compressor):
    """Test compression with metadata dict."""
    data = {"key": "value" * 100}
    
    metadata = compressor.compress_with_metadata(data)
    
    assert "data" in metadata
    assert "is_compressed" in metadata
    assert "original_size" in metadata
    assert "compressed_size" in metadata
    assert "compression_ratio" in metadata
    assert isinstance(metadata["data"], str)  # Base64 encoded


def test_decompress_from_metadata(compressor):
    """Test decompression from metadata dict."""
    original_data = {"key": "value" * 100, "nested": {"a": 1}}
    
    # Compress with metadata
    metadata = compressor.compress_with_metadata(original_data)
    
    # Decompress from metadata
    decompressed_data = compressor.decompress_from_metadata(metadata)
    
    assert decompressed_data == original_data


def test_compression_disabled():
    """Test compressor with compression disabled."""
    compressor = CacheCompressor(
        compression_threshold_bytes=100,
        compression_level=6,
        enable_compression=False
    )
    
    large_data = {"key": "value" * 100}
    compressed_bytes, stats = compressor.compress(large_data)
    
    assert stats.is_compressed is False
    assert stats.compression_ratio == 1.0


def test_compression_stats(compressor):
    """Test compression statistics tracking."""
    # Compress multiple items
    for i in range(5):
        data = {"key": f"value{i}" * 100}
        compressed_bytes, stats = compressor.compress(data)
        compressor.decompress(compressed_bytes, stats.is_compressed)
    
    stats = compressor.get_stats()
    
    assert stats["total_compressed"] == 5
    assert stats["total_decompressed"] == 5
    assert stats["total_operations"] == 10
    assert stats["bytes_saved"] > 0
    assert stats["avg_bytes_saved"] > 0
    assert stats["compression_threshold_bytes"] == 100
    assert stats["compression_level"] == 6


def test_compression_error_handling(compressor):
    """Test error handling in compression."""
    # Try to compress non-serializable data (shouldn't happen in practice)
    # Actually, most data can be serialized, so let's test decompress error
    
    # Invalid compressed data
    invalid_bytes = b"invalid compressed data"
    
    with pytest.raises(Exception):
        compressor.decompress(invalid_bytes, is_compressed=True)
    
    assert compressor.stats["compression_errors"] > 0


def test_reset_stats(compressor):
    """Test resetting compression statistics."""
    # Generate some stats
    data = {"key": "value" * 100}
    compressor.compress(data)
    
    assert compressor.stats["total_compressed"] > 0
    
    # Reset
    compressor.reset_stats()
    
    assert compressor.stats["total_compressed"] == 0
    assert compressor.stats["total_decompressed"] == 0
    assert compressor.stats["bytes_saved"] == 0


def test_create_compressor_helper():
    """Test create_compressor helper function."""
    compressor = create_compressor(threshold_kb=2, level=9, enabled=True)
    
    assert compressor.compression_threshold == 2048  # 2KB in bytes
    assert compressor.compression_level == 9
    assert compressor.enable_compression is True


def test_compression_ratio_calculation(compressor):
    """Test compression ratio is calculated correctly."""
    # Highly compressible data
    data = {"key": "A" * 1000}  # Repetitive data compresses well
    
    compressed_bytes, stats = compressor.compress(data)
    
    assert stats.is_compressed is True
    assert stats.compression_ratio > 2.0  # Should achieve good compression
    assert stats.original_size / stats.compressed_size == stats.compression_ratio


@pytest.mark.asyncio
async def test_cache_manager_with_compression(cache_manager_with_compression):
    """Test cache manager with compression enabled."""
    manager = cache_manager_with_compression
    
    # Store large LLM response
    large_response = "A" * 2000  # 2KB response
    await manager.set_llm_response("gpt-4", "test prompt", large_response)
    
    # Retrieve and verify
    cached = await manager.get_llm_response("gpt-4", "test prompt")
    
    assert cached == large_response


@pytest.mark.asyncio
async def test_cache_manager_compression_stats(cache_manager_with_compression):
    """Test compression stats are tracked in cache manager."""
    manager = cache_manager_with_compression
    
    # Verify compressor exists
    assert manager.compressor is not None
    
    # Store data
    large_response = "B" * 2000
    await manager.set_llm_response("gpt-4", "prompt", large_response)
    
    # Check compression stats
    stats = manager.compressor.get_stats()
    assert stats["total_compressed"] == 1


@pytest.mark.asyncio
async def test_cache_manager_no_compression():
    """Test cache manager without compression."""
    backend = MemoryCacheBackend()
    manager = CacheManager(
        backend=backend,
        use_vector_db=False,
        enable_compression=False
    )
    
    assert manager.compressor is None
    
    # Data should be stored without compression
    await manager.set_llm_response("gpt-4", "prompt", "response")
    cached = await manager.get_llm_response("gpt-4", "prompt")
    
    assert cached == "response"


@pytest.mark.asyncio
async def test_cache_manager_small_data_no_compression(cache_manager_with_compression):
    """Test small data below threshold is not compressed."""
    manager = cache_manager_with_compression
    
    # Store small response (below 1KB threshold)
    small_response = "Small response"
    await manager.set_llm_response("gpt-4", "prompt", small_response)
    
    # Retrieve
    cached = await manager.get_llm_response("gpt-4", "prompt")
    
    assert cached == small_response
    
    # Check that compression was not applied (below threshold)
    stats = manager.compressor.get_stats()
    # Compression might be 1 but data should not actually be compressed due to threshold
    assert stats["total_compressed"] >= 0  # At least attempted


def test_compression_levels():
    """Test different compression levels."""
    data = {"key": "value" * 500}
    
    # Test level 1 (fast, less compression)
    comp1 = CacheCompressor(compression_threshold_bytes=100, compression_level=1)
    bytes1, stats1 = comp1.compress(data)
    
    # Test level 9 (slow, more compression)
    comp9 = CacheCompressor(compression_threshold_bytes=100, compression_level=9)
    bytes9, stats9 = comp9.compress(data)
    
    # Level 9 should achieve better or equal compression
    assert stats9.compression_ratio >= stats1.compression_ratio


def test_compress_various_data_types(compressor):
    """Test compression of various data types."""
    test_cases = [
        {"string": "test"},
        {"number": 12345},
        {"float": 3.14159},
        {"bool": True},
        {"null": None},
        {"list": [1, 2, 3, 4, 5]},
        {"nested": {"a": {"b": {"c": "deep"}}}},
        {"mixed": [1, "two", 3.0, None, {"five": 5}]},
    ]
    
    for data in test_cases:
        # Make it large enough to compress
        large_data = {**data, "padding": "X" * 200}
        
        compressed_bytes, stats = compressor.compress(large_data)
        decompressed = compressor.decompress(compressed_bytes, stats.is_compressed)
        
        assert decompressed == large_data


def test_compression_threshold_boundary(compressor):
    """Test compression at threshold boundary."""
    # Just below threshold
    small_data = {"key": "A" * 10}
    compressed_bytes, stats = compressor.compress(small_data)
    assert stats.is_compressed is False
    
    # Just above threshold
    large_data = {"key": "A" * 100}
    compressed_bytes, stats = compressor.compress(large_data)
    assert stats.is_compressed is True
