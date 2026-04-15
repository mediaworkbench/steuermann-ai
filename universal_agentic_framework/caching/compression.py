"""Cache compression utilities for reducing memory footprint.

Implements compression/decompression for cached data:
- gzip compression for large cache entries
- Configurable compression threshold
- Automatic compression ratio tracking
- Transparent compression/decompression
"""

import gzip
import json
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from universal_agentic_framework.monitoring.metrics import track_cache_operation

logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics for compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    is_compressed: bool


class CacheCompressor:
    """Cache compression manager."""
    
    def __init__(
        self,
        compression_threshold_bytes: int = 1024,
        compression_level: int = 6,
        enable_compression: bool = True
    ):
        """Initialize cache compressor.
        
        Args:
            compression_threshold_bytes: Minimum size to compress (default: 1KB)
            compression_level: gzip compression level 1-9 (default: 6)
            enable_compression: Whether to enable compression (default: True)
        """
        self.compression_threshold = compression_threshold_bytes
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        self.stats = {
            "total_compressed": 0,
            "total_decompressed": 0,
            "bytes_saved": 0,
            "compression_errors": 0,
        }
        
        logger.info(
            f"CacheCompressor initialized: threshold={compression_threshold_bytes}B, "
            f"level={compression_level}, enabled={enable_compression}"
        )
    
    def compress(self, data: Any) -> Tuple[bytes, CompressionStats]:
        """Compress data if above threshold.
        
        Args:
            data: Data to compress (will be JSON-serialized)
        
        Returns:
            Tuple of (compressed_bytes, stats)
        """
        try:
            # Serialize to JSON
            json_data = json.dumps(data)
            original_bytes = json_data.encode('utf-8')
            original_size = len(original_bytes)
            
            # Check if compression is enabled and data is above threshold
            if not self.enable_compression or original_size < self.compression_threshold:
                stats = CompressionStats(
                    original_size=original_size,
                    compressed_size=original_size,
                    compression_ratio=1.0,
                    is_compressed=False
                )
                return original_bytes, stats
            
            # Compress data
            compressed_bytes = gzip.compress(
                original_bytes,
                compresslevel=self.compression_level
            )
            compressed_size = len(compressed_bytes)
            
            # Calculate compression ratio
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Update stats
            self.stats["total_compressed"] += 1
            self.stats["bytes_saved"] += (original_size - compressed_size)
            
            stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                is_compressed=True
            )
            
            logger.debug(
                f"Compressed data: {original_size}B -> {compressed_size}B "
                f"(ratio: {compression_ratio:.2f}x)"
            )
            
            return compressed_bytes, stats
            
        except Exception as e:
            logger.error(f"Compression failed: {e}", exc_info=True)
            self.stats["compression_errors"] += 1
            # Return original data uncompressed
            json_data = json.dumps(data)
            original_bytes = json_data.encode('utf-8')
            stats = CompressionStats(
                original_size=len(original_bytes),
                compressed_size=len(original_bytes),
                compression_ratio=1.0,
                is_compressed=False
            )
            return original_bytes, stats
    
    def decompress(self, data: bytes, is_compressed: bool = True) -> Any:
        """Decompress data and deserialize to original format.
        
        Args:
            data: Compressed or uncompressed bytes
            is_compressed: Whether data is compressed
        
        Returns:
            Deserialized data
        """
        try:
            if is_compressed:
                # Decompress
                decompressed_bytes = gzip.decompress(data)
                self.stats["total_decompressed"] += 1
                logger.debug(f"Decompressed data: {len(data)}B -> {len(decompressed_bytes)}B")
            else:
                decompressed_bytes = data
            
            # Deserialize JSON
            json_str = decompressed_bytes.decode('utf-8')
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}", exc_info=True)
            self.stats["compression_errors"] += 1
            raise
    
    def compress_with_metadata(self, data: Any) -> Dict[str, Any]:
        """Compress data and return with metadata.
        
        Returns dict with:
        - data: compressed bytes (base64 encoded for JSON compatibility)
        - is_compressed: bool
        - original_size: int
        - compressed_size: int
        
        Args:
            data: Data to compress
        
        Returns:
            Dict with compressed data and metadata
        """
        import base64
        
        compressed_bytes, stats = self.compress(data)
        
        return {
            "data": base64.b64encode(compressed_bytes).decode('utf-8'),
            "is_compressed": stats.is_compressed,
            "original_size": stats.original_size,
            "compressed_size": stats.compressed_size,
            "compression_ratio": stats.compression_ratio,
        }
    
    def decompress_from_metadata(self, metadata: Dict[str, Any]) -> Any:
        """Decompress data from metadata dict.
        
        Args:
            metadata: Dict with compressed data and metadata
        
        Returns:
            Deserialized original data
        """
        import base64
        
        compressed_bytes = base64.b64decode(metadata["data"])
        is_compressed = metadata.get("is_compressed", True)
        
        return self.decompress(compressed_bytes, is_compressed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics.
        
        Returns:
            Dict with compression stats
        """
        total_ops = self.stats["total_compressed"] + self.stats["total_decompressed"]
        avg_bytes_saved = (
            self.stats["bytes_saved"] / self.stats["total_compressed"]
            if self.stats["total_compressed"] > 0
            else 0
        )
        
        return {
            **self.stats,
            "total_operations": total_ops,
            "avg_bytes_saved": avg_bytes_saved,
            "compression_threshold_bytes": self.compression_threshold,
            "compression_level": self.compression_level,
            "enabled": self.enable_compression,
        }
    
    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self.stats = {
            "total_compressed": 0,
            "total_decompressed": 0,
            "bytes_saved": 0,
            "compression_errors": 0,
        }
        logger.info("Compression stats reset")


def create_compressor(
    threshold_kb: int = 1,
    level: int = 6,
    enabled: bool = True
) -> CacheCompressor:
    """Create cache compressor with convenient KB threshold.
    
    Args:
        threshold_kb: Minimum size to compress in KB (default: 1KB)
        level: Compression level 1-9 (default: 6)
        enabled: Whether to enable compression (default: True)
    
    Returns:
        CacheCompressor instance
    """
    return CacheCompressor(
        compression_threshold_bytes=threshold_kb * 1024,
        compression_level=level,
        enable_compression=enabled
    )
