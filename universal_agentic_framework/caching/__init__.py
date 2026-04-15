"""Caching module for performance optimization.

Provides Redis-backed and in-memory caching for LLM responses,
memory queries, and conversation summaries.
"""

from .manager import (
    CacheBackend,
    CacheManager,
    RedisCacheBackend,
    MemoryCacheBackend,
)
from .decorators import cache_llm_response

__all__ = [
    "CacheBackend",
    "CacheManager",
    "RedisCacheBackend",
    "MemoryCacheBackend",
    "cache_llm_response",
]
