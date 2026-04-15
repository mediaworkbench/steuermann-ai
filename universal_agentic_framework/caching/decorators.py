"""Decorators for caching integration in handler functions."""

import asyncio
import functools
import logging
from typing import Callable, Optional, Any
import os

from .manager import CacheManager, RedisCacheBackend, MemoryCacheBackend

logger = logging.getLogger(__name__)


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache(use_redis: bool = True) -> CacheManager:
    """Initialize global cache manager.
    
    Args:
        use_redis: Use Redis backend (default) or memory backend
        
    Returns:
        Initialized CacheManager instance
    """
    global _cache_manager
    
    if use_redis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        backend = RedisCacheBackend(redis_url)
        logger.info(f"Initializing Redis cache: {redis_url}")
    else:
        backend = MemoryCacheBackend()
        logger.info("Initializing memory cache")
    
    _cache_manager = CacheManager(backend)
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager, initializing if needed."""
    global _cache_manager
    if _cache_manager is None:
        initialize_cache()
    return _cache_manager


def cache_llm_response(
    ttl_seconds: int = 86400,
    include_params: Optional[list] = None
) -> Callable:
    """Decorator for caching LLM responses.
    
    Wraps async functions that return LLM responses. Attempts cache hit
    before calling wrapped function, stores result in cache.
    
    Args:
        ttl_seconds: Cache time-to-live in seconds (default: 24 hours)
        include_params: List of parameter names to include in cache key
                       (default: ['model', 'prompt', 'max_tokens'])
    
    Example:
        @cache_llm_response(ttl_seconds=3600)
        async def get_answer(model: str, prompt: str) -> str:
            return await llm.generate(model, prompt)
    """
    if include_params is None:
        include_params = ["model", "prompt", "max_tokens"]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            cache = get_cache_manager()
            
            # Extract parameters for cache key
            cache_params = {}
            for param_name in include_params:
                if param_name in kwargs:
                    cache_params[param_name] = kwargs[param_name]
            
            # Try to get from cache
            if cache_params:
                try:
                    key = cache._make_key(func.__name__, *cache_params.values())
                    cached = await cache.backend.get(key)
                    if cached is not None:
                        logger.debug(f"Cache hit for {func.__name__}: {key}")
                        return cached
                except Exception as e:
                    logger.warning(f"Cache lookup error: {e}")
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Store in cache
            if result and cache_params:
                try:
                    key = cache._make_key(func.__name__, *cache_params.values())
                    await cache.backend.set(key, result, ttl_seconds)
                    logger.debug(f"Cached {func.__name__}: {key}")
                except Exception as e:
                    logger.warning(f"Cache store error: {e}")
            
            return result
        
        return wrapper
    
    return decorator
