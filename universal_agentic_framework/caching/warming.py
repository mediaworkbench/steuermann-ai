"""Cache warming strategies for preloading common queries.

Implements various cache warming patterns:
- Preload common queries from configuration
- Time-based warming (e.g., daily reports at start of day)
- Usage pattern-based warming (most frequently accessed)
- Predictive warming based on user behavior
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from universal_agentic_framework.caching.manager import CacheManager

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Cache warming service for preloading common queries."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize cache warmer.
        
        Args:
            cache_manager: CacheManager instance to warm
            config: Warming configuration:
                - warming_queries: List of queries to preload
                - warm_on_startup: Whether to warm cache on startup
                - warming_interval_hours: How often to refresh (default: 24)
        """
        self.cache_manager = cache_manager
        self.config = config or {}
        self.warming_queries = self.config.get("warming_queries", [])
        self.warm_on_startup = self.config.get("warm_on_startup", True)
        self.warming_interval_hours = self.config.get("warming_interval_hours", 24)
        
        logger.info(
            f"CacheWarmer initialized with {len(self.warming_queries)} queries"
        )
    
    async def warm_crew_cache(
        self,
        crew_name: str,
        user_id: str,
        query: str,
        result_generator: Callable,
        language: str = "en",
        ttl_seconds: int = 86400
    ) -> bool:
        """Warm cache for a specific crew query.
        
        Args:
            crew_name: Name of the crew
            user_id: User ID for the query
            query: Query text to cache
            result_generator: Async function that generates the result
            language: Language code
            ttl_seconds: TTL for cache entry
        
        Returns:
            True if warming succeeded, False otherwise
        """
        try:
            logger.info(f"Warming cache for crew={crew_name}, query='{query}'")
            
            # Check if already cached
            existing = await self.cache_manager.get_crew_result(
                crew_name, user_id, query, language
            )
            
            if existing is not None:
                logger.debug(f"Cache already warm for query: {query}")
                return True
            
            # Generate result
            result = await result_generator()
            
            # Store in cache
            await self.cache_manager.set_crew_result(
                crew_name, user_id, query, result, language, ttl_seconds
            )
            
            logger.info(f"Successfully warmed cache for query: {query}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to warm cache for query '{query}': {e}", exc_info=True)
            return False
    
    async def warm_from_config(self) -> Dict[str, Any]:
        """Warm cache from configured queries.
        
        Config format:
        {
            "warming_queries": [
                {
                    "crew_name": "research",
                    "user_id": "system",
                    "query": "What is AI?",
                    "result": {"output": "AI is..."},
                    "language": "en",
                    "ttl_seconds": 86400
                }
            ]
        }
        
        Returns:
            Dict with warming statistics
        """
        stats = {
            "total": len(self.warming_queries),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "duration_seconds": 0.0
        }
        
        if not self.warming_queries:
            logger.info("No warming queries configured")
            return stats
        
        logger.info(f"Warming cache with {len(self.warming_queries)} queries")
        start_time = datetime.now()
        
        for query_config in self.warming_queries:
            try:
                crew_name = query_config.get("crew_name")
                user_id = query_config.get("user_id", "system")
                query = query_config.get("query")
                result = query_config.get("result")
                language = query_config.get("language", "en")
                ttl_seconds = query_config.get("ttl_seconds", 86400)
                
                if not all([crew_name, query, result]):
                    logger.warning(f"Skipping invalid warming query: {query_config}")
                    stats["skipped"] += 1
                    continue
                
                # Check if already cached
                existing = await self.cache_manager.get_crew_result(
                    crew_name, user_id, query, language
                )
                
                if existing is not None:
                    logger.debug(f"Cache already warm for: {query}")
                    stats["skipped"] += 1
                    continue
                
                # Store result
                await self.cache_manager.set_crew_result(
                    crew_name, user_id, query, result, language, ttl_seconds
                )
                
                stats["success"] += 1
                logger.debug(f"Warmed cache for: {query}")
                
            except Exception as e:
                logger.error(f"Failed to warm query '{query_config}': {e}")
                stats["failed"] += 1
        
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Cache warming completed: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['skipped']} skipped "
            f"in {stats['duration_seconds']:.2f}s"
        )
        
        return stats
    
    async def warm_top_queries(
        self,
        top_n: int = 10,
        query_generator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Warm cache with top N most frequent queries.
        
        Args:
            top_n: Number of top queries to warm
            query_generator: Async function that returns list of (crew, user, query, result) tuples
        
        Returns:
            Dict with warming statistics
        """
        if query_generator is None:
            logger.warning("No query generator provided for top queries warming")
            return {"total": 0, "success": 0, "failed": 0}
        
        stats = {"total": 0, "success": 0, "failed": 0, "duration_seconds": 0.0}
        start_time = datetime.now()
        
        try:
            # Get top queries
            top_queries = await query_generator(top_n)
            stats["total"] = len(top_queries)
            
            logger.info(f"Warming cache with {len(top_queries)} top queries")
            
            for crew_name, user_id, query, result, language in top_queries:
                try:
                    # Check if already cached
                    existing = await self.cache_manager.get_crew_result(
                        crew_name, user_id, query, language
                    )
                    
                    if existing is None:
                        # Store result
                        await self.cache_manager.set_crew_result(
                            crew_name, user_id, query, result, language
                        )
                        stats["success"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to warm top query '{query}': {e}")
                    stats["failed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to get top queries: {e}", exc_info=True)
        
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Top queries warming completed: {stats['success']} success, "
            f"{stats['failed']} failed in {stats['duration_seconds']:.2f}s"
        )
        
        return stats
    
    async def warm_scheduled(self) -> Dict[str, Any]:
        """Execute scheduled cache warming.
        
        This method should be called by the scheduler at regular intervals.
        It refreshes the cache with configured queries.
        
        Returns:
            Dict with warming statistics
        """
        logger.info("Starting scheduled cache warming")
        return await self.warm_from_config()
    
    async def warm_on_demand(
        self,
        crew_name: str,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Warm cache on-demand for specific crew and queries.
        
        Args:
            crew_name: Name of the crew
            queries: List of query dicts with keys: query, result, user_id, language
        
        Returns:
            Dict with warming statistics
        """
        stats = {
            "total": len(queries),
            "success": 0,
            "failed": 0,
            "duration_seconds": 0.0
        }
        
        start_time = datetime.now()
        logger.info(f"On-demand warming for {crew_name} with {len(queries)} queries")
        
        for query_config in queries:
            try:
                query = query_config.get("query")
                result = query_config.get("result")
                user_id = query_config.get("user_id", "system")
                language = query_config.get("language", "en")
                ttl_seconds = query_config.get("ttl_seconds", 86400)
                
                if not query or not result:
                    logger.warning(f"Skipping invalid query config: {query_config}")
                    continue
                
                await self.cache_manager.set_crew_result(
                    crew_name, user_id, query, result, language, ttl_seconds
                )
                
                stats["success"] += 1
                
            except Exception as e:
                logger.error(f"Failed to warm query '{query_config}': {e}")
                stats["failed"] += 1
        
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"On-demand warming completed: {stats['success']} success, "
            f"{stats['failed']} failed in {stats['duration_seconds']:.2f}s"
        )
        
        return stats


async def create_cache_warmer(
    cache_manager: CacheManager,
    config: Optional[Dict[str, Any]] = None,
    warm_on_create: bool = False
) -> CacheWarmer:
    """Create cache warmer and optionally warm cache immediately.
    
    Args:
        cache_manager: CacheManager instance
        config: Warmer configuration
        warm_on_create: Whether to warm cache immediately
    
    Returns:
        CacheWarmer instance
    """
    warmer = CacheWarmer(cache_manager, config)
    
    if warm_on_create and warmer.warm_on_startup:
        await warmer.warm_from_config()
    
    return warmer
