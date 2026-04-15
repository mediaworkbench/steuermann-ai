"""Scheduled cache cleanup and maintenance tasks.

Uses APScheduler for periodic cache cleanup operations including:
- Expired key removal for in-memory caches
- TTL-based cleanup for Redis
- Cache statistics collection
- Cache warming for common queries
"""

import logging
import asyncio
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from universal_agentic_framework.caching.manager import CacheManager
from universal_agentic_framework.monitoring.metrics import (
    track_cache_cleanup,
    update_cache_size,
)

logger = logging.getLogger(__name__)


class CacheScheduler:
    """Scheduler for cache maintenance tasks."""
    
    def __init__(self, cache_manager: CacheManager, config: Optional[Dict[str, Any]] = None):
        """Initialize cache scheduler.
        
        Args:
            cache_manager: CacheManager instance to clean
            config: Scheduler configuration:
                - cleanup_interval_minutes: How often to run cleanup (default: 60)
                - stats_interval_minutes: How often to collect stats (default: 5)
                - enable_cleanup: Enable cleanup job (default: True)
                - enable_stats: Enable stats collection (default: True)
                - cleanup_cron: Cron expression for cleanup (overrides interval)
                - max_cache_age_hours: Max age for cache entries (default: 24)
        """
        self.cache_manager = cache_manager
        self.config = config or {}
        self.scheduler = AsyncIOScheduler()
        self._running = False
        
        # Configuration
        self.cleanup_interval = self.config.get("cleanup_interval_minutes", 60)
        self.stats_interval = self.config.get("stats_interval_minutes", 5)
        self.enable_cleanup = self.config.get("enable_cleanup", True)
        self.enable_stats = self.config.get("enable_stats", True)
        self.cleanup_cron = self.config.get("cleanup_cron", None)
        self.max_cache_age_hours = self.config.get("max_cache_age_hours", 24)
        
        logger.info(
            f"CacheScheduler initialized with "
            f"cleanup_interval={self.cleanup_interval}m, "
            f"stats_interval={self.stats_interval}m"
        )
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("CacheScheduler already running")
            return
        
        # Add cleanup job
        if self.enable_cleanup:
            if self.cleanup_cron:
                # Use cron expression
                self.scheduler.add_job(
                    self._cleanup_task,
                    CronTrigger.from_crontab(self.cleanup_cron),
                    id="cache_cleanup",
                    name="Cache Cleanup (Cron)",
                    replace_existing=True,
                )
                logger.info(f"Added cache cleanup job with cron: {self.cleanup_cron}")
            else:
                # Use interval
                self.scheduler.add_job(
                    self._cleanup_task,
                    IntervalTrigger(minutes=self.cleanup_interval),
                    id="cache_cleanup",
                    name="Cache Cleanup (Interval)",
                    replace_existing=True,
                )
                logger.info(f"Added cache cleanup job every {self.cleanup_interval} minutes")
        
        # Add stats collection job
        if self.enable_stats:
            self.scheduler.add_job(
                self._stats_collection_task,
                IntervalTrigger(minutes=self.stats_interval),
                id="cache_stats",
                name="Cache Stats Collection",
                replace_existing=True,
            )
            logger.info(f"Added cache stats job every {self.stats_interval} minutes")
        
        # Start scheduler
        self.scheduler.start()
        self._running = True
        logger.info("CacheScheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            logger.warning("CacheScheduler not running")
            return
        
        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("CacheScheduler stopped")
    
    async def _cleanup_task(self) -> None:
        """Execute cache cleanup task.
        
        Performs:
        1. Remove expired entries from memory cache backend
        2. Collect stats on cache efficiency
        3. Track cleanup metrics
        """
        try:
            logger.info("Starting scheduled cache cleanup")
            start_time = datetime.now()
            
            # Get stats before cleanup
            stats_before = self.cache_manager.get_stats()
            
            # Perform cleanup on memory backend
            backend = self.cache_manager.backend
            if hasattr(backend, 'cleanup'):
                # Use backend-specific cleanup if available
                removed_count = await backend.cleanup()
                logger.info(f"Backend cleanup removed {removed_count} entries")
            else:
                # Generic cleanup: iterate and remove expired
                removed_count = await self._cleanup_expired_entries()
                logger.info(f"Generic cleanup removed {removed_count} entries")
            
            # Get stats after cleanup
            stats_after = self.cache_manager.get_stats()
            
            # Calculate cleanup duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Track metrics
            track_cache_cleanup(
                removed_count=removed_count,
                duration_seconds=duration,
                cache_size_before=stats_before.get("hits", 0) + stats_before.get("misses", 0),
                cache_size_after=stats_after.get("hits", 0) + stats_after.get("misses", 0),
            )
            
            logger.info(
                f"Cache cleanup completed: removed {removed_count} entries "
                f"in {duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Cache cleanup task failed: {e}", exc_info=True)
    
    async def _cleanup_expired_entries(self) -> int:
        """Generic cleanup for expired entries in memory backend.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        backend = self.cache_manager.backend
        
        # Only works with MemoryCacheBackend
        if hasattr(backend, 'cache') and isinstance(backend.cache, dict):
            import time
            current_time = time.time()
            expired_keys = []
            
            for key, (value, expiry) in list(backend.cache.items()):
                if expiry is not None and current_time >= expiry:
                    expired_keys.append(key)
            
            # Remove expired keys
            for key in expired_keys:
                await backend.delete(key)
                removed += 1
        
        return removed
    
    async def _stats_collection_task(self) -> None:
        """Collect and track cache statistics.
        
        Tracks:
        - Hit rate
        - Cache size
        - Memory usage (if available)
        """
        try:
            stats = self.cache_manager.get_stats()
            
            # Track cache size
            cache_size = stats.get("hits", 0) + stats.get("misses", 0)
            update_cache_size(cache_size)
            
            # Log stats for visibility
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cache stats: {stats}")
            
        except Exception as e:
            logger.error(f"Cache stats collection failed: {e}", exc_info=True)
    
    def add_custom_job(
        self,
        func: Callable,
        trigger: str = "interval",
        **trigger_args
    ) -> None:
        """Add a custom scheduled job.
        
        Args:
            func: Async function to execute
            trigger: Trigger type ("interval" or "cron")
            **trigger_args: Arguments for trigger (e.g., minutes=30 for interval)
        
        Example:
            scheduler.add_custom_job(
                my_cache_warming_task,
                trigger="interval",
                hours=1
            )
        """
        if not self._running:
            raise RuntimeError("Scheduler must be started before adding custom jobs")
        
        if trigger == "interval":
            trigger_obj = IntervalTrigger(**trigger_args)
        elif trigger == "cron":
            trigger_obj = CronTrigger.from_crontab(trigger_args.get("cron"))
        else:
            raise ValueError(f"Unsupported trigger type: {trigger}")
        
        self.scheduler.add_job(
            func,
            trigger_obj,
            name=f"Custom: {func.__name__}",
        )
        logger.info(f"Added custom job: {func.__name__} with {trigger} trigger")
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get list of scheduled jobs.
        
        Returns:
            List of job info dicts with id, name, next_run_time
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger),
            })
        return jobs
    
    @property
    def running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


async def create_scheduler(
    cache_manager: CacheManager,
    config: Optional[Dict[str, Any]] = None,
    start: bool = True,
) -> CacheScheduler:
    """Create and optionally start a cache scheduler.
    
    Args:
        cache_manager: CacheManager instance
        config: Scheduler configuration
        start: Whether to start scheduler immediately
    
    Returns:
        CacheScheduler instance
    """
    scheduler = CacheScheduler(cache_manager, config)
    if start:
        scheduler.start()
    return scheduler
