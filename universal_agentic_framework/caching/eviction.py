"""Cache eviction policies for managing cache size.

Implements various eviction strategies:
- LRU (Least Recently Used): Evicts oldest accessed items
- LFU (Least Frequently Used): Evicts least frequently accessed items  
- FIFO (First In First Out): Evicts oldest inserted items
- TTL (Time To Live): Evicts expired items first
"""

import time
import logging
from typing import Any, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for eviction policies."""
    key: str
    value: Any
    expiry: Optional[float]
    inserted_at: float
    last_accessed: float
    access_count: int


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """Called when an entry is accessed."""
        pass
    
    @abstractmethod
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """Called when an entry is added."""
        pass
    
    @abstractmethod
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select keys to evict.
        
        Args:
            cache: Current cache state
            count: Number of entries to evict
            
        Returns:
            List of keys to evict
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get policy name."""
        pass


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """Update last accessed time."""
        now = time.time()
        # Keep access timestamps strictly increasing per entry even when
        # clocks are coarse/frozen, so recently accessed entries stay newer.
        entry.last_accessed = max(now, entry.last_accessed + 1e-9)
    
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """Record insertion."""
        pass  # Nothing special needed
    
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select least recently used entries."""
        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: x[1].last_accessed
        )
        return [key for key, _ in sorted_entries[:count]]
    
    def get_name(self) -> str:
        return "LRU"


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """Increment access count."""
        entry.access_count += 1
        now = time.time()
        # Mirror LRU monotonic update behavior for deterministic LFU tie-breaks.
        entry.last_accessed = max(now, entry.last_accessed + 1e-9)
    
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """Initialize access count."""
        pass  # Access count already initialized to 0
    
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select least frequently used entries."""
        # Sort by access_count (lowest first), then by last_accessed
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        return [key for key, _ in sorted_entries[:count]]
    
    def get_name(self) -> str:
        return "LFU"


class FIFOPolicy(EvictionPolicy):
    """First In First Out eviction policy."""
    
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """No action needed for FIFO on get."""
        pass
    
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """Record insertion time."""
        pass  # inserted_at already set
    
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select oldest inserted entries."""
        # Sort by inserted_at (oldest first)
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: x[1].inserted_at
        )
        return [key for key, _ in sorted_entries[:count]]
    
    def get_name(self) -> str:
        return "FIFO"


class TTLPolicy(EvictionPolicy):
    """Time To Live priority eviction policy.
    
    Evicts expired entries first, then falls back to LRU.
    """
    
    def __init__(self):
        self.lru_fallback = LRUPolicy()
    
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """Update last accessed time."""
        self.lru_fallback.on_get(key, entry)
    
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """Record insertion."""
        pass
    
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select expired entries first, then least recently used."""
        current_time = time.time()
        victims = []
        
        # First, collect expired entries
        for key, entry in cache.items():
            if entry.expiry is not None and current_time >= entry.expiry:
                victims.append(key)
                if len(victims) >= count:
                    return victims[:count]
        
        # If we need more victims, use LRU
        if len(victims) < count:
            remaining = count - len(victims)
            lru_victims = self.lru_fallback.select_victims(
                {k: v for k, v in cache.items() if k not in victims},
                remaining
            )
            victims.extend(lru_victims)
        
        return victims[:count]
    
    def get_name(self) -> str:
        return "TTL"


class RandomPolicy(EvictionPolicy):
    """Random eviction policy (for testing/comparison)."""
    
    def on_get(self, key: str, entry: CacheEntry) -> None:
        """No action needed."""
        pass
    
    def on_set(self, key: str, entry: CacheEntry) -> None:
        """No action needed."""
        pass
    
    def select_victims(self, cache: Dict[str, CacheEntry], count: int) -> List[str]:
        """Select random entries."""
        import random
        keys = list(cache.keys())
        return random.sample(keys, min(count, len(keys)))
    
    def get_name(self) -> str:
        return "Random"


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Create eviction policy by name.
    
    Args:
        policy_name: One of "LRU", "LFU", "FIFO", "TTL", "Random"
        
    Returns:
        EvictionPolicy instance
        
    Raises:
        ValueError: If policy name is unknown
    """
    policies = {
        "LRU": LRUPolicy,
        "LFU": LFUPolicy,
        "FIFO": FIFOPolicy,
        "TTL": TTLPolicy,
        "Random": RandomPolicy,
    }
    
    policy_class = policies.get(policy_name.upper())
    if not policy_class:
        raise ValueError(
            f"Unknown eviction policy: {policy_name}. "
            f"Valid options: {', '.join(policies.keys())}"
        )
    
    logger.info(f"Created eviction policy: {policy_name}")
    return policy_class()
