"""Cross-session memory linking for knowledge graph construction.

CRITICAL CAVEAT (Phase 3 planned fix):
This module currently maintains a NONPERSISTENT co-occurrence knowledge graph.
On process restart, all links are lost. This is a known limitation being addressed in Phase 3
with hybrid PostgreSQL + Qdrant persistence.

CURRENT ARCHITECTURE (IN-MEMORY ONLY):
This module tracks which memories are frequently retrieved together across sessions to build
a knowledge graph of related memories. This enables context expansion: when retrieving memory A,
also fetch related memories B, C.

PLANNED ARCHITECTURE (PHASE 3 - HYBRID PERSISTENCE):
- Durable source: PostgreSQL co_occurrence_edges table
  - Schema: (user_id, memory_id, related_memory_id, session_evidence, decayed_strength, created_at, updated_at)
  - Indexes: (user_id, memory_id) and (user_id, updated_at)
- Acceleration layer: top related edges projected into Qdrant memory metadata
- Read path: reconstruct from durable edges + projection with precedence handling
- Write path: transactional upsert after retrieval; async projection refresh

CURRENT BEHAVIOR:
- In-memory storage: defaultdict keyed by memory_id → related_id → [(timestamp, session_id), ...]
- Decay model: sliding window (only count within N days)
- Strength calculation: normalized 0.0-1.0 based on co-occurrence frequency
- Pruning: manual via prune_old_co_occurrences() or via init parameter

OPERATIONS:
- record_co_occurrence(memory_ids, session_id): Called during retrieval to track co-occurrence
- get_related_memories(memory_id, top_k): Called to expand context during retrieval
- get_metadata_format(): Returns related memories formatted for metadata injection

SOURCE OF TRUTH OWNERSHIP:
- Current: MemoryCoOccurrenceTracker instance (session-scoped)
- Planned: PostgreSQL durable + in-memory cache for performance
- Invalidation: Planned to clear cache on co_occurrence_edges table update

VALIDATION & TESTING:
- See checkpoint #8 in plan: PostgreSQL schema validation gate
- Checkpoint #7 verifies digest chain end-to-end (related piece of knowledge graph)
- Integration tests in test_memory_linking.py

See: docs/technical_architecture.md (Memory Architecture) for full context
"""

from __future__ import annotations

import structlog
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = structlog.get_logger(__name__)

# Import for feature flag kill switch
try:
    from universal_agentic_framework.config import load_features_config
except Exception:
    # Fallback: feature flag unavailable, always enable co-occurrence tracking
    load_features_config = None


class CoOccurrenceDurableStore(Protocol):
    def record_co_occurrence_pair(
        self,
        *,
        user_id: str,
        memory_id: str,
        related_memory_id: str,
        session_id: str,
        event_time: Optional[datetime] = None,
    ) -> None:
        ...

    def get_related_edges(
        self,
        *,
        user_id: str,
        memory_id: str,
        top_k: int,
        min_strength: float,
        updated_since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def prune_old_edges(self, *, cutoff_time: datetime) -> int:
        ...


def _try_build_default_durable_store() -> Optional[CoOccurrenceDurableStore]:
    try:
        from backend.db import CoOccurrenceEdgeStore, init_db_pool

        return CoOccurrenceEdgeStore(init_db_pool())
    except Exception as exc:
        logger.warning("co_occurrence_durable_store_unavailable", error=str(exc))
        return None


class MemoryCoOccurrenceTracker:
    """Track memories retrieved together to build knowledge graph.
    
    This class maintains co-occurrence statistics for memories retrieved in
    the same session. It builds a knowledge graph where:
    - Nodes: Memory IDs
    - Edges: Co-occurrence strength (normalized 0.0-1.0)
    
    Co-occurrence links decay over time using a sliding window approach.
    """
    
    def __init__(
        self,
        *,
        decay_window_days: int = 30,
        min_co_occurrence_strength: float = 0.1,
        max_related_memories: int = 10,
        durable_store: Optional[CoOccurrenceDurableStore] = None,
        enable_durable_store: bool = False,
    ):
        """Initialize co-occurrence tracker.
        
        Args:
            decay_window_days: Only count co-occurrences within this window
            min_co_occurrence_strength: Minimum strength to store (0.0-1.0)
            max_related_memories: Maximum related memories to store per memory
        """
        self.decay_window_days = decay_window_days
        self.min_co_occurrence_strength = min_co_occurrence_strength
        self.max_related_memories = max_related_memories
        self._durable_store = durable_store
        if self._durable_store is None and enable_durable_store:
            self._durable_store = _try_build_default_durable_store()
        
        # In-memory storage: {memory_id: {related_id: [(timestamp, session_id), ...]}}
        # In production, this would be persisted to PostgreSQL or Qdrant metadata
        self._co_occurrences: Dict[str, Dict[str, List[Tuple[datetime, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        logger.info(
            "initialized_co_occurrence_tracker",
            decay_window_days=decay_window_days,
            min_strength=min_co_occurrence_strength,
            max_related=max_related_memories,
            durable_store=bool(self._durable_store),
        )
    
    def record_co_occurrence(
        self,
        memory_ids: List[str],
        session_id: str,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record that memories were retrieved together in a session.
        
        Args:
            memory_ids: List of memory IDs retrieved together
            session_id: Session identifier
            timestamp: When this co-occurrence happened (defaults to now)
            
        KILL SWITCH: Set features.memory_co_occurrence_enabled=false to disable tracking.
        """
        # EMERGENCY ROLLBACK: Check memory_co_occurrence_enabled feature flag
        if load_features_config is not None:
            try:
                features = load_features_config()
                if not features.memory_co_occurrence_enabled:
                    logger.info("co_occurrence_tracking_disabled_by_feature_flag")
                    return
            except Exception as e:
                logger.warning("failed_to_load_features_for_kill_switch", error=str(e))
                # Continue normally if features config unavailable
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if len(memory_ids) < 2:
            # Need at least 2 memories to have co-occurrence
            return
        
        # Record pairwise co-occurrence
        for i, mem_id_a in enumerate(memory_ids):
            for mem_id_b in memory_ids[i + 1:]:
                # Bidirectional: A→B and B→A
                self._co_occurrences[mem_id_a][mem_id_b].append((timestamp, session_id))
                self._co_occurrences[mem_id_b][mem_id_a].append((timestamp, session_id))

                if self._durable_store and user_id:
                    try:
                        self._durable_store.record_co_occurrence_pair(
                            user_id=user_id,
                            memory_id=mem_id_a,
                            related_memory_id=mem_id_b,
                            session_id=session_id,
                            event_time=timestamp,
                        )
                        self._durable_store.record_co_occurrence_pair(
                            user_id=user_id,
                            memory_id=mem_id_b,
                            related_memory_id=mem_id_a,
                            session_id=session_id,
                            event_time=timestamp,
                        )
                    except Exception as exc:
                        logger.warning(
                            "co_occurrence_durable_write_failed",
                            user_id=user_id,
                            memory_id=mem_id_a,
                            related_memory_id=mem_id_b,
                            error=str(exc),
                        )
        
        logger.debug(
            "recorded_co_occurrence",
            memory_count=len(memory_ids),
            pairs=len(memory_ids) * (len(memory_ids) - 1) // 2,
            session_id=session_id,
        )
    
    def get_related_memories(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get memories frequently retrieved with this memory.
        
        Args:
            memory_id: Memory to find related memories for
            top_k: Maximum number of related memories to return
            current_time: Current timestamp (defaults to now)
        
        Returns:
            List of dicts with keys: 'memory_id', 'strength', 'co_occurrence_count'
            Sorted by strength (descending)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Prefer durable edges when available (restart-safe source of truth).
        if self._durable_store and user_id:
            try:
                cutoff_time = current_time - timedelta(days=self.decay_window_days)
                durable_related = self._durable_store.get_related_edges(
                    user_id=user_id,
                    memory_id=memory_id,
                    top_k=top_k,
                    min_strength=self.min_co_occurrence_strength,
                    updated_since=cutoff_time,
                )
                if durable_related:
                    return durable_related
            except Exception as exc:
                logger.warning(
                    "co_occurrence_durable_read_failed",
                    user_id=user_id,
                    memory_id=memory_id,
                    error=str(exc),
                )
        
        if memory_id not in self._co_occurrences:
            return []
        
        # Calculate co-occurrence strength for each related memory
        related_with_strength = []
        cutoff_time = current_time - timedelta(days=self.decay_window_days)
        
        for related_id, occurrences in self._co_occurrences[memory_id].items():
            # Filter by decay window
            recent_occurrences = [
                (ts, sid) for ts, sid in occurrences if ts >= cutoff_time
            ]
            
            if not recent_occurrences:
                continue
            
            # Calculate strength (normalized by total co-occurrences)
            # Simple approach: count / max_count in window
            count = len(recent_occurrences)
            
            # Normalize: assume max realistic co-occurrence is 10 times in window
            # This prevents single memory from dominating all others
            strength = min(1.0, count / 10.0)
            
            if strength >= self.min_co_occurrence_strength:
                related_with_strength.append({
                    "memory_id": related_id,
                    "strength": strength,
                    "co_occurrence_count": count,
                })
        
        # Sort by strength (descending)
        related_with_strength.sort(key=lambda x: x["strength"], reverse=True)
        
        logger.debug(
            "retrieved_related_memories",
            memory_id=memory_id,
            related_count=len(related_with_strength),
            top_k=top_k,
        )
        
        return related_with_strength[:top_k]
    
    def get_metadata_format(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get related memories in format suitable for Qdrant metadata.
        
        Args:
            memory_id: Memory to get related memories for
            current_time: Current timestamp (defaults to now)
        
        Returns:
            List of dicts with keys: 'id', 'strength'
            Limited to max_related_memories, sorted by strength
        """
        related = self.get_related_memories(
            memory_id,
            user_id=user_id,
            top_k=self.max_related_memories,
            current_time=current_time,
        )
        
        # Convert to metadata format (simplified)
        return [
            {"id": rel["memory_id"], "strength": rel["strength"]}
            for rel in related
        ]
    
    def prune_old_co_occurrences(
        self,
        current_time: Optional[datetime] = None,
    ) -> int:
        """Remove co-occurrences outside the decay window.
        
        Args:
            current_time: Current timestamp (defaults to now)
        
        Returns:
            Number of co-occurrence records pruned
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        cutoff_time = current_time - timedelta(days=self.decay_window_days)
        pruned_count = 0
        
        # Prune old occurrences
        for memory_id in list(self._co_occurrences.keys()):
            for related_id in list(self._co_occurrences[memory_id].keys()):
                # Filter occurrences
                occurrences = self._co_occurrences[memory_id][related_id]
                recent = [
                    (ts, sid) for ts, sid in occurrences if ts >= cutoff_time
                ]
                
                pruned_count += len(occurrences) - len(recent)
                
                if recent:
                    self._co_occurrences[memory_id][related_id] = recent
                else:
                    # No recent co-occurrences, remove this edge
                    del self._co_occurrences[memory_id][related_id]
            
            # Remove memory if no related memories left
            if not self._co_occurrences[memory_id]:
                del self._co_occurrences[memory_id]
        
        logger.info(
            "pruned_old_co_occurrences",
            pruned_count=pruned_count,
            cutoff_time=cutoff_time.isoformat(),
        )

        if self._durable_store:
            try:
                pruned_count += self._durable_store.prune_old_edges(cutoff_time=cutoff_time)
            except Exception as exc:
                logger.warning("co_occurrence_durable_prune_failed", error=str(exc))
        
        return pruned_count
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """Get statistics about the co-occurrence graph.
        
        Returns:
            Dict with keys: 'node_count', 'edge_count', 'avg_edges_per_node'
        """
        node_count = len(self._co_occurrences)
        edge_count = sum(
            len(related) for related in self._co_occurrences.values()
        )
        avg_edges = edge_count / node_count if node_count > 0 else 0.0
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_edges_per_node": round(avg_edges, 2),
        }
    
    def clear(self) -> None:
        """Clear all co-occurrence data."""
        self._co_occurrences.clear()
        logger.info("cleared_co_occurrence_data")


# Utility function for convenience
def extract_memory_ids(memories: List[Dict[str, any]]) -> List[str]:
    """Extract memory IDs from memory records.
    
    Args:
        memories: List of memory dicts (should have 'id' or 'memory_id' key)
    
    Returns:
        List of memory ID strings
    """
    ids = []
    for memory in memories:
        # Try different key names
        mem_id = memory.get("id") or memory.get("memory_id") or memory.get("text")
        if mem_id:
            # For memories without explicit ID, use text hash as ID
            if mem_id == memory.get("text"):
                mem_id = f"mem_{hash(mem_id) % 10**8}"
            ids.append(str(mem_id))
    return ids
