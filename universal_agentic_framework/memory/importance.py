"""Memory importance scoring and temporal decay.

Implements intelligent memory ranking based on:
- Relevance: Semantic similarity to current query
- Recency: Time since last access
- Frequency: Number of times accessed
- User feedback: Explicit ratings (optional)

Temporal decay ensures older memories become less prominent over time.
"""

import math
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class MemoryImportanceScorer:
    """Calculates importance scores for memories to prioritize retrieval.
    
    Formula: importance = relevance * recency_factor * frequency_factor * feedback_factor
    
    Where:
    - relevance: Cosine similarity from Qdrant search (0.0-1.0)
    - recency_factor: e^(-λ * days_since_access)
    - frequency_factor: log(1 + access_count) / log(1 + max_frequency)
    - feedback_factor: user rating (1.0 default, 0.5-2.0 range)
    """
    
    def __init__(
        self,
        recency_decay_rate: float = 0.1,  # λ, decay per day
        frequency_weight: float = 0.3,
        feedback_weight: float = 0.2,
        min_importance: float = 0.01,  # Minimum score to keep memory
    ):
        """Initialize scorer with configurable parameters.
        
        Args:
            recency_decay_rate: Exponential decay rate (default: 0.1/day)
            frequency_weight: Weight for frequency factor (0.0-1.0)
            feedback_weight: Weight for user feedback (0.0-1.0)
            min_importance: Minimum score threshold for retrieval
        """
        self.recency_decay_rate = recency_decay_rate
        self.frequency_weight = frequency_weight
        self.feedback_weight = feedback_weight
        self.min_importance = min_importance
    
    def calculate_importance(
        self,
        relevance_score: float,
        metadata: Dict[str, Any],
        current_time: Optional[datetime] = None,
    ) -> float:
        """Calculate importance score for a memory.
        
        Args:
            relevance_score: Semantic similarity from Qdrant (0.0-1.0)
            metadata: Memory metadata containing:
                - created_at: ISO timestamp
                - last_accessed: ISO timestamp (optional)
                - access_count: Number of retrievals
                - user_rating: Explicit rating 1-5 (optional)
                - related_memory_ids: List of linked memory IDs (optional)
            current_time: Current timestamp (for testing, defaults to now)
        
        Returns:
            Importance score (0.0-1.0+, can exceed 1.0 with high frequency/feedback)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Base relevance from semantic search
        base_score = max(0.0, min(1.0, relevance_score))
        
        # Recency factor (exponential decay)
        recency_factor = self._calculate_recency_factor(metadata, current_time)
        
        # Frequency factor (logarithmic scaling)
        frequency_factor = self._calculate_frequency_factor(metadata)
        
        # User feedback factor (explicit ratings)
        feedback_factor = self._calculate_feedback_factor(metadata)
        
        # Combined importance with weights
        importance = base_score * recency_factor
        importance *= (1.0 + self.frequency_weight * (frequency_factor - 1.0))
        importance *= (1.0 + self.feedback_weight * (feedback_factor - 1.0))
        
        return max(0.0, importance)
    
    def _calculate_recency_factor(
        self,
        metadata: Dict[str, Any],
        current_time: datetime,
    ) -> float:
        """Calculate recency factor using exponential decay.
        
        Formula: e^(-λ * days_since_access)
        """
        # Use last_accessed if available, otherwise created_at
        timestamp_str = metadata.get("last_accessed") or metadata.get("created_at")
        
        if not timestamp_str:
            # No timestamp, assume recent
            return 1.0
        
        try:
            if isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
            else:
                # Parse ISO format timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            # Calculate days since access
            time_diff = current_time - timestamp
            days_since_access = time_diff.total_seconds() / 86400.0  # Convert to days
            
            # Exponential decay
            decay_factor = math.exp(-self.recency_decay_rate * days_since_access)
            
            return max(0.01, decay_factor)  # Minimum 1% to avoid zeroing out old memories
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse timestamp: {timestamp_str}, error: {e}")
            return 1.0  # Default to no decay on parse error
    
    def _calculate_frequency_factor(self, metadata: Dict[str, Any]) -> float:
        """Calculate frequency factor using logarithmic scaling.
        
        Formula: log(1 + access_count) / log(1 + max_count_assumption)
        
        Logarithmic scaling prevents very frequent memories from dominating.
        """
        access_count = metadata.get("access_count", 0)
        
        if access_count <= 0:
            return 0.5  # New memories get 50% frequency boost to encourage exploration
        
        # Assume max frequency of 100 accesses for normalization
        max_count = 100.0
        
        # Logarithmic scaling
        normalized_frequency = math.log(1 + access_count) / math.log(1 + max_count)
        
        # Scale to 0.5-2.0 range (new to very frequent)
        return 0.5 + 1.5 * normalized_frequency
    
    def _calculate_feedback_factor(self, metadata: Dict[str, Any]) -> float:
        """Calculate feedback factor from explicit user ratings.
        
        User ratings: 1-5 stars → 0.5-2.0 multiplier
        """
        user_rating = metadata.get("user_rating")
        
        if user_rating is None:
            return 1.0  # Neutral (no feedback)
        
        # Normalize 1-5 rating to 0.5-2.0 multiplier
        # 1 star = 0.5x, 3 stars = 1.0x, 5 stars = 2.0x
        rating_normalized = max(1.0, min(5.0, float(user_rating)))
        
        # Piecewise linear: steeper above neutral (3 stars)
        if rating_normalized <= 3.0:
            feedback_multiplier = 0.5 + (rating_normalized - 1.0) * 0.25
        else:
            feedback_multiplier = 1.0 + (rating_normalized - 3.0) * 0.5
        
        return feedback_multiplier
    
    def apply_temporal_decay(
        self,
        memories: List[Dict[str, Any]],
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Apply temporal decay to memory scores.
        
        Args:
            memories: List of memory dicts with 'score' and 'metadata' keys
            current_time: Current timestamp (defaults to now)
        
        Returns:
            List of memories with updated scores, sorted by decayed score
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        for memory in memories:
            original_score = memory.get("score", 0.0)
            metadata = memory.get("metadata", {})
            
            # Apply decay to the original relevance score
            recency_factor = self._calculate_recency_factor(metadata, current_time)
            memory["decayed_score"] = original_score * recency_factor
            memory["decay_factor"] = recency_factor
        
        # Sort by decayed score
        memories.sort(key=lambda m: m.get("decayed_score", 0.0), reverse=True)
        
        return memories
    
    def rank_memories(
        self,
        search_results: List[Dict[str, Any]],
        current_time: Optional[datetime] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rank memories by importance score.
        
        Args:
            search_results: List of dicts with 'score' (relevance) and 'metadata' keys
            current_time: Current timestamp (defaults to now)
            min_score: Minimum importance threshold (defaults to self.min_importance)
        
        Returns:
            List of memories ranked by importance, with 'importance' key added
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        if min_score is None:
            min_score = self.min_importance
        
        ranked_memories = []
        
        for result in search_results:
            relevance = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            importance = self.calculate_importance(relevance, metadata, current_time)
            
            if importance >= min_score:
                result["importance"] = importance
                ranked_memories.append(result)
        
        # Sort by importance (highest first)
        ranked_memories.sort(key=lambda m: m["importance"], reverse=True)
        
        return ranked_memories
    
    def update_access_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update metadata when a memory is accessed.
        
        Args:
            metadata: Current memory metadata
        
        Returns:
            Updated metadata with access_count incremented and last_accessed updated
        """
        updated = metadata.copy()
        
        # Increment access count
        updated["access_count"] = metadata.get("access_count", 0) + 1
        
        # Update last accessed timestamp
        updated["last_accessed"] = datetime.now(timezone.utc).isoformat()
        
        return updated


class TemporalMemoryDecay:
    """Standalone temporal decay calculator for memory aging."""
    
    def __init__(self, decay_rate: float = 0.1):
        """Initialize with decay rate (per day)."""
        self.decay_rate = decay_rate
    
    def calculate_decay_factor(
        self,
        created_at: datetime,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Calculate exponential decay factor.
        
        Args:
            created_at: Memory creation timestamp
            current_time: Current timestamp (defaults to now)
        
        Returns:
            Decay factor (0.0-1.0)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        days_elapsed = (current_time - created_at).total_seconds() / 86400.0
        return math.exp(-self.decay_rate * days_elapsed)
    
    def apply_decay(
        self,
        score: float,
        created_at: datetime,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Apply temporal decay to a score.
        
        Args:
            score: Original importance score
            created_at: Memory creation timestamp
            current_time: Current timestamp (defaults to now)
        
        Returns:
            Score with decay applied
        """
        decay_factor = self.calculate_decay_factor(created_at, current_time)
        return score * decay_factor


# Utility functions for easy integration

def calculate_importance_score(
    relevance: float,
    metadata: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Convenience function to calculate importance score.
    
    Args:
        relevance: Semantic similarity score (0.0-1.0)
        metadata: Memory metadata
        config: Scorer configuration (optional)
    
    Returns:
        Importance score
    """
    if config is None:
        config = {}
    
    scorer = MemoryImportanceScorer(**config)
    return scorer.calculate_importance(relevance, metadata)


def rank_memories_by_importance(
    search_results: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to rank memories.
    
    Args:
        search_results: Qdrant search results
        config: Scorer configuration (optional)
    
    Returns:
        Ranked memories with importance scores
    """
    if config is None:
        config = {}
    
    scorer = MemoryImportanceScorer(**config)
    return scorer.rank_memories(search_results)
