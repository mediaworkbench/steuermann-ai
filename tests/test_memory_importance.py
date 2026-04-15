"""Tests for memory importance scoring and temporal decay."""

import pytest
from datetime import datetime, timedelta, timezone
import math

from universal_agentic_framework.memory.importance import (
    MemoryImportanceScorer,
    TemporalMemoryDecay,
    calculate_importance_score,
    rank_memories_by_importance,
)


class TestMemoryImportanceScorer:
    """Tests for MemoryImportanceScorer class."""
    
    def test_calculate_importance_basic(self):
        """Test basic importance calculation with minimal metadata."""
        scorer = MemoryImportanceScorer()
        
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
        }
        
        importance = scorer.calculate_importance(0.8, metadata)
        
        # Should be close to relevance score for new memory
        assert 0.3 < importance < 1.0  # 0.8 * recency * frequency(0.5 for new)
    
    def test_calculate_importance_with_recency(self):
        """Test that older memories have lower scores."""
        scorer = MemoryImportanceScorer(recency_decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        
        # Recent memory
        recent_metadata = {
            "created_at": (current_time - timedelta(days=1)).isoformat(),
            "access_count": 1,
        }
        
        # Old memory
        old_metadata = {
            "created_at": (current_time - timedelta(days=30)).isoformat(),
            "access_count": 1,
        }
        
        recent_score = scorer.calculate_importance(0.8, recent_metadata, current_time)
        old_score = scorer.calculate_importance(0.8, old_metadata, current_time)
        
        # Recent should have higher score
        assert recent_score > old_score
        assert old_score > 0.01  # Should not be zero
    
    def test_calculate_importance_with_frequency(self):
        """Test that frequently accessed memories rank higher."""
        scorer = MemoryImportanceScorer(frequency_weight=0.3)
        current_time = datetime.now(timezone.utc)
        
        # Low frequency
        low_freq_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 1,
        }
        
        # High frequency
        high_freq_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 50,
        }
        
        low_score = scorer.calculate_importance(0.8, low_freq_metadata, current_time)
        high_score = scorer.calculate_importance(0.8, high_freq_metadata, current_time)
        
        # High frequency should rank higher
        assert high_score > low_score
    
    def test_calculate_importance_with_user_feedback(self):
        """Test that user ratings affect importance."""
        scorer = MemoryImportanceScorer(feedback_weight=0.2)
        current_time = datetime.now(timezone.utc)
        
        # Low rating
        low_rating_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 1,
            "user_rating": 1,
        }
        
        # High rating
        high_rating_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 1,
            "user_rating": 5,
        }
        
        low_score = scorer.calculate_importance(0.8, low_rating_metadata, current_time)
        high_score = scorer.calculate_importance(0.8, high_rating_metadata, current_time)
        
        # High rating should rank higher
        assert high_score > low_score
    
    def test_recency_factor_exponential_decay(self):
        """Test that recency factor follows exponential decay."""
        scorer = MemoryImportanceScorer(recency_decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        
        # Calculate expected decay for 10 days
        days = 10
        metadata = {
            "created_at": (current_time - timedelta(days=days)).isoformat(),
        }
        
        factor = scorer._calculate_recency_factor(metadata, current_time)
        expected = math.exp(-0.1 * days)
        
        assert abs(factor - expected) < 0.01  # Within 1% tolerance
    
    def test_frequency_factor_logarithmic_scaling(self):
        """Test that frequency factor uses logarithmic scaling."""
        scorer = MemoryImportanceScorer()
        
        # New memory (0 accesses)
        new_metadata = {"access_count": 0}
        new_factor = scorer._calculate_frequency_factor(new_metadata)
        assert new_factor == 0.5  # New memories get exploration boost
        
        # Moderate frequency
        moderate_metadata = {"access_count": 10}
        moderate_factor = scorer._calculate_frequency_factor(moderate_metadata)
        
        # High frequency
        high_metadata = {"access_count": 100}
        high_factor = scorer._calculate_frequency_factor(high_metadata)
        
        # Should be logarithmic (not linear)
        # 100x more accesses should NOT give 100x higher factor
        assert high_factor < new_factor + 10 * (moderate_factor - new_factor)
        assert moderate_factor > new_factor
        assert high_factor > moderate_factor
    
    def test_feedback_factor_normalization(self):
        """Test that user ratings normalize to appropriate multipliers."""
        scorer = MemoryImportanceScorer()
        
        # 1 star = 0.5x
        low_metadata = {"user_rating": 1}
        low_factor = scorer._calculate_feedback_factor(low_metadata)
        assert abs(low_factor - 0.5) < 0.01
        
        # 3 stars = 1.0x (neutral)
        neutral_metadata = {"user_rating": 3}
        neutral_factor = scorer._calculate_feedback_factor(neutral_metadata)
        assert abs(neutral_factor - 1.0) < 0.01
        
        # 5 stars = 2.0x
        high_metadata = {"user_rating": 5}
        high_factor = scorer._calculate_feedback_factor(high_metadata)
        assert abs(high_factor - 2.0) < 0.01
        
        # No rating = 1.0x (neutral)
        no_rating_metadata = {}
        no_rating_factor = scorer._calculate_feedback_factor(no_rating_metadata)
        assert no_rating_factor == 1.0
    
    def test_apply_temporal_decay(self):
        """Test temporal decay application to memory list."""
        scorer = MemoryImportanceScorer(recency_decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        
        memories = [
            {
                "text": "Old memory",
                "score": 0.9,
                "metadata": {"created_at": (current_time - timedelta(days=30)).isoformat()},
            },
            {
                "text": "Recent memory",
                "score": 0.7,
                "metadata": {"created_at": (current_time - timedelta(days=1)).isoformat()},
            },
        ]
        
        decayed = scorer.apply_temporal_decay(memories, current_time)
        
        # Recent memory should rank higher despite lower original score
        assert decayed[0]["text"] == "Recent memory"
        assert decayed[1]["text"] == "Old memory"
        
        # Should have decay_factor and decayed_score
        assert "decay_factor" in decayed[0]
        assert "decayed_score" in decayed[0]
        assert decayed[0]["decay_factor"] > decayed[1]["decay_factor"]
    
    def test_rank_memories(self):
        """Test memory ranking by importance."""
        scorer = MemoryImportanceScorer()
        current_time = datetime.now(timezone.utc)
        
        search_results = [
            {
                "text": "Low relevance, high frequency",
                "score": 0.5,
                "metadata": {
                    "created_at": current_time.isoformat(),
                    "access_count": 50,
                },
            },
            {
                "text": "High relevance, new",
                "score": 0.9,
                "metadata": {
                    "created_at": current_time.isoformat(),
                    "access_count": 0,
                },
            },
            {
                "text": "Medium relevance, rated",
                "score": 0.7,
                "metadata": {
                    "created_at": current_time.isoformat(),
                    "access_count": 5,
                    "user_rating": 5,
                },
            },
        ]
        
        ranked = scorer.rank_memories(search_results, current_time, min_score=0.0)
        
        # Should have importance scores
        assert all("importance" in m for m in ranked)
        
        # Should be sorted by importance
        for i in range(len(ranked) - 1):
            assert ranked[i]["importance"] >= ranked[i + 1]["importance"]
    
    def test_rank_memories_filters_low_scores(self):
        """Test that ranking filters out low importance scores."""
        scorer = MemoryImportanceScorer()
        current_time = datetime.now(timezone.utc)
        
        search_results = [
            {
                "text": "High importance",
                "score": 0.8,
                "metadata": {"created_at": current_time.isoformat(), "access_count": 10},
            },
            {
                "text": "Very old, low importance",
                "score": 0.3,
                "metadata": {
                    "created_at": (current_time - timedelta(days=100)).isoformat(),
                    "access_count": 0,
                },
            },
        ]
        
        ranked = scorer.rank_memories(search_results, current_time, min_score=0.1)
        
        # Should filter out very low importance memories
        assert len(ranked) <= len(search_results)
        assert all(m["importance"] >= 0.1 for m in ranked)
    
    def test_update_access_metadata(self):
        """Test metadata updating on memory access."""
        scorer = MemoryImportanceScorer()
        
        # Initial metadata
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 5,
            "user_rating": 3,
        }
        
        updated = scorer.update_access_metadata(metadata)
        
        # Should increment access count
        assert updated["access_count"] == 6
        
        # Should update last_accessed
        assert "last_accessed" in updated
        
        # Should preserve other fields
        assert updated["user_rating"] == 3
        assert updated["created_at"] == metadata["created_at"]


class TestTemporalMemoryDecay:
    """Tests for TemporalMemoryDecay class."""
    
    def test_calculate_decay_factor(self):
        """Test decay factor calculation."""
        decay = TemporalMemoryDecay(decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        created_at = current_time - timedelta(days=10)
        
        factor = decay.calculate_decay_factor(created_at, current_time)
        expected = math.exp(-0.1 * 10)
        
        assert abs(factor - expected) < 0.01
    
    def test_apply_decay(self):
        """Test decay application to score."""
        decay = TemporalMemoryDecay(decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        created_at = current_time - timedelta(days=10)
        
        original_score = 0.8
        decayed_score = decay.apply_decay(original_score, created_at, current_time)
        
        # Should be lower than original
        assert decayed_score < original_score
        assert decayed_score > 0
        
        # Should equal original * decay_factor
        factor = decay.calculate_decay_factor(created_at, current_time)
        expected = original_score * factor
        assert abs(decayed_score - expected) < 0.01
    
    def test_decay_factor_zero_days(self):
        """Test that same-day memories have no decay."""
        decay = TemporalMemoryDecay(decay_rate=0.1)
        current_time = datetime.now(timezone.utc)
        
        factor = decay.calculate_decay_factor(current_time, current_time)
        
        # Should be 1.0 (no decay)
        assert abs(factor - 1.0) < 0.01


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_calculate_importance_score_function(self):
        """Test convenience function for importance calculation."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 5,
        }
        
        score = calculate_importance_score(0.8, metadata)
        
        assert 0.0 < score < 2.0
        assert isinstance(score, float)
    
    def test_calculate_importance_score_with_config(self):
        """Test convenience function with custom config."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 5,
        }
        
        config = {"recency_decay_rate": 0.2, "frequency_weight": 0.5}
        score = calculate_importance_score(0.8, metadata, config)
        
        assert 0.0 < score < 2.0
    
    def test_rank_memories_by_importance_function(self):
        """Test convenience function for ranking."""
        current_time = datetime.now(timezone.utc)
        search_results = [
            {
                "text": "Memory 1",
                "score": 0.7,
                "metadata": {"created_at": current_time.isoformat(), "access_count": 10},
            },
            {
                "text": "Memory 2",
                "score": 0.9,
                "metadata": {"created_at": current_time.isoformat(), "access_count": 1},
            },
        ]
        
        ranked = rank_memories_by_importance(search_results)
        
        assert len(ranked) == 2
        assert all("importance" in m for m in ranked)
        assert ranked[0]["importance"] >= ranked[1]["importance"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_metadata_fields(self):
        """Test handling of missing metadata fields."""
        scorer = MemoryImportanceScorer()
        
        # Empty metadata
        empty_metadata = {}
        score = scorer.calculate_importance(0.8, empty_metadata)
        assert score > 0  # Should not crash
        
        # Partial metadata
        partial_metadata = {"access_count": 5}  # Missing created_at
        score = scorer.calculate_importance(0.8, partial_metadata)
        assert score > 0
    
    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp formats."""
        scorer = MemoryImportanceScorer()
        
        metadata = {
            "created_at": "invalid-timestamp",
            "access_count": 1,
        }
        
        # Should not crash, should use default (no decay)
        score = scorer.calculate_importance(0.8, metadata)
        assert score > 0
    
    def test_extreme_relevance_scores(self):
        """Test handling of extreme relevance scores."""
        scorer = MemoryImportanceScorer()
        metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "access_count": 1}
        
        # Negative score (should be clamped to 0)
        score_neg = scorer.calculate_importance(-0.5, metadata)
        assert score_neg >= 0
        
        # Score > 1.0 (should be clamped)
        score_high = scorer.calculate_importance(2.0, metadata)
        assert score_high >= 0  # Valid (frequency/feedback can push above 1.0)
    
    def test_extreme_access_counts(self):
        """Test handling of very large access counts."""
        scorer = MemoryImportanceScorer()
        current_time = datetime.now(timezone.utc)
        
        metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 10000,  # Very high
        }
        
        score = scorer.calculate_importance(0.8, metadata, current_time)
        
        # Should use logarithmic scaling, not go to infinity
        assert score < 10.0  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
