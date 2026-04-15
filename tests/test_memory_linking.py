"""Tests for cross-session memory linking."""

import pytest
from datetime import datetime, timedelta, timezone

from universal_agentic_framework.memory.linking import (
    MemoryCoOccurrenceTracker,
    extract_memory_ids,
)


class TestMemoryCoOccurrenceTracker:
    """Tests for MemoryCoOccurrenceTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization with default parameters."""
        tracker = MemoryCoOccurrenceTracker()
        
        assert tracker.decay_window_days == 30
        assert tracker.min_co_occurrence_strength == 0.1
        assert tracker.max_related_memories == 10
    
    def test_initialization_custom_params(self):
        """Test tracker initialization with custom parameters."""
        tracker = MemoryCoOccurrenceTracker(
            decay_window_days=60,
            min_co_occurrence_strength=0.2,
            max_related_memories=5,
        )
        
        assert tracker.decay_window_days == 60
        assert tracker.min_co_occurrence_strength == 0.2
        assert tracker.max_related_memories == 5
    
    def test_record_co_occurrence_two_memories(self):
        """Test recording co-occurrence for two memories."""
        tracker = MemoryCoOccurrenceTracker()
        
        tracker.record_co_occurrence(
            memory_ids=["mem_1", "mem_2"],
            session_id="session_1",
        )
        
        # Check bidirectional recording
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 1
        assert related[0]["memory_id"] == "mem_2"
        
        related = tracker.get_related_memories("mem_2")
        assert len(related) == 1
        assert related[0]["memory_id"] == "mem_1"
    
    def test_record_co_occurrence_multiple_memories(self):
        """Test recording co-occurrence for multiple memories."""
        tracker = MemoryCoOccurrenceTracker()
        
        tracker.record_co_occurrence(
            memory_ids=["mem_1", "mem_2", "mem_3"],
            session_id="session_1",
        )
        
        # mem_1 should be related to mem_2 and mem_3
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 2
        related_ids = {r["memory_id"] for r in related}
        assert related_ids == {"mem_2", "mem_3"}
        
        # mem_2 should be related to mem_1 and mem_3
        related = tracker.get_related_memories("mem_2")
        assert len(related) == 2
        related_ids = {r["memory_id"] for r in related}
        assert related_ids == {"mem_1", "mem_3"}
    
    def test_record_co_occurrence_single_memory(self):
        """Test that single memory produces no co-occurrence."""
        tracker = MemoryCoOccurrenceTracker()
        
        tracker.record_co_occurrence(
            memory_ids=["mem_1"],
            session_id="session_1",
        )
        
        # No related memories (need at least 2)
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 0
    
    def test_co_occurrence_strength_increases_with_frequency(self):
        """Test that strength increases with repeated co-occurrence."""
        tracker = MemoryCoOccurrenceTracker()
        
        # Record co-occurrence once
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        related = tracker.get_related_memories("mem_1")
        strength_1 = related[0]["strength"]
        
        # Record co-occurrence again
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_2")
        related = tracker.get_related_memories("mem_1")
        strength_2 = related[0]["strength"]
        
        assert strength_2 > strength_1
        assert related[0]["co_occurrence_count"] == 2
    
    def test_get_related_memories_sorted_by_strength(self):
        """Test that related memories are sorted by strength."""
        tracker = MemoryCoOccurrenceTracker()
        
        # mem_1 co-occurs with mem_2 once
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        
        # mem_1 co-occurs with mem_3 three times
        tracker.record_co_occurrence(["mem_1", "mem_3"], "session_2")
        tracker.record_co_occurrence(["mem_1", "mem_3"], "session_3")
        tracker.record_co_occurrence(["mem_1", "mem_3"], "session_4")
        
        related = tracker.get_related_memories("mem_1")
        
        # mem_3 should be first (higher strength)
        assert related[0]["memory_id"] == "mem_3"
        assert related[1]["memory_id"] == "mem_2"
        assert related[0]["strength"] > related[1]["strength"]
    
    def test_get_related_memories_top_k_limit(self):
        """Test that top_k limits the number of results."""
        tracker = MemoryCoOccurrenceTracker()
        
        # Create many related memories
        for i in range(10):
            tracker.record_co_occurrence(["mem_1", f"mem_{i}"], f"session_{i}")
        
        related = tracker.get_related_memories("mem_1", top_k=3)
        assert len(related) == 3
    
    def test_decay_window_filters_old_co_occurrences(self):
        """Test that old co-occurrences outside decay window are ignored."""
        tracker = MemoryCoOccurrenceTracker(decay_window_days=7)
        current_time = datetime.now(timezone.utc)
        
        # Record co-occurrence 10 days ago (outside window)
        old_time = current_time - timedelta(days=10)
        tracker.record_co_occurrence(
            ["mem_1", "mem_2"],
            "session_1",
            timestamp=old_time,
        )
        
        # Record co-occurrence 3 days ago (inside window)
        recent_time = current_time - timedelta(days=3)
        tracker.record_co_occurrence(
            ["mem_1", "mem_3"],
            "session_2",
            timestamp=recent_time,
        )
        
        # Only recent co-occurrence should be returned
        related = tracker.get_related_memories("mem_1", current_time=current_time)
        assert len(related) == 1
        assert related[0]["memory_id"] == "mem_3"
    
    def test_min_co_occurrence_strength_filters_weak_links(self):
        """Test that weak co-occurrences below min_strength are filtered."""
        tracker = MemoryCoOccurrenceTracker(min_co_occurrence_strength=0.5)
        
        # Record co-occurrence once (strength will be 0.1)
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        
        # Should be filtered out (0.1 < 0.5)
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 0
        
        # Record multiple times to increase strength
        for i in range(5):
            tracker.record_co_occurrence(["mem_1", "mem_2"], f"session_{i}")
        
        # Now should pass filter (strength ~0.6)
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 1
        assert related[0]["strength"] >= 0.5
    
    def test_get_metadata_format(self):
        """Test metadata format for Qdrant integration."""
        tracker = MemoryCoOccurrenceTracker()
        
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        tracker.record_co_occurrence(["mem_1", "mem_3"], "session_2")
        
        metadata = tracker.get_metadata_format("mem_1")
        
        assert isinstance(metadata, list)
        assert len(metadata) == 2
        assert all("id" in item and "strength" in item for item in metadata)
        assert all(0.0 <= item["strength"] <= 1.0 for item in metadata)
    
    def test_get_metadata_format_respects_max_related_memories(self):
        """Test that metadata format respects max_related_memories limit."""
        tracker = MemoryCoOccurrenceTracker(max_related_memories=3)
        
        # Create 5 related memories
        for i in range(5):
            tracker.record_co_occurrence(["mem_1", f"mem_{i}"], f"session_{i}")
        
        metadata = tracker.get_metadata_format("mem_1")
        
        # Should only return top 3
        assert len(metadata) <= 3
    
    def test_prune_old_co_occurrences(self):
        """Test pruning of old co-occurrences."""
        tracker = MemoryCoOccurrenceTracker(decay_window_days=7)
        current_time = datetime.now(timezone.utc)
        
        # Record old co-occurrence
        old_time = current_time - timedelta(days=10)
        tracker.record_co_occurrence(
            ["mem_1", "mem_2"],
            "session_1",
            timestamp=old_time,
        )
        
        # Record recent co-occurrence
        recent_time = current_time - timedelta(days=3)
        tracker.record_co_occurrence(
            ["mem_3", "mem_4"],
            "session_2",
            timestamp=recent_time,
        )
        
        # Prune old co-occurrences
        pruned_count = tracker.prune_old_co_occurrences(current_time)
        
        assert pruned_count > 0
        
        # Old co-occurrence should be removed
        related = tracker.get_related_memories("mem_1", current_time=current_time)
        assert len(related) == 0
        
        # Recent co-occurrence should remain
        related = tracker.get_related_memories("mem_3", current_time=current_time)
        assert len(related) == 1
    
    def test_get_graph_statistics(self):
        """Test graph statistics calculation."""
        tracker = MemoryCoOccurrenceTracker()
        
        # Empty graph
        stats = tracker.get_graph_statistics()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["avg_edges_per_node"] == 0.0
        
        # Add some co-occurrences
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        tracker.record_co_occurrence(["mem_1", "mem_3"], "session_2")
        tracker.record_co_occurrence(["mem_2", "mem_3"], "session_3")
        
        stats = tracker.get_graph_statistics()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 6  # Bidirectional: 3 pairs × 2
        assert stats["avg_edges_per_node"] == 2.0
    
    def test_clear(self):
        """Test clearing all co-occurrence data."""
        tracker = MemoryCoOccurrenceTracker()
        
        tracker.record_co_occurrence(["mem_1", "mem_2"], "session_1")
        tracker.clear()
        
        stats = tracker.get_graph_statistics()
        assert stats["node_count"] == 0
        
        related = tracker.get_related_memories("mem_1")
        assert len(related) == 0
    
    def test_nonexistent_memory_returns_empty_list(self):
        """Test that querying nonexistent memory returns empty list."""
        tracker = MemoryCoOccurrenceTracker()
        
        related = tracker.get_related_memories("nonexistent_mem")
        assert related == []


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_extract_memory_ids_with_id_key(self):
        """Test extracting memory IDs when 'id' key exists."""
        memories = [
            {"id": "mem_1", "text": "Memory 1"},
            {"id": "mem_2", "text": "Memory 2"},
        ]
        
        ids = extract_memory_ids(memories)
        assert ids == ["mem_1", "mem_2"]
    
    def test_extract_memory_ids_with_memory_id_key(self):
        """Test extracting memory IDs when 'memory_id' key exists."""
        memories = [
            {"memory_id": "mem_1", "text": "Memory 1"},
            {"memory_id": "mem_2", "text": "Memory 2"},
        ]
        
        ids = extract_memory_ids(memories)
        assert ids == ["mem_1", "mem_2"]
    
    def test_extract_memory_ids_with_text_fallback(self):
        """Test extracting memory IDs using text hash as fallback."""
        memories = [
            {"text": "Memory 1"},
            {"text": "Memory 2"},
        ]
        
        ids = extract_memory_ids(memories)
        assert len(ids) == 2
        assert all(id.startswith("mem_") for id in ids)
        # Same text should give same ID
        assert extract_memory_ids([{"text": "Memory 1"}])[0] == ids[0]
    
    def test_extract_memory_ids_mixed_formats(self):
        """Test extracting memory IDs from mixed formats."""
        memories = [
            {"id": "mem_1", "text": "Memory 1"},
            {"memory_id": "mem_2", "text": "Memory 2"},
            {"text": "Memory 3"},
        ]
        
        ids = extract_memory_ids(memories)
        assert len(ids) == 3
        assert ids[0] == "mem_1"
        assert ids[1] == "mem_2"
        assert ids[2].startswith("mem_")


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def test_multi_session_knowledge_graph_building(self):
        """Test building knowledge graph across multiple sessions."""
        tracker = MemoryCoOccurrenceTracker()
        
        # Session 1: User asks about AI and ML
        tracker.record_co_occurrence(
            ["ai_intro", "ml_basics", "neural_nets"],
            session_id="session_1",
        )
        
        # Session 2: User asks about ML and deep learning
        tracker.record_co_occurrence(
            ["ml_basics", "deep_learning", "neural_nets"],
            session_id="session_2",
        )
        
        # Session 3: User asks about neural nets specifically
        tracker.record_co_occurrence(
            ["neural_nets", "backprop", "optimization"],
            session_id="session_3",
        )
        
        # neural_nets should be a hub (connected to many memories)
        related = tracker.get_related_memories("neural_nets", top_k=10)
        related_ids = {r["memory_id"] for r in related}
        
        assert "ml_basics" in related_ids
        assert "deep_learning" in related_ids
        assert "ai_intro" in related_ids
    
    def test_context_expansion_use_case(self):
        """Test context expansion: retrieve related memories with primary match."""
        tracker = MemoryCoOccurrenceTracker()
        
        # Build knowledge graph: Python → [OOP, Functions, Data Structures]
        tracker.record_co_occurrence(
            ["python_intro", "python_oop", "python_functions"],
            session_id="session_1",
        )
        tracker.record_co_occurrence(
            ["python_intro", "python_data_structures"],
            session_id="session_2",
        )
        
        # User searches for "python" → primary match is "python_intro"
        # Context expansion: also retrieve related memories
        primary_match = "python_intro"
        related = tracker.get_related_memories(primary_match, top_k=3)
        
        # Should get related Python topics
        assert len(related) >= 2
        related_ids = {r["memory_id"] for r in related}
        assert "python_oop" in related_ids or "python_functions" in related_ids
    
    def test_temporal_decay_realistic_scenario(self):
        """Test that old co-occurrences decay over time."""
        tracker = MemoryCoOccurrenceTracker(decay_window_days=30)
        current_time = datetime.now(timezone.utc)
        
        # User was interested in topic A 60 days ago
        old_time = current_time - timedelta(days=60)
        tracker.record_co_occurrence(
            ["topic_a", "subtopic_a1", "subtopic_a2"],
            session_id="old_session",
            timestamp=old_time,
        )
        
        # User is now interested in topic B (recent)
        tracker.record_co_occurrence(
            ["topic_b", "subtopic_b1", "subtopic_b2"],
            session_id="recent_session",
            timestamp=current_time,
        )
        
        # topic_b should have related memories, topic_a should not (outside window)
        related_b = tracker.get_related_memories("topic_b", current_time=current_time)
        related_a = tracker.get_related_memories("topic_a", current_time=current_time)
        
        assert len(related_b) > 0
        assert len(related_a) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
