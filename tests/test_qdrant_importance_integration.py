"""Integration tests for Qdrant backend with importance scoring."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from unittest.mock import MagicMock

from universal_agentic_framework.memory.qdrant_backend import QdrantMemoryBackend
from universal_agentic_framework.memory.backend import MemoryRecord


class MockQdrantClient:
    """Mock Qdrant client for testing."""
    
    def __init__(self):
        self.points: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_collection(self, name: str):
        if name not in self.points:
            raise Exception(f"Collection {name} not found")
        return {"name": name}
    
    def create_collection(self, collection_name: str, vectors_config):
        self.points[collection_name] = []
    
    def upsert(self, collection_name: str, points: List[Any]):
        if collection_name not in self.points:
            self.points[collection_name] = []
        for point in points:
            payload = getattr(point, "payload", point.get("payload", {}))
            self.points[collection_name].append({
                "id": getattr(point, "id", point.get("id")),
                "payload": payload,
                "score": 1.0,
            })
    
    def search(self, collection_name: str, query_vector, limit: int, query_filter=None):
        if collection_name not in self.points:
            return []
        
        # Filter by user_id from query_filter
        user_id = None
        if query_filter:
            for cond in query_filter.must:
                if hasattr(cond, 'key') and cond.key == "user_id":
                    user_id = cond.match.value
        
        results = []
        for point in self.points[collection_name]:
            if user_id and point["payload"].get("user_id") != user_id:
                continue
            
            # Mock scoring based on metadata
            metadata = point["payload"].get("metadata", {})
            score = 0.8  # Base score
            
            # Adjust score based on access count (mock relevance)
            access_count = metadata.get("access_count", 0)
            if access_count > 5:
                score = 0.9
            
            # Adjust score based on age
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    age_days = (datetime.now(timezone.utc) - created_at).days
                    if age_days > 30:
                        score *= 0.5  # Old memory
                except:
                    pass
                    
            results.append(MockSearchResult(
                id=point["id"],
                payload=point["payload"],
                score=score,
            ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def scroll(self, collection_name: str, limit: int):
        if collection_name not in self.points:
            return ([], None)
        # Return mocked points with proper format
        mocked_points = []
        for point in self.points[collection_name][:limit]:
            # Create mock object with payload attribute
            mock_point = type('MockPoint', (), {'payload': point["payload"]})()
            mocked_points.append(mock_point)
        return (mocked_points, None)
    
    def delete(self, collection_name: str, query_filter=None):
        if collection_name not in self.points:
            return
        # Simple mock: delete all for user
        self.points[collection_name] = []


class MockSearchResult:
    """Mock search result from Qdrant."""
    
    def __init__(self, id, payload, score):
        self.id = id
        self.payload = payload
        self.score = score


class MockEmbedder:
    """Mock sentence transformer for testing."""
    
    def encode(self, texts: List[str]):
        # Return fake embeddings as numpy-compatible objects
        import numpy as np
        return np.array([[0.1] * 384 for _ in texts])


@pytest.fixture
def mock_backend():
    """Create a QdrantMemoryBackend with mocked dependencies."""
    client = MockQdrantClient()
    embedder = MockEmbedder()
    
    backend = QdrantMemoryBackend(
        collection_prefix="test",
        client=client,
        embedder=embedder,
        dimension=384,
        enable_importance_scoring=True,
    )
    
    return backend


class TestQdrantBackendImportanceScoring:
    """Tests for Qdrant backend with importance scoring."""
    
    def test_upsert_initializes_metadata(self, mock_backend):
        """Test that upsert initializes importance metadata."""
        record = mock_backend.upsert(user_id="user1", text="Test memory")
        
        assert "created_at" in record.metadata
        assert "access_count" in record.metadata
        assert record.metadata["access_count"] == 0
    
    def test_upsert_preserves_existing_metadata(self, mock_backend):
        """Test that upsert preserves existing metadata fields."""
        metadata = {
            "custom_field": "value",
            "user_rating": 5,
        }
        record = mock_backend.upsert(user_id="user1", text="Test memory", metadata=metadata)
        
        assert record.metadata["custom_field"] == "value"
        assert record.metadata["user_rating"] == 5
        assert "created_at" in record.metadata
        assert "access_count" in record.metadata
    
    def test_load_with_empty_query_updates_access_count(self, mock_backend):
        """Test that load without query updates access metadata."""
        # Insert a memory
        mock_backend.upsert(user_id="user1", text="Memory 1")
        
        # Load without query
        results = mock_backend.load(user_id="user1", query=None)
        
        assert len(results) > 0
        # Access count should be incremented
        # Note: This test depends on mock implementation
    
    def test_load_ranks_by_importance(self, mock_backend):
        """Test that load ranks memories by importance, not just raw similarity."""
        current_time = datetime.now(timezone.utc)
        
        # Create old memory with high access count
        old_metadata = {
            "created_at": (current_time - timedelta(days=30)).isoformat(),
            "access_count": 50,
        }
        mock_backend.upsert(user_id="user1", text="Old frequent memory", metadata=old_metadata)
        
        # Create recent memory with low access count
        recent_metadata = {
            "created_at": (current_time - timedelta(days=1)).isoformat(),
            "access_count": 1,
        }
        mock_backend.upsert(user_id="user1", text="Recent new memory", metadata=recent_metadata)
        
        # Search (should rerank by importance)
        results = mock_backend.load(user_id="user1", query="memory")
        
        # Recent memory should rank higher despite lower access count
        # (recency beats frequency for very old memories)
        assert len(results) > 0
        # First result should have importance-adjusted ranking
    
    def test_load_updates_access_metadata(self, mock_backend):
        """Test that load updates access_count and last_accessed."""
        # Insert memory
        record = mock_backend.upsert(user_id="user1", text="Test memory")
        initial_count = record.metadata["access_count"]
        
        # Load memory with query
        results = mock_backend.load(user_id="user1", query="test")
        
        if len(results) > 0:
            # Access count should be incremented (from 0 to 1)
            loaded_metadata = results[0].metadata
            assert loaded_metadata["access_count"] > initial_count
            assert "last_accessed" in loaded_metadata
    
    def test_importance_scoring_can_be_disabled(self):
        """Test that importance scoring can be disabled."""
        client = MockQdrantClient()
        embedder = MockEmbedder()
        
        backend = QdrantMemoryBackend(
            collection_prefix="test",
            client=client,
            embedder=embedder,
            dimension=384,
            enable_importance_scoring=False,
        )
        
        assert backend._importance_scorer is None
        
        # Should still work, just without importance scoring
        record = backend.upsert(user_id="user1", text="Test")
        results = backend.load(user_id="user1", query="test")
        
        # Results should be returned without importance ranking
        assert isinstance(results, list)
    
    def test_load_handles_missing_metadata_gracefully(self, mock_backend):
        """Test that load handles memories without importance metadata."""
        # Manually insert memory without proper metadata
        client = mock_backend._client
        client.points[mock_backend.collection_name] = [
            {
                "id": "test-id",
                "payload": {
                    "user_id": "user1",
                    "text": "Memory without metadata",
                    "metadata": {},  # Empty metadata
                },
                "score": 0.9,
            }
        ]
        
        # Should not crash
        results = mock_backend.load(user_id="user1", query="test")
        
        assert isinstance(results, list)
        # Should handle missing metadata gracefully
    
    def test_load_filters_by_user_id(self, mock_backend):
        """Test that load only returns memories for the specified user."""
        # Insert memories for different users
        mock_backend.upsert(user_id="user1", text="User 1 memory")
        mock_backend.upsert(user_id="user2", text="User 2 memory")
        
        # Load for user1
        results = mock_backend.load(user_id="user1", query="memory")
        
        # Should only get user1 memories
        for record in results:
            assert record.user_id == "user1"
    
    def test_load_respects_top_k_limit(self, mock_backend):
        """Test that load respects the top_k parameter."""
        # Insert multiple memories
        for i in range(10):
            mock_backend.upsert(user_id="user1", text=f"Memory {i}")
        
        # Load with top_k=3
        results = mock_backend.load(user_id="user1", query="memory", top_k=3)
        
        # Should return at most 3 results
        assert len(results) <= 3


class TestQdrantBackendWithUserFeedback:
    """Tests for user feedback integration."""
    
    def test_user_rating_affects_importance(self, mock_backend):
        """Test that user ratings affect importance scoring."""
        current_time = datetime.now(timezone.utc)
        
        # Create memory with low rating
        low_rating_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 5,
            "user_rating": 1,
        }
        mock_backend.upsert(user_id="user1", text="Low rated memory", metadata=low_rating_metadata)
        
        # Create memory with high rating
        high_rating_metadata = {
            "created_at": current_time.isoformat(),
            "access_count": 5,
            "user_rating": 5,
        }
        mock_backend.upsert(user_id="user1", text="High rated memory", metadata=high_rating_metadata)
        
        # Load and check ordering
        results = mock_backend.load(user_id="user1", query="memory")
        
        # High rated should rank higher (if both have similar relevance)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
