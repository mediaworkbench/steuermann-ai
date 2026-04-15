"""Integration test for memory analytics and co-occurrence tracking."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from universal_agentic_framework.memory.qdrant_backend import QdrantMemoryBackend
from universal_agentic_framework.memory.backend import MemoryRecord


def test_qdrant_backend_with_co_occurrence_tracking():
    """Test QdrantMemoryBackend with co-occurrence tracking enabled."""
    # Mock Qdrant client
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock embedding with numpy array
    import numpy as np
    mock_embedder.encode.return_value = [np.array([0.1] * 384)]
    
    # Create backend with co-occurrence tracking
    backend = QdrantMemoryBackend(
        client=mock_client,
        embedder=mock_embedder,
        collection_prefix="test",
        enable_importance_scoring=True,
        enable_co_occurrence_tracking=True,
        fork_name="test-fork",
    )
    
    # Verify co-occurrence tracker is initialized
    assert backend._co_occurrence_tracker is not None
    assert backend.enable_co_occurrence_tracking is True
    

def test_load_with_include_related_parameter():
    """Test that load() accepts include_related parameter without errors."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock search results with high score to pass filtering
    mock_result = MagicMock()
    mock_result.score = 0.95  # High score to pass importance filtering
    mock_result.payload = {
        "user_id": "user123",
        "text": "Memory 1",
        "metadata": {
            "memory_id": "mem1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 5,  # Non-zero access count
        }
    }
    mock_client.search.return_value = [mock_result]
    # Mock scroll for _fetch_memories_by_ids
    mock_client.scroll.return_value = ([], None)
    
    # Mock embedder returns numpy-like arrays
    import numpy as np
    mock_embedder.encode.return_value = [np.array([0.1] * 384)]
    
    backend = QdrantMemoryBackend(
        client=mock_client,
        embedder=mock_embedder,
        enable_co_occurrence_tracking=True,
        fork_name="test-fork",
    )
    
    # Call load with include_related - should not raise errors
    try:
        memories = backend.load(
            user_id="user123",
            query="test query",
            top_k=5,
            include_related=True,
            session_id="session_123",
        )
        # Test passes if no error is raised
        assert True, "load() with include_related executed without errors"
    except Exception as e:
        pytest.fail(f"load() with include_related raised error: {e}")


def test_upsert_adds_memory_id():
    """Test that upsert automatically adds memory_id to metadata."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock embedder returns numpy-like arrays
    import numpy as np
    mock_embedder.encode.return_value = [np.array([0.1] * 384)]
    
    backend = QdrantMemoryBackend(
        client=mock_client,
        embedder=mock_embedder,
        enable_co_occurrence_tracking=True,
    )
    
    # Upsert a memory
    with patch('httpx.put') as mock_put:
        mock_put.return_value.status_code = 200
        record = backend.upsert(
            user_id="user123",
            text="Test memory",
            metadata={"custom": "value"},
        )
    
    # Verify memory_id was added
    assert "memory_id" in record.metadata
    assert len(record.metadata["memory_id"]) == 12  # UUID hex[:12]


@patch('universal_agentic_framework.monitoring.metrics.track_memory_importance_ranking')
@patch('universal_agentic_framework.monitoring.metrics.track_memory_quality')
@patch('universal_agentic_framework.monitoring.metrics.track_memory_age')
@patch('universal_agentic_framework.monitoring.metrics.track_related_memories')
@patch('universal_agentic_framework.monitoring.metrics.track_memory_graph_statistics')
def test_load_tracks_metrics(
    mock_graph_stats,
    mock_related,
    mock_age,
    mock_quality,
    mock_ranking,
):
    """Test that load() tracks Prometheus metrics."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock search result with importance data
    created_at = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    mock_result = MagicMock()
    mock_result.score = 0.9
    mock_result.payload = {
        "user_id": "user123",
        "text": "Memory 1",
        "metadata": {
            "memory_id": "mem1",
            "created_at": created_at,
            "access_count": 3,
            "importance_score": 0.75,
        }
    }
    mock_client.search.return_value = [mock_result]
    mock_client.scroll.return_value = ([], None)
    
    # Mock embedder with numpy array
    import numpy as np
    mock_embedder.encode.return_value = [np.array([0.1] * 384)]
    
    backend = QdrantMemoryBackend(
        client=mock_client,
        embedder=mock_embedder,
        enable_importance_scoring=True,
        enable_co_occurrence_tracking=True,
        fork_name="test-fork",
    )
    
    # Call load with include_related
    memories = backend.load(
        user_id="user123",
        query="test query",
        top_k=5,
        include_related=True,
        session_id="session_123",
    )
    
    # Verify metrics were tracked
    assert mock_ranking.called, "Should track importance ranking duration"
    assert mock_quality.called, "Should track memory quality score"
    assert mock_age.called, "Should track memory age"
    # Related memories and graph stats only tracked if co-occurrence tracker returns results
    

def test_co_occurrence_tracker_is_optional():
    """Test that co-occurrence tracking can be disabled."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock embedder with numpy array
    import numpy as np
    mock_embedder.encode.return_value = [np.array([0.1] * 384)]
    
    backend = QdrantMemoryBackend(
        client=mock_client,
        embedder=mock_embedder,
        enable_co_occurrence_tracking=False,
    )
    
    # Verify co-occurrence tracker is None
    assert backend._co_occurrence_tracker is None
    assert backend.enable_co_occurrence_tracking is False
    
    # Load should still work without errors
    mock_result = MagicMock()
    mock_result.score = 0.9
    mock_result.payload = {
        "user_id": "user123",
        "text": "Memory 1",
        "metadata": {"created_at": datetime.now(timezone.utc).isoformat(), "access_count": 0}
    }
    mock_client.search.return_value = [mock_result]
    
    memories = backend.load(
        user_id="user123",
        query="test",
        top_k=5,
        include_related=True,  # Should be ignored since tracking is disabled
        session_id="session_123",
    )
    
    assert len(memories) == 1
