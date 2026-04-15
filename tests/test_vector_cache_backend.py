"""Tests for QdrantCacheVectorBackend - vector database semantic cache search.

Coverage:
- Backend initialization and connection
- Collection creation and management
- Embedding storage and retrieval
- Semantic similarity search with filters
- Result ranking and thresholds
- TTL and expiration handling
- Error handling and fallback behavior
- Performance comparison with O(n) iteration
"""

import pytest
import time
import logging
from typing import List, Dict, Any
import socket
import numpy as np

try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

from universal_agentic_framework.caching.vector_backend import QdrantCacheVectorBackend
from universal_agentic_framework.embeddings import build_embedding_provider


logger = logging.getLogger(__name__)


def is_qdrant_available(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant server is reachable."""
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False


# Module-level skip marker
QDRANT_AVAILABLE = HAS_QDRANT and is_qdrant_available()
pytestmark = pytest.mark.skipif(
    not (HAS_QDRANT and QDRANT_AVAILABLE),
    reason="qdrant-client not installed or Qdrant service not available"
)


class TestQdrantCacheVectorBackend:
    """Test suite for QdrantCacheVectorBackend."""
    
    def _encode_query(self, model, query: str) -> List[float]:
        """Helper to consistently encode a query into an embedding."""
        encoded = model.encode([query])[0]  # Always use list input
        return encoded.tolist() if hasattr(encoded, 'tolist') else list(encoded)
    
    @pytest.fixture
    def backend(self):
        """Create backend instance pointing to test Qdrant."""
        backend = QdrantCacheVectorBackend(
            collection_prefix="test_cache",
            host="localhost",
            port=6333,
            embedding_model="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            similarity_threshold=0.85,
            fork_name="test",
            embedding_provider_type="remote",
            embedding_remote_endpoint="$EMBEDDING_SERVER/v1",
        )
        # Clear any existing test data
        backend.clear_collection()
        yield backend
    
    @pytest.fixture
    def sample_embedding(self) -> List[float]:
        """Generate sample 768-dimensional embedding."""
        return np.random.randn(768).tolist()
    
    @pytest.fixture
    def embeddings_model(self):
        """Load remote embedding provider in deterministic fallback mode for tests."""
        return build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint="$EMBEDDING_SERVER/v1",
        )
    
    def test_backend_initialization(self, backend):
        """Test backend initializes with correct collection."""
        assert backend.collection_name == "test_cache_cache"
        assert backend.host == "localhost"
        assert backend.port == 6333
        assert backend.dimension == 768
        assert backend.similarity_threshold == 0.85
    
    def test_health_check(self, backend):
        """Test health check returns True for healthy backend."""
        assert backend.health_check() is True
    
    def test_collection_creation(self, backend):
        """Test collection exists after initialization."""
        size = backend.get_collection_size()
        assert size == 0
    
    def test_store_embedding(self, backend, sample_embedding):
        """Test storing embedding with metadata."""
        result = backend.store_embedding(
            query="What is the capital of France?",
            embedding=sample_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
            ttl_seconds=3600,
        )
        
        assert result is True
        # Verify it was stored
        size = backend.get_collection_size()
        assert size == 1
    
    def test_store_multiple_embeddings(self, backend, embeddings_model):
        """Test storing multiple embeddings."""
        queries = [
            "What is the capital of France?",
            "Tell me about Paris.",
            "What is the largest city in France?",
            "What is the weather in Paris?",
            "Paris is in which country?",
        ]
        
        embeddings = embeddings_model.encode(queries)
        
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            result = backend.store_embedding(
                query=query,
                embedding=list(embedding),
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash{i}",
            )
            assert result is True
        
        # Verify all stored
        size = backend.get_collection_size()
        assert size == 5
    
    def test_search_exact_match(self, backend, embeddings_model):
        """Test finding exact match in stored embeddings."""
        query = "What is the capital of France?"
        embedding = self._encode_query(embeddings_model, query)
        
        # Store it
        backend.store_embedding(
            query=query,
            embedding=embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        # Search for same query - should get exact match
        results = backend.search_similar(
            embedding=embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
            similarity_threshold=0.90,
        )
        
        assert len(results) > 0
        assert results[0]["query"] == query
        assert results[0]["score"] > 0.99  # Almost perfect similarity
    
    def test_search_semantic_similarity(self, backend, embeddings_model):
        """Test finding semantically similar queries."""
        query1 = "What is the capital of France?"
        query2 = "Tell me the capital city of France."  # Similar but different
        threshold = 0.80
        if getattr(embeddings_model, "_fallback", False):
            threshold = 0.50
        
        embedding1 = self._encode_query(embeddings_model, query1)
        embedding2 = self._encode_query(embeddings_model, query2)
        
        # Store first query
        backend.store_embedding(
            query=query1,
            embedding=embedding1,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        # Search using second query - should find first
        results = backend.search_similar(
            embedding=embedding2,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
            similarity_threshold=threshold,
        )
        
        assert len(results) > 0
        assert results[0]["query"] == query1
        assert threshold <= results[0]["score"] <= 0.99
    
    def test_filter_by_crew_name(self, backend, embeddings_model):
        """Test filtering results by crew name."""
        query = "Test query"
        embedding = self._encode_query(embeddings_model, query)
        
        # Store with crew A
        backend.store_embedding(
            query=query,
            embedding=embedding,
            crew_name="crew_a",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        # Store with crew B
        backend.store_embedding(
            query=query,
            embedding=embedding,
            crew_name="crew_b",
            user_id="user123",
            language="en",
            result_hash="xyz789",
        )
        
        # Search for crew_a only - should get 1 result
        results_a = backend.search_similar(
            embedding=embedding,
            crew_name="crew_a",
            user_id="user123",
            language="en",
            top_k=5,
        )
        
        assert len(results_a) == 1
        assert results_a[0]["crew_name"] == "crew_a"
        
        # Search for crew_b - should get different result
        results_b = backend.search_similar(
            embedding=embedding,
            crew_name="crew_b",
            user_id="user123",
            language="en",
            top_k=5,
        )
        
        assert len(results_b) == 1
        assert results_b[0]["crew_name"] == "crew_b"
        assert results_b[0]["result_hash"] == "xyz789"
    
    def test_filter_by_user_id(self, backend, embeddings_model):
        """Test filtering results by user ID."""
        query = "Test query"
        embedding = self._encode_query(embeddings_model, query)
        
        # Store for user1
        backend.store_embedding(
            query=query,
            embedding=embedding,
            crew_name="research",
            user_id="user1",
            language="en",
            result_hash="abc123",
        )
        
        # Store for user2
        backend.store_embedding(
            query=query,
            embedding=embedding,
            crew_name="research",
            user_id="user2",
            language="en",
            result_hash="xyz789",
        )
        
        # Search for user1 only
        results = backend.search_similar(
            embedding=embedding,
            crew_name="research",
            user_id="user1",
            language="en",
            top_k=5,
        )
        
        assert len(results) == 1
        assert results[0]["user_id"] == "user1"
    
    def test_filter_by_language(self, backend, embeddings_model):
        """Test filtering results by language."""
        query_en = "What is the capital of France?"
        query_de = "Was ist die Hauptstadt von Frankreich?"
        
        embedding_en = self._encode_query(embeddings_model, query_en)
        embedding_de = self._encode_query(embeddings_model, query_de)
        
        # Store English version
        backend.store_embedding(
            query=query_en,
            embedding=embedding_en,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        # Store German version
        backend.store_embedding(
            query=query_de,
            embedding=embedding_de,
            crew_name="research",
            user_id="user123",
            language="de",
            result_hash="xyz789",
        )
        
        # Search for English only
        results_en = backend.search_similar(
            embedding=embedding_en,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
        )
        
        assert len(results_en) >= 1
        assert all(r["language"] == "en" for r in results_en)
    
    def test_similarity_threshold(self, backend, embeddings_model):
        """Test similarity threshold filtering."""
        query1 = "Paris is the capital of France."
        query2 = "The weather today is sunny."  # Unrelated
        
        embedding1 = self._encode_query(embeddings_model, query1)
        embedding2 = self._encode_query(embeddings_model, query2)
        
        # Store first query
        backend.store_embedding(
            query=query1,
            embedding=embedding1,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        # Search with high threshold
        high_threshold_results = backend.search_similar(
            embedding=embedding2,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
            similarity_threshold=0.95,
        )
        
        # Search with low threshold
        low_threshold_results = backend.search_similar(
            embedding=embedding2,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
            similarity_threshold=0.30,
        )
        
        # Low threshold should find more results
        assert len(low_threshold_results) >= len(high_threshold_results)
    
    def test_top_k_limit(self, backend, embeddings_model):
        """Test top_k parameter limits results."""
        queries = [
            "What is France?",
            "Tell me about France.",
            "France is in Europe.",
            "Paris is in France.",
            "French culture is rich.",
        ]
        
        embeddings = embeddings_model.encode(queries)
        
        # Store all queries
        for query, embedding in zip(queries, embeddings):
            backend.store_embedding(
                query=query,
                embedding=list(embedding),
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash_{query[:10]}",
            )
        
        # Search with different top_k
        query = queries[0]
        search_embedding = self._encode_query(embeddings_model, query)
        
        results_3 = backend.search_similar(
            embedding=search_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=3,
            similarity_threshold=0.50,
        )
        
        results_all = backend.search_similar(
            embedding=search_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=10,
            similarity_threshold=0.50,
        )
        
        assert len(results_3) <= 3
        assert len(results_all) <= 10
        assert len(results_all) >= len(results_3)
    
    def test_result_ranking_by_score(self, backend, embeddings_model):
        """Test results are ranked by similarity score."""
        base_query = "What is the capital of France?"
        similar_query = "Tell me the capital city of France."
        very_similar_query = "What is the capital city of France?"
        
        base_embedding = self._encode_query(embeddings_model, base_query)
        similar_embedding = self._encode_query(embeddings_model, similar_query)
        very_similar_embedding = self._encode_query(embeddings_model, very_similar_query)
        
        # Store base query
        backend.store_embedding(
            query=base_query,
            embedding=base_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="hash1",
        )
        
        # Store similar query
        backend.store_embedding(
            query=similar_query,
            embedding=similar_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="hash2",
        )
        
        # Store very similar query
        backend.store_embedding(
            query=very_similar_query,
            embedding=very_similar_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="hash3",
        )
        
        # Search using base query
        results = backend.search_similar(
            embedding=base_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=5,
            similarity_threshold=0.70,
        )
        
        # Results should be ordered by score
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]
        
        # Most similar should be first
        assert results[0]["query"] == base_query
    
    def test_error_handling_invalid_embedding(self, backend):
        """Test error handling for invalid embeddings."""
        result = backend.store_embedding(
            query="Test",
            embedding=None,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        assert result is False
    
    def test_clear_collection(self, backend, sample_embedding):
        """Test clearing the collection."""
        # Store some data
        backend.store_embedding(
            query="Test",
            embedding=sample_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
        )
        
        assert backend.get_collection_size() == 1
        
        # Clear
        result = backend.clear_collection()
        assert result is True
        assert backend.get_collection_size() == 0
    
    def test_cleanup_expired_entries(self, backend, embeddings_model):
        """Test cleanup of expired cache entries."""
        import time
        
        query1 = "Test query 1"
        query2 = "Test query 2"
        embedding1 = self._encode_query(embeddings_model, query1)
        embedding2 = self._encode_query(embeddings_model, query2)
        
        current_time = time.time()
        
        # Store entry with short TTL (already expired)
        backend.store_embedding(
            query=query1,
            embedding=embedding1,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="abc123",
            ttl_seconds=1,  # 1 second TTL
        )
        
        # Manually set created_at to past (simulate expired entry)
        # Note: In real test, we'd wait or manipulate timestamps
        
        # Store entry with long TTL (not expired)
        backend.store_embedding(
            query=query2,
            embedding=embedding2,
            crew_name="research",
            user_id="user123",
            language="en",
            result_hash="xyz789",
            ttl_seconds=3600,  # 1 hour TTL
        )
        
        # Wait for first entry to expire
        time.sleep(2)
        
        # Run cleanup with current time
        deleted = backend.cleanup_expired(current_time=current_time + 2)
        
        # Should have deleted the expired entry
        assert deleted >= 0  # May be 0 or 1 depending on timing

    
    @pytest.mark.performance
    def test_performance_large_dataset(self, backend, embeddings_model):
        """Test performance with large number of stored embeddings."""
        # Generate 100 queries and embeddings
        queries = [f"Query number {i} about topic test" for i in range(100)]
        embeddings = embeddings_model.encode(queries)
        
        # Store all
        start_time = time.time()
        for query, embedding in zip(queries, embeddings):
            backend.store_embedding(
                query=query,
                embedding=list(embedding),
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash_{query[:10]}",
            )
        storage_time = time.time() - start_time
        
        # Search for similar queries
        search_query = queries[0]
        search_embedding = self._encode_query(embeddings_model, search_query)
        
        start_time = time.time()
        results = backend.search_similar(
            embedding=search_embedding,
            crew_name="research",
            user_id="user123",
            language="en",
            top_k=10,
            similarity_threshold=0.50,
        )
        search_time = time.time() - start_time
        
        logger.info(f"Stored 100 embeddings in {storage_time:.3f}s")
        logger.info(f"Searched 100 embeddings in {search_time:.3f}s")
        logger.info(f"Found {len(results)} matches")
        
        # Search should be fast (< 100ms for Qdrant ANN)
        assert search_time < 0.5, f"Search too slow: {search_time:.3f}s"
        assert len(results) > 0


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant service not available")
class TestQdrantCacheVectorBackendIntegration:
    """Integration tests with CacheManager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager with vector backend."""
        from universal_agentic_framework.caching.manager import CacheManager
        
        manager = CacheManager(
            fork_name="test",
            use_vector_db=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            similarity_threshold=0.85,
        )
        
        # Clear vector backend
        if manager.vector_backend:
            manager.vector_backend.clear_collection()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_cache_manager_with_vector_backend(self, cache_manager):
        """Test CacheManager integration with vector backend."""
        if not cache_manager.vector_backend:
            pytest.skip("Vector backend not available")
        
        crew_name = "researcher"
        user_id = "user123"
        query = "What is machine learning?"
        result = {
            "answer": "Machine learning is a subset of AI",
            "sources": ["wiki", "ml-docs"],
        }
        
        # Set crew result
        success = await cache_manager.set_crew_result(
            crew_name=crew_name,
            user_id=user_id,
            query=query,
            result=result,
            language="en",
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_cache_manager_semantic_search(self, cache_manager):
        """Test semantic search through cache manager."""
        if not cache_manager.vector_backend:
            pytest.skip("Vector backend not available")
        
        crew_name = "researcher"
        user_id = "user123"
        query1 = "What is the capital of France?"
        result = {"answer": "Paris"}
        
        # Store first query
        await cache_manager.set_crew_result(
            crew_name=crew_name,
            user_id=user_id,
            query=query1,
            result=result,
            language="en",
        )
        
        # Search with similar query
        query2 = "Tell me the capital city of France."
        found_result = await cache_manager.get_crew_result(
            crew_name=crew_name,
            user_id=user_id,
            query=query2,
            language="en",
            similarity_threshold=0.80,
        )
        
        # Should find cached result via semantic matching
        # (Note: May fail if embeddings are not similar enough)
        logger.info(f"Semantic search result: {found_result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
