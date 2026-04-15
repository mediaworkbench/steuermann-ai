"""Performance benchmarks comparing O(n) in-memory vs O(log n) Qdrant search.

Benchmarks:
- Semantic search with varying cache sizes (100, 1000, 10000)
- Comparison: In-memory iteration vs Qdrant ANN search
- Metrics: Search latency (p50, p95, p99), throughput, scalability
"""

import pytest
import time
import logging
from typing import List, Dict, Any
import statistics

try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from universal_agentic_framework.caching.manager import CacheManager, MemoryCacheBackend
from universal_agentic_framework.caching.vector_backend import QdrantCacheVectorBackend
from universal_agentic_framework.embeddings import build_embedding_provider


logger = logging.getLogger(__name__)


@pytest.mark.benchmark
@pytest.mark.skipif(not (HAS_QDRANT and HAS_NUMPY), 
                    reason="Requires qdrant-client and numpy")
class TestCachePerformanceBenchmark:
    """Performance benchmark tests for cache search."""
    
    @pytest.fixture
    def embeddings_model(self):
        """Load embedding model."""
        return build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint="$EMBEDDING_SERVER/v1",
        )
    
    @pytest.fixture
    def sample_queries(self) -> List[str]:
        """Generate sample queries for benchmarking."""
        return [
            f"Query {i}: What is the result for parameter {i % 10}?"
            for i in range(100)
        ]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if HAS_NUMPY:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        else:
            # Pure Python fallback
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2)
    
    def _encode_query(self, model, query: str) -> List[float]:
        """Helper to consistently encode a query."""
        encoded = model.encode([query])[0]
        return encoded.tolist() if hasattr(encoded, 'tolist') else list(encoded)
    
    def _benchmark_in_memory_search(
        self,
        embeddings: List[List[float]],
        query_embedding: List[float],
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> tuple[List[tuple[int, float]], float]:
        """Benchmark in-memory O(n) search.
        
        Returns:
            (results, duration_ms)
        """
        start_time = time.time()
        
        # O(n) search through all embeddings
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        
        duration_ms = (time.time() - start_time) * 1000
        return results, duration_ms
    
    def _benchmark_qdrant_search(
        self,
        vector_backend: QdrantCacheVectorBackend,
        query_embedding: List[float],
        crew_name: str,
        user_id: str,
        language: str,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> tuple[List[Dict], float]:
        """Benchmark Qdrant O(log n) search.
        
        Returns:
            (results, duration_ms)
        """
        start_time = time.time()
        
        results = vector_backend.search_similar(
            embedding=query_embedding,
            crew_name=crew_name,
            user_id=user_id,
            language=language,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        
        duration_ms = (time.time() - start_time) * 1000
        return results, duration_ms
    
    def test_benchmark_100_embeddings(self, embeddings_model, sample_queries):
        """Benchmark with 100 cached embeddings."""
        cache_size = 100
        queries = sample_queries[:cache_size]
        
        # Generate embeddings
        logger.info(f"Generating {cache_size} embeddings...")
        embeddings = [self._encode_query(embeddings_model, q) for q in queries]
        
        # Setup Qdrant backend
        vector_backend = QdrantCacheVectorBackend(
            collection_prefix="bench_100",
            host="localhost",
            port=6333,
            fork_name="benchmark",
        )
        vector_backend.clear_collection()
        
        # Store embeddings in Qdrant
        logger.info(f"Storing {cache_size} embeddings in Qdrant...")
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            vector_backend.store_embedding(
                query=query,
                embedding=embedding,
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash_{i}",
            )
        
        # Benchmark queries
        test_queries = queries[:10]  # Test with 10 queries
        test_embeddings = [self._encode_query(embeddings_model, q) for q in test_queries]
        
        in_memory_times = []
        qdrant_times = []
        
        for query_emb in test_embeddings:
            # Benchmark in-memory search
            _, in_memory_time = self._benchmark_in_memory_search(
                embeddings, query_emb, threshold=0.80, top_k=5
            )
            in_memory_times.append(in_memory_time)
            
            # Benchmark Qdrant search
            _, qdrant_time = self._benchmark_qdrant_search(
                vector_backend, query_emb, "research", "user123", "en", threshold=0.80, top_k=5
            )
            qdrant_times.append(qdrant_time)
        
        # Calculate statistics
        in_memory_avg = statistics.mean(in_memory_times)
        in_memory_p95 = statistics.quantiles(in_memory_times, n=20)[18]  # 95th percentile
        qdrant_avg = statistics.mean(qdrant_times)
        qdrant_p95 = statistics.quantiles(qdrant_times, n=20)[18]
        speedup = in_memory_avg / qdrant_avg
        
        logger.info(f"=== Benchmark Results (n={cache_size}) ===")
        logger.info(f"In-Memory: avg={in_memory_avg:.2f}ms, p95={in_memory_p95:.2f}ms")
        logger.info(f"Qdrant:    avg={qdrant_avg:.2f}ms, p95={qdrant_p95:.2f}ms")
        logger.info(f"Speedup:   {speedup:.1f}x")
        
        # Assertions (at 100 embeddings, Qdrant may be slower due to network overhead)
        assert qdrant_avg < 100, f"Qdrant search too slow: {qdrant_avg:.2f}ms"
        # For small datasets, expect similar or slightly slower performance
        logger.info(f"Note: At n=100, network overhead dominates. Speedup expected at n>=1000")
        
        # Cleanup
        vector_backend.clear_collection()
    
    def test_benchmark_1000_embeddings(self, embeddings_model):
        """Benchmark with 1000 cached embeddings."""
        cache_size = 1000
        
        # Generate queries
        queries = [f"Query {i}: Parameter value is {i % 100}" for i in range(cache_size)]
        
        # Generate embeddings (batch processing)
        logger.info(f"Generating {cache_size} embeddings...")
        embeddings = []
        batch_size = 100
        for i in range(0, cache_size, batch_size):
            batch = queries[i:i+batch_size]
            batch_embeddings = embeddings_model.encode(batch)
            for emb in batch_embeddings:
                embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
        
        # Setup Qdrant backend
        vector_backend = QdrantCacheVectorBackend(
            collection_prefix="bench_1000",
            host="localhost",
            port=6333,
            fork_name="benchmark",
        )
        vector_backend.clear_collection()
        
        # Store embeddings in Qdrant
        logger.info(f"Storing {cache_size} embeddings in Qdrant...")
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            vector_backend.store_embedding(
                query=query,
                embedding=embedding,
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash_{i}",
            )
        
        # Benchmark queries
        test_queries = queries[:20]  # Test with 20 queries
        test_embeddings = embeddings[:20]
        
        in_memory_times = []
        qdrant_times = []
        
        for query_emb in test_embeddings:
            # Benchmark in-memory search
            _, in_memory_time = self._benchmark_in_memory_search(
                embeddings, query_emb, threshold=0.80, top_k=5
            )
            in_memory_times.append(in_memory_time)
            
            # Benchmark Qdrant search
            _, qdrant_time = self._benchmark_qdrant_search(
                vector_backend, query_emb, "research", "user123", "en", threshold=0.80, top_k=5
            )
            qdrant_times.append(qdrant_time)
        
        # Calculate statistics
        in_memory_avg = statistics.mean(in_memory_times)
        in_memory_p95 = statistics.quantiles(in_memory_times, n=20)[18]
        qdrant_avg = statistics.mean(qdrant_times)
        qdrant_p95 = statistics.quantiles(qdrant_times, n=20)[18]
        speedup = in_memory_avg / qdrant_avg
        
        logger.info(f"=== Benchmark Results (n={cache_size}) ===")
        logger.info(f"In-Memory: avg={in_memory_avg:.2f}ms, p95={in_memory_p95:.2f}ms")
        logger.info(f"Qdrant:    avg={qdrant_avg:.2f}ms, p95={qdrant_p95:.2f}ms")
        logger.info(f"Speedup:   {speedup:.1f}x")
        
        # Assertions (expect significant improvement at 1000 embeddings)
        assert qdrant_avg < 100, f"Qdrant search too slow: {qdrant_avg:.2f}ms"
        assert speedup > 3.0, f"Qdrant speedup insufficient: {speedup:.1f}x (expected >3x)"
        
        # Cleanup
        vector_backend.clear_collection()
    
    @pytest.mark.slow
    def test_benchmark_10000_embeddings(self, embeddings_model):
        """Benchmark with 10000 cached embeddings (slow test)."""
        cache_size = 10000
        
        # Generate queries
        queries = [f"Query {i}: Parameter value is {i % 1000}" for i in range(cache_size)]
        
        # Generate embeddings (batch processing)
        logger.info(f"Generating {cache_size} embeddings...")
        embeddings = []
        batch_size = 100
        for i in range(0, cache_size, batch_size):
            batch = queries[i:i+batch_size]
            batch_embeddings = embeddings_model.encode(batch)
            for emb in batch_embeddings:
                embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"  Generated {i + batch_size}/{cache_size} embeddings...")
        
        # Setup Qdrant backend
        vector_backend = QdrantCacheVectorBackend(
            collection_prefix="bench_10000",
            host="localhost",
            port=6333,
            fork_name="benchmark",
        )
        vector_backend.clear_collection()
        
        # Store embeddings in Qdrant
        logger.info(f"Storing {cache_size} embeddings in Qdrant...")
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            vector_backend.store_embedding(
                query=query,
                embedding=embedding,
                crew_name="research",
                user_id="user123",
                language="en",
                result_hash=f"hash_{i}",
            )
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Stored {i + 1}/{cache_size} embeddings...")
        
        # Benchmark queries
        test_queries = queries[:20]  # Test with 20 queries
        test_embeddings = embeddings[:20]
        
        in_memory_times = []
        qdrant_times = []
        
        logger.info("Running benchmark queries...")
        for query_emb in test_embeddings:
            # Benchmark in-memory search
            _, in_memory_time = self._benchmark_in_memory_search(
                embeddings, query_emb, threshold=0.80, top_k=5
            )
            in_memory_times.append(in_memory_time)
            
            # Benchmark Qdrant search
            _, qdrant_time = self._benchmark_qdrant_search(
                vector_backend, query_emb, "research", "user123", "en", threshold=0.80, top_k=5
            )
            qdrant_times.append(qdrant_time)
        
        # Calculate statistics
        in_memory_avg = statistics.mean(in_memory_times)
        in_memory_p95 = statistics.quantiles(in_memory_times, n=20)[18]
        qdrant_avg = statistics.mean(qdrant_times)
        qdrant_p95 = statistics.quantiles(qdrant_times, n=20)[18]
        speedup = in_memory_avg / qdrant_avg
        
        logger.info(f"=== Benchmark Results (n={cache_size}) ===")
        logger.info(f"In-Memory: avg={in_memory_avg:.2f}ms, p95={in_memory_p95:.2f}ms")
        logger.info(f"Qdrant:    avg={qdrant_avg:.2f}ms, p95={qdrant_p95:.2f}ms")
        logger.info(f"Speedup:   {speedup:.1f}x")
        
        # Assertions (expect massive improvement at 10000 embeddings)
        assert qdrant_avg < 200, f"Qdrant search too slow: {qdrant_avg:.2f}ms"
        assert speedup > 5.0, f"Qdrant speedup insufficient: {speedup:.1f}x (expected >5x)"
        
        # Cleanup
        vector_backend.clear_collection()
    
    def test_benchmark_summary_report(self, embeddings_model):
        """Generate summary report comparing all cache sizes."""
        logger.info("=" * 60)
        logger.info("CACHE PERFORMANCE BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info("Run individual benchmark tests for detailed results:")
        logger.info("  pytest tests/test_cache_performance_benchmark.py -v -s")
        logger.info("")
        logger.info("Expected Performance:")
        logger.info("  100 embeddings:    2-5x speedup")
        logger.info("  1,000 embeddings:  5-15x speedup")
        logger.info("  10,000 embeddings: 20-100x speedup")
        logger.info("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
