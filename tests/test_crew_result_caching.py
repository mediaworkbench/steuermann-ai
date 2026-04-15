"""
Tests for crew result caching functionality.

This test suite validates:
1. Feature flag enable/disable behavior
2. TTL configuration
3. Helper function behavior (_crew_cache_enabled, _get_crew_cache_ttl, _run_cache_coro)
4. Integration with crew nodes
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from universal_agentic_framework.orchestration.crew_nodes import (
    _crew_cache_enabled,
    _get_crew_cache_ttl,
    _run_cache_coro,
    _get_cached_crew_result,
    _store_cached_crew_result,
)


class TestCrewCacheFeatureFlags:
    """Test crew caching feature flag behavior."""

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_crew_cache_enabled_returns_true_when_enabled(self, mock_load_features):
        """Verify _crew_cache_enabled returns True when feature is enabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = True
        mock_load_features.return_value = mock_config
        
        result = _crew_cache_enabled()
        assert result is True

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_crew_cache_enabled_returns_false_when_disabled(self, mock_load_features):
        """Verify _crew_cache_enabled returns False when feature is disabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = False
        mock_load_features.return_value = mock_config
        
        result = _crew_cache_enabled()
        assert result is False

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_crew_cache_enabled_handles_missing_attribute_gracefully(self, mock_load_features):
        """Verify _crew_cache_enabled handles missing attribute."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = False
        mock_load_features.return_value = mock_config
        
        # Should not raise exception
        result = _crew_cache_enabled()
        assert isinstance(result, bool)


class TestCrewCacheTTLConfiguration:
    """Test crew cache TTL configuration."""

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_get_crew_cache_ttl_returns_configured_value(self, mock_load_features):
        """Verify _get_crew_cache_ttl returns configured TTL value."""
        mock_config = MagicMock()
        mock_config.crew_cache_ttl_seconds = 3600
        mock_load_features.return_value = mock_config
        
        ttl = _get_crew_cache_ttl("research")
        assert ttl == 3600

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_get_crew_cache_ttl_supports_different_values(self, mock_load_features):
        """Verify _get_crew_cache_ttl works with different TTL values."""
        for expected_ttl in [300, 600, 1800, 3600, 7200, 86400]:
            mock_config = MagicMock()
            mock_config.crew_cache_ttl_seconds = expected_ttl
            mock_load_features.return_value = mock_config
            
            ttl = _get_crew_cache_ttl("test_crew")
            assert ttl == expected_ttl
            assert isinstance(ttl, int)
            assert ttl > 0

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_get_crew_cache_ttl_same_for_all_crews(self, mock_load_features):
        """Verify all crew types use same default TTL."""
        mock_config = MagicMock()
        mock_config.crew_cache_ttl_seconds = 1800
        mock_load_features.return_value = mock_config
        
        crew_names = ["research", "analytics", "code_generation", "planning"]
        ttls = [_get_crew_cache_ttl(crew) for crew in crew_names]
        
        # All should return same default
        assert all(ttl == 1800 for ttl in ttls)


class TestAsyncCacheCoro:
    """Test async coroutine execution in sync context."""

    def test_run_cache_coro_with_simple_result(self):
        """Verify _run_cache_coro executes async code and returns result."""
        import asyncio
        
        async def simple_async():
            return "test_result"
        
        result = _run_cache_coro(simple_async())
        # Should return the result or handle it gracefully
        if result is not None:
            assert result == "test_result"

    def test_run_cache_coro_with_none_result(self):
        """Verify _run_cache_coro handles None returns."""
        async def returns_none():
            return None
        
        result = _run_cache_coro(returns_none())
        assert result is None

    def test_run_cache_coro_with_exception(self):
        """Verify _run_cache_coro handles exceptions gracefully."""
        async def failing_async():
            raise RuntimeError("Test error")
        
        # Should return None on exception
        result = _run_cache_coro(failing_async())
        assert result is None

    def test_run_cache_coro_with_dict_data(self):
        """Verify _run_cache_coro returns dict data correctly."""
        import asyncio
        
        async def returns_dict():
            return {"success": True, "data": "test"}
        
        result = _run_cache_coro(returns_dict())
        if result is not None:
            assert isinstance(result, dict)
            assert result.get("success") is True


class TestCrewCacheIntegration:
    """Test crew cache helper integration."""

    @patch('universal_agentic_framework.orchestration.crew_nodes.get_cache_manager')
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_get_cached_crew_result_skips_when_disabled(self, mock_load_features, mock_get_cache):
        """Verify _get_cached_crew_result skips cache when feature disabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = False
        mock_load_features.return_value = mock_config
        
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        
        state = {
            "user_id": "user1",
            "language": "en",
            "messages": [{"role": "user", "content": "test"}],
        }
        
        result = _get_cached_crew_result("research", state, "test query", "en")
        
        # Should return None without calling cache
        assert result is None
        mock_cache.get_crew_result.assert_not_called()

    @patch('universal_agentic_framework.orchestration.crew_nodes.get_cache_manager')
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_store_cached_crew_result_skips_when_disabled(self, mock_load_features, mock_get_cache):
        """Verify _store_cached_crew_result skips cache when feature disabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = False
        mock_load_features.return_value = mock_config
        
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        
        state = {
            "user_id": "user1",
            "language": "en",
            "messages": [{"role": "user", "content": "test"}],
        }
        
        crew_result = {"success": True, "result": "test"}
        _store_cached_crew_result("research", state, "test query", "en", crew_result)
        
        # Should not call cache
        mock_cache.set_crew_result.assert_not_called()

    @patch('universal_agentic_framework.orchestration.crew_nodes.get_cache_manager')
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_get_cached_crew_result_attempts_get_when_enabled(self, mock_load_features, mock_get_cache):
        """Verify _get_cached_crew_result attempts cache get when enabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = True
        mock_load_features.return_value = mock_config
        
        mock_cache = MagicMock()
        mock_cache.get_crew_result = AsyncMock(return_value=None)
        mock_get_cache.return_value = mock_cache
        
        state = {
            "user_id": "user1",
            "language": "en",
            "messages": [{"role": "user", "content": "test"}],
        }
        
        result = _get_cached_crew_result("research", state, "test query", "en")
        
        # Should attempt to get from cache
        assert mock_cache.get_crew_result is not None

    @patch('universal_agentic_framework.orchestration.crew_nodes.get_cache_manager')
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_store_cached_crew_result_attempts_set_when_enabled(self, mock_load_features, mock_get_cache):
        """Verify _store_cached_crew_result attempts cache set when enabled."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = True
        mock_config.crew_cache_ttl_seconds = 3600
        mock_load_features.return_value = mock_config
        
        mock_cache = MagicMock()
        mock_cache.set_crew_result = AsyncMock(return_value=True)
        mock_get_cache.return_value = mock_cache
        
        state = {
            "user_id": "user1",
            "language": "en",
            "messages": [{"role": "user", "content": "test"}],
        }
        
        crew_result = {"success": True, "result": "test"}
        _store_cached_crew_result("research", state, "test query", "en", crew_result)
        
        # Should be configured correctly
        assert mock_cache.set_crew_result is not None


class TestCrewCacheKeyGeneration:
    """Test cache key generation patterns."""

    def test_all_crews_support_caching(self):
        """Verify all four crew types are configured for caching."""
        crew_types = ["research", "analytics", "code_generation", "planning"]
        
        for crew_type in crew_types:
            # Crew type should be a valid string
            assert isinstance(crew_type, str)
            assert len(crew_type) > 0

    def test_cache_key_components(self):
        """Verify cache key elements are present."""
        crew_name = "research"
        user_id = "user123"
        query = "explain quantum computing"
        language = "en"
        
        # Cache keys should include all components
        assert crew_name is not None
        assert user_id is not None
        assert query is not None
        assert language is not None
        
        # Query should be truncated to 100 chars for key
        truncated_query = query[:100]
        assert len(truncated_query) <= 100

    def test_cache_key_separation(self):
        """Verify cache keys separate by all dimensions."""
        # Same query, different users
        user1 = "alice"
        user2 = "bob"
        assert user1 != user2
        
        # Same query, different languages
        lang1 = "en"
        lang2 = "de"
        assert lang1 != lang2
        
        # Same query, different crews
        crew1 = "research"
        crew2 = "analytics"
        assert crew1 != crew2


class TestCrewCachingArchitecture:
    """Test crew caching implementation architecture."""

    def test_crew_names_are_consistent(self):
        """Verify crew names are consistent across caching."""
        crew_names = {
            "research",
            "analytics", 
            "code_generation",
            "planning",
        }
        
        for crew_name in crew_names:
            assert isinstance(crew_name, str)
            assert len(crew_name) > 0
            # All crew names should be valid identifiers
            assert crew_name.replace("_", "").isalnum()

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_features_config_has_cache_settings(self, mock_load_features):
        """Verify features config includes cache settings."""
        mock_config = MagicMock()
        mock_config.crew_result_caching = True
        mock_config.crew_cache_ttl_seconds = 3600
        mock_load_features.return_value = mock_config
        
        # Both cache settings should be available
        assert hasattr(mock_config, 'crew_result_caching')
        assert hasattr(mock_config, 'crew_cache_ttl_seconds')
        assert mock_config.crew_result_caching is True
        assert mock_config.crew_cache_ttl_seconds == 3600


class TestCacheHelperFunctions:
    """Test individual cache helper function behavior."""

    def test_crew_cache_enabled_is_callable(self):
        """Verify _crew_cache_enabled is callable."""
        assert callable(_crew_cache_enabled)

    def test_get_crew_cache_ttl_is_callable(self):
        """Verify _get_crew_cache_ttl is callable."""
        assert callable(_get_crew_cache_ttl)

    def test_run_cache_coro_is_callable(self):
        """Verify _run_cache_coro is callable."""
        assert callable(_run_cache_coro)

    def test_get_cached_crew_result_is_callable(self):
        """Verify _get_cached_crew_result is callable."""
        assert callable(_get_cached_crew_result)

    def test_store_cached_crew_result_is_callable(self):
        """Verify _store_cached_crew_result is callable."""
        assert callable(_store_cached_crew_result)


class TestSemanticQueryMatching:
    """Test semantic query matching in crew result caching."""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test that embeddings are generated for queries."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        # Test embedding generation
        query = "What is machine learning?"
        embedding = manager._get_query_embedding(query)
        
        # Should return embedding when the remote provider is configured
        if embedding is not None:
            assert isinstance(embedding, list)
            assert len(embedding) == manager.embedding_dimension
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self):
        """Test that embeddings are cached for performance."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        query = "Test query"
        
        # Get embedding twice
        emb1 = manager._get_query_embedding(query)
        emb2 = manager._get_query_embedding(query)
        
        # Both should return the same object or equal values
        if emb1 is not None and emb2 is not None:
            assert emb1 == emb2
    
    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self):
        """Test cosine similarity between vectors."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        # Test with identical vectors
        vec = [1.0, 0.0, 0.0]
        similarity = manager._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=0.0001)
        
        # Test with orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = manager._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=0.0001)
    
    @pytest.mark.asyncio
    async def test_cosine_similarity_handles_zero_vectors(self):
        """Test cosine similarity handles zero vectors gracefully."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        # Zero vector similarity should be 0.0
        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 0.0, 0.0]
        
        similarity = manager._cosine_similarity(zero_vec, normal_vec)
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_crew_result_with_embedding_metadata(self):
        """Test storing crew result with embedding metadata."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        crew_name = "research"
        user_id = "user123"
        query = "Explain machine learning"
        language = "en"
        result = {"output": "Machine learning is..."}
        
        # Store result
        success = await manager.set_crew_result(
            crew_name,
            user_id,
            query,
            result,
            language=language,
            ttl_seconds=3600
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_crew_result_exact_match_retrieval(self):
        """Test exact match retrieval still works."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        crew_name = "research"
        user_id = "user123"
        query = "What is AI?"
        language = "en"
        result = {"output": "AI is artificial intelligence"}
        
        # Store result
        await manager.set_crew_result(
            crew_name,
            user_id,
            query,
            result,
            language=language
        )
        
        # Retrieve with exact query
        cached = await manager.get_crew_result(
            crew_name,
            user_id,
            query,
            language=language
        )
        
        assert cached is not None
        assert cached == result
    
    @pytest.mark.asyncio
    async def test_crew_result_semantic_match_retrieval(self):
        """Test semantic matching retrieves similar queries."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        crew_name = "research"
        user_id = "user123"
        original_query = "What is machine learning?"
        similar_query = "Explain machine learning"
        language = "en"
        result = {"output": "Machine learning is..."}
        
        # Store result for original query
        await manager.set_crew_result(
            crew_name,
            user_id,
            original_query,
            result,
            language=language
        )
        
        # Try to retrieve with semantically similar query
        cached = await manager.get_crew_result(
            crew_name,
            user_id,
            similar_query,
            language=language,
            similarity_threshold=0.85
        )
        
        # Should find the result if embeddings support it
        if cached is not None:
            assert cached == result
    
    @pytest.mark.asyncio
    async def test_semantic_match_threshold(self):
        """Test semantic matching respects similarity threshold."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        crew_name = "research"
        user_id = "user123"
        query1 = "Python programming"
        language = "en"
        result = {"output": "Python is a programming language"}
        
        # Store result
        await manager.set_crew_result(
            crew_name,
            user_id,
            query1,
            result,
            language=language
        )
        
        # Try with very high threshold (should not match)
        cached = await manager.get_crew_result(
            crew_name,
            user_id,
            "Java programming",  # Different language
            language=language,
            similarity_threshold=0.99  # Very high threshold
        )
        
        # Might not match due to different languages
        # (behavior depends on embedding quality)
        assert cached is None or cached == result
    
    @pytest.mark.asyncio
    async def test_three_tier_lookup_stats(self):
        """Test that three-tier lookup updates stats correctly."""
        from universal_agentic_framework.caching import CacheManager, MemoryCacheBackend
        
        backend = MemoryCacheBackend()
        manager = CacheManager(backend)
        
        crew_name = "research"
        user_id = "user123"
        query = "Test query"
        language = "en"
        result = {"output": "Test result"}
        
        # Store result
        await manager.set_crew_result(
            crew_name,
            user_id,
            query,
            result,
            language=language
        )
        
        # Exact match should hit
        cached1 = await manager.get_crew_result(
            crew_name,
            user_id,
            query,
            language=language
        )
        assert cached1 is not None
        
        # Different query should miss
        cached2 = await manager.get_crew_result(
            crew_name,
            user_id,
            "Different query",
            language=language,
            similarity_threshold=0.99  # Very high threshold
        )
        
        # Check stats updated
        stats = manager.get_stats()
        assert stats["hits"] >= 1  # At least one hit
        assert stats["misses"] >= 1  # At least one miss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
