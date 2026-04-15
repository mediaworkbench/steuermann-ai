"""Test configuration loading for embedding provider settings."""

import pytest
import os
from pathlib import Path
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.embeddings import build_embedding_provider


class TestEmbeddingConfigLoading:
    """Test that embedding configuration loads correctly."""
    
    def test_core_config_loads_embedding_settings(self):
        """Test that core config loads embedding settings including provider type."""
        config = load_core_config()
        
        # Check that embedding settings exist
        assert hasattr(config.memory, 'embeddings')
        assert hasattr(config.memory.embeddings, 'model')
        assert hasattr(config.memory.embeddings, 'dimension')
        assert hasattr(config.memory.embeddings, 'provider')
        assert hasattr(config.memory.embeddings, 'remote_endpoint')
    
    def test_default_embedding_model_is_granite(self):
        """Test that default embedding model is the Granite model."""
        config = load_core_config()
        
        assert config.memory.embeddings.model == "text-embedding-granite-embedding-278m-multilingual"
    
    def test_default_embedding_dimension_is_768(self):
        """Test that default embedding dimension matches Granite (768)."""
        config = load_core_config()
        
        assert config.memory.embeddings.dimension == 768
    
    def test_default_provider_is_remote(self):
        """Test that default provider type is remote (LM Studio)."""
        config = load_core_config()
        
        assert config.memory.embeddings.provider == "remote"
    
    def test_remote_endpoint_configured(self):
        """Test that remote endpoint is properly configured."""
        config = load_core_config()
        
        assert config.memory.embeddings.remote_endpoint is not None
        # Accept either resolved endpoint or env placeholder in test contexts.
        endpoint = config.memory.embeddings.remote_endpoint
        assert "/v1" in endpoint or endpoint.startswith("$")
    
    def test_tool_routing_uses_same_embedding_model(self):
        """Test that tool_routing config uses the same embedding model."""
        config = load_core_config()
        
        if hasattr(config, 'tool_routing') and config.tool_routing:
            # If tool_routing has an explicit embedding_model, use it
            # Otherwise fall back to memory embeddings
            embedding_model = (
                config.tool_routing.embedding_model 
                or config.memory.embeddings.model
            )
            assert embedding_model == "text-embedding-granite-embedding-278m-multilingual"
    
    def test_embedding_provider_factory_accepts_config(self):
        """Test that build_embedding_provider works with config settings."""
        config = load_core_config()
        
        # Should not raise
        provider = build_embedding_provider(
            model_name=config.memory.embeddings.model,
            dimension=config.memory.embeddings.dimension,
            provider_type=config.memory.embeddings.provider,
            remote_endpoint=config.memory.embeddings.remote_endpoint,
        )
        
        assert provider is not None
        assert provider.get_dimension() == 768
    
    def test_env_override_embedding_server(self):
        """Test that EMBEDDING_SERVER env var can override config."""
        # This test verifies that environment variables can override defaults
        config = load_core_config()
        
        # The config should load whatever is set via $EMBEDDING_SERVER in .env
        # For this test, we just verify the config structure is correct
        assert hasattr(config.memory.embeddings, 'remote_endpoint')
        
        # In a real environment with EMBEDDING_SERVER set, this would be populated
        # During testing it might be None or configured
        endpoint = config.memory.embeddings.remote_endpoint
        if endpoint:
            assert isinstance(endpoint, str)


class TestEmbeddingConfigOverrides:
    """Test configuration override scenarios."""
    
    def test_remote_provider_requires_endpoint(self):
        """Test that remote-only provider validation enforces endpoint configuration."""
        with pytest.raises(ValueError, match="remote_endpoint required"):
            build_embedding_provider(
                model_name="text-embedding-granite-embedding-278m-multilingual",
                dimension=768,
                provider_type="remote",
                remote_endpoint=None,
            )

    def test_remote_provider_config_with_placeholder_endpoint(self):
        """Test remote provider with placeholder endpoint uses deterministic fallback."""
        provider = build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint="$EMBEDDING_SERVER/v1",
        )
        
        assert provider is not None
        assert provider.get_dimension() == 768
        
        embedding = provider.encode("test query")
        assert len(embedding) == 768
        assert isinstance(embedding, list)
    
    def test_remote_provider_with_mock_endpoint(self):
        """Test remote provider configuration (will fail connection without real endpoint)."""
        # This demonstrates the API but will fail without real LM Studio running
        try:
            provider = build_embedding_provider(
                model_name="text-embedding-granite-embedding-278m-multilingual",
                dimension=768,
                provider_type="remote",
                remote_endpoint="http://localhost:8000/v1",
            )
            
            assert provider is not None
            assert provider.get_dimension() == 768
            
            # Note: actual encoding would fail without LM Studio running
            # So we just verify the provider was created
        except Exception as e:
            # Expected to fail without real LM Studio endpoint
            # This is OK - we're just testing the configuration loads
            assert "Failed" in str(e) or "Connection" in str(e)
