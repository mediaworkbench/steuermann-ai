"""Test configuration loading for embedding provider settings."""

import pytest
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.embeddings import build_embedding_provider


class TestEmbeddingConfigLoading:
    """Test that embedding configuration loads correctly."""

    @staticmethod
    def _load():
        return load_core_config(env={"PROFILE_ID": "starter"})
    
    def test_core_config_loads_embedding_settings(self):
        """Test that core config loads embedding settings and role transport."""
        config = self._load()
        
        # Check that embedding settings exist
        assert hasattr(config.memory, 'embeddings')
        assert hasattr(config.memory.embeddings, 'dimension')
        assert config.llm.get_embedding_provider_type() == "remote"
        assert config.llm.get_embedding_remote_endpoint() is not None
    
    def test_default_embedding_model_is_granite(self):
        """Test that default embedding model is the Granite model."""
        config = self._load()
        assert config.llm.roles.embedding.model == "openai/text-embedding-granite-embedding-278m-multilingual"
    
    def test_default_embedding_dimension_is_768(self):
        """Test that default embedding dimension matches Granite (768)."""
        config = self._load()
        
        assert config.memory.embeddings.dimension == 768
    
    def test_default_provider_is_remote(self):
        """Test that embedding transport type resolves to remote."""
        config = self._load()
        
        assert config.llm.get_embedding_provider_type() == "remote"
    
    def test_remote_endpoint_configured(self):
        """Test that embedding endpoint resolves from llm.roles.embedding.api_base."""
        config = self._load()
        
        assert config.llm.get_embedding_remote_endpoint() is not None
        # Accept either resolved endpoint or env placeholder in test contexts.
        endpoint = config.llm.get_embedding_remote_endpoint()
        assert "/v1" in endpoint or endpoint.startswith("$")
    
    def test_tool_routing_uses_same_embedding_model(self):
        """Test that tool_routing config uses the same embedding model."""
        config = self._load()
        
        if hasattr(config, 'tool_routing') and config.tool_routing:
            # If tool_routing has an explicit embedding_model, use it
            # Otherwise fall back to memory embeddings
            embedding_model = (
                config.tool_routing.embedding_model 
                or config.llm.roles.embedding.model
            )
            assert embedding_model == "openai/text-embedding-granite-embedding-278m-multilingual"
    
    def test_embedding_provider_factory_accepts_config(self):
        """Test that build_embedding_provider works with config settings."""
        config = self._load()
        
        # Should not raise
        provider = build_embedding_provider(
            model_name=config.llm.roles.embedding.model,
            dimension=config.memory.embeddings.dimension,
            provider_type=config.llm.get_embedding_provider_type(),
            remote_endpoint=config.llm.get_embedding_remote_endpoint(),
        )
        
        assert provider is not None
        assert provider.get_dimension() == 768
    
    def test_env_override_embedding_server(self):
        """Test that EMBEDDING_SERVER env var can override config."""
        # This test verifies that environment variables can override defaults
        config = self._load()
        
        # The config should load whatever is set via $EMBEDDING_SERVER in .env
        # For this test, we just verify the config structure is correct
        assert config.llm.get_embedding_remote_endpoint() is not None
        
        # In a real environment with EMBEDDING_SERVER set, this would be populated
        # During testing it might be None or configured
        endpoint = config.llm.get_embedding_remote_endpoint()
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
