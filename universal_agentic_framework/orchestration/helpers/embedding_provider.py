"""Embedding provider management for tool routing."""

from typing import Any, Optional, Tuple


# Module-level cache for embedding provider (to avoid reinitialization)
_embedding_provider_cache: Optional[Any] = None
_embedding_provider_config_key: Optional[str] = None


def get_routing_embedding_provider(config: Any, logger: Any = None) -> Tuple[Any, str]:
    """Return cached embedding provider and model name used for tool routing.
    
    The provider is cached to avoid re-instantiation on every graph step.
    Cache is invalidated if the embedding configuration changes.
    
    Args:
        config: Configuration object
        logger: Optional logger for debug output
        
    Returns:
        Tuple of (embedding_provider, embedding_model_name)
    """
    from universal_agentic_framework.embeddings import build_embedding_provider
    
    global _embedding_provider_cache, _embedding_provider_config_key
    
    embedding_model_name = (
        getattr(getattr(config, "tool_routing", None), "embedding_model", None)
        or config.memory.embeddings.model
    )
    embedding_dimension = config.memory.embeddings.dimension
    embedding_provider_type = config.memory.embeddings.provider
    embedding_remote_endpoint = config.memory.embeddings.remote_endpoint

    config_key = f"{embedding_model_name}:{embedding_provider_type}:{embedding_remote_endpoint}"

    if _embedding_provider_cache is None or _embedding_provider_config_key != config_key:
        if logger:
            logger.info(
                f"Loading embedding provider (first time): {embedding_model_name}",
                provider_type=embedding_provider_type,
            )
        _embedding_provider_cache = build_embedding_provider(
            model_name=embedding_model_name,
            dimension=embedding_dimension,
            provider_type=embedding_provider_type,
            remote_endpoint=embedding_remote_endpoint,
        )
        _embedding_provider_config_key = config_key

    return _embedding_provider_cache, embedding_model_name


def clear_embedding_cache():
    """Clear the embedding provider cache (for testing or config reloads)."""
    global _embedding_provider_cache, _embedding_provider_config_key
    _embedding_provider_cache = None
    _embedding_provider_config_key = None
