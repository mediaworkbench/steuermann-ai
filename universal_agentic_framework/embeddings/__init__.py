"""Embedding provider abstraction layer.

Supports remote (LM Studio/OpenAI-compatible) embeddings.
"""

from .provider import (
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    RemoteEmbeddingProvider,
    build_embedding_provider,
    normalize_embedding_model_name,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderUnavailableError",
    "RemoteEmbeddingProvider",
    "build_embedding_provider",
    "normalize_embedding_model_name",
]
