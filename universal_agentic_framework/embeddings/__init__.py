"""Embedding provider abstraction layer.

Supports remote (LM Studio/OpenAI-compatible) embeddings.
"""

from .provider import EmbeddingProvider, RemoteEmbeddingProvider, build_embedding_provider

__all__ = [
    "EmbeddingProvider",
    "RemoteEmbeddingProvider",
    "build_embedding_provider",
]
