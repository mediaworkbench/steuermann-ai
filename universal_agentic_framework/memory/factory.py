from __future__ import annotations

from typing import Optional

from universal_agentic_framework.config import CoreConfig
from . import InMemoryMemoryManager, QdrantMemoryBackend, MemoryBackend


def build_memory_backend(
    config: CoreConfig,
    *,
    client: Optional[object] = None,
    embedder: Optional[object] = None,
) -> MemoryBackend:
    """Build memory backend from CoreConfig.

    - If vector_store.type is 'qdrant', returns QdrantMemoryBackend configured
      from `config.memory` settings (host, port, collection_prefix, embedding model, dimension).
    - Otherwise returns an in-memory backend.

    Fake/stub `client` and `embedder` can be injected for testing.
    """
    vs = config.memory.vector_store
    emb = config.memory.embeddings

    if vs.type.lower() == "qdrant":
        return QdrantMemoryBackend(
            host=vs.host,
            port=vs.port,
            collection_prefix=vs.collection_prefix,
            embedding_model=emb.model,
            client=client,
            embedder=embedder,
            distance="Cosine",
            dimension=emb.dimension,
            embedding_provider_type=emb.provider,
            embedding_remote_endpoint=emb.remote_endpoint,
        )

    return InMemoryMemoryManager()
