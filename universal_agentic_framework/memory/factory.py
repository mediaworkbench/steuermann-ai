from __future__ import annotations

from typing import Optional

from universal_agentic_framework.config import CoreConfig
from . import InMemoryMemoryManager, Mem0MemoryBackend, MemoryBackend


def build_memory_backend(
    config: CoreConfig,
    *,
    client: Optional[object] = None,
    embedder: Optional[object] = None,
) -> MemoryBackend:
    """Build memory backend from CoreConfig.

    Production path is Mem0 OSS (embedded mode with Qdrant vector storage).
    In-memory backend remains available for testing and emergency fallback.

    Fake/stub `client` and `embedder` can be injected for testing.
    """
    vs = config.memory.vector_store
    emb = config.memory.embeddings

    if vs.type.lower() == "mem0":
        llm_primary = config.llm.providers.primary
        mem0_settings = config.memory.mem0
        return Mem0MemoryBackend(
            host=vs.host,
            port=vs.port,
            collection_prefix=vs.collection_prefix,
            embedding_model=emb.model,
            dimension=emb.dimension,
            embedding_remote_endpoint=emb.remote_endpoint,
            llm_model=str(llm_primary.models.model_dump().get(config.fork.language) or llm_primary.models.en or "openai/gpt-4o-mini"),
            llm_api_base=str(llm_primary.api_base) if llm_primary.api_base else None,
            llm_temperature=float(llm_primary.temperature or 0.0),
            llm_max_tokens=int(llm_primary.max_tokens) if llm_primary.max_tokens else None,
            llm_api_key=llm_primary.api_key,
            search_limit=mem0_settings.search_limit,
            custom_instructions=mem0_settings.custom_instructions,
            client=client,
            embedder=embedder,
        )

    return InMemoryMemoryManager()
