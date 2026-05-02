from __future__ import annotations

from typing import Dict, Optional, Tuple

import structlog

from universal_agentic_framework.config import CoreConfig
from . import InMemoryMemoryManager, Mem0MemoryBackend, MemoryBackend

logger = structlog.get_logger(__name__)

# Module-level singleton cache: avoids reinitializing Memory.from_config() and
# triggering Qdrant index creation round-trips on every graph node invocation.
# Keyed on config fields that affect backend identity. Bypassed when test
# client/embedder overrides are injected (to preserve test isolation).
_backend_cache: Dict[Tuple, MemoryBackend] = {}


def _cache_key(config: CoreConfig) -> Tuple:
    vs = config.memory.vector_store
    emb = config.memory.embeddings
    llm = config.llm.providers.primary
    mem0 = config.memory.mem0
    return (
        vs.type,
        vs.host,
        vs.port,
        vs.collection_prefix,
        emb.model,
        emb.dimension,
        emb.remote_endpoint,
        str(llm.api_base),
        mem0.llm_provider,
        mem0.search_limit,
    )


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
    When test overrides are provided the cache is bypassed to preserve isolation.
    """
    vs = config.memory.vector_store
    emb = config.memory.embeddings

    if vs.type.lower() == "mem0":
        llm_primary = config.llm.providers.primary
        mem0_settings = config.memory.mem0

        # Bypass cache when test overrides are present.
        if client is None and embedder is None:
            key = _cache_key(config)
            if key in _backend_cache:
                return _backend_cache[key]

        backend = Mem0MemoryBackend(
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
            llm_provider=mem0_settings.llm_provider,
            client=client,
            embedder=embedder,
        )

        if client is None and embedder is None:
            key = _cache_key(config)
            _backend_cache[key] = backend
            logger.info("memory_backend_cached", backend_type=type(backend).__name__)

        return backend

    return InMemoryMemoryManager()
