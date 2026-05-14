from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.memory import Mem0MemoryBackend, InMemoryMemoryManager
from universal_agentic_framework.config.schemas import (
    CoreConfig,
    IngestionSettings,
    ForkSettings,
    LLMRoleSettings,
    LLMRoles,
    LLMSettings,
    LLMProviders,
    DatabaseSettings,
    MemorySettings,
    VectorStoreSettings,
    EmbeddingSettings,
    RetentionSettings,
    TokensSettings,
)


class _FakeClient:
    def add(self, *args, **kwargs):
        return {"id": "mem_1"}

    def search(self, *args, **kwargs):
        return {"results": []}

    def get_all(self, *args, **kwargs):
        return {"results": []}

    def delete_all(self, *args, **kwargs):
        return {"status": "ok"}

    def get(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return {"status": "ok"}


class _FakeEmbedder:
    def encode(self, texts):
        # Return 384-dim vectors filled with zeros
        return [[0.0] * 384 for _ in texts]


def _minimal_llm_settings() -> LLMSettings:
    return LLMSettings(
        providers=LLMProviders(),
        roles=LLMRoles(
            chat=LLMRoleSettings(provider_id="ollama", api_base="http://localhost:11434/v1", model="ollama/llama-3.1-8b"),
            embedding=LLMRoleSettings(provider_id="ollama", api_base="http://localhost:11434/v1", model="ollama/nomic-embed-text"),
            vision=LLMRoleSettings(provider_id="ollama", api_base="http://localhost:11434/v1", model="ollama/llama-3.1-8b"),
            auxiliary=LLMRoleSettings(provider_id="ollama", api_base="http://localhost:11434/v1", model="ollama/llama-3.1-8b"),
        ),
    )


def _base_config(vs_type: str) -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="test", language="en"),
        llm=_minimal_llm_settings(),
        database=DatabaseSettings(url="postgresql://x:y@localhost:5432/db"),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type=vs_type, host="qdrant", port=6333, collection_prefix="test"),
            embeddings=EmbeddingSettings(dimension=384),
            retention=RetentionSettings(),
        ),
        tokens=TokensSettings(default_budget=1000, per_node_budgets={}),
        ingestion=IngestionSettings(collection_name="test"),
    )


def test_build_memory_backend_mem0_returns_mem0_backend():
    cfg = _base_config("mem0")
    backend = build_memory_backend(cfg, client=_FakeClient(), embedder=_FakeEmbedder())
    assert isinstance(backend, Mem0MemoryBackend)


def test_build_memory_backend_non_mem0_returns_in_memory():
    cfg = _base_config("mem0")
    cfg.memory.vector_store.type = "legacy"
    backend = build_memory_backend(cfg)
    assert isinstance(backend, InMemoryMemoryManager)
