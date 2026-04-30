from universal_agentic_framework.memory.factory import build_memory_backend
from universal_agentic_framework.memory import QdrantMemoryBackend, InMemoryMemoryManager
from universal_agentic_framework.config.schemas import (
    CoreConfig,
    ForkSettings,
    LLMSettings,
    LLMProviders,
    ProviderSettings,
    DatabaseSettings,
    MemorySettings,
    VectorStoreSettings,
    EmbeddingSettings,
    RetentionSettings,
    TokensSettings,
)


class _FakeClient:
    def get_collection(self, name):
        return None

    def create_collection(self, collection_name, vectors_config):
        pass

    def upsert(self, *args, **kwargs):
        pass

    def search(self, *args, **kwargs):
        return []

    def scroll(self, *args, **kwargs):
        return [], None


class _FakeEmbedder:
    def encode(self, texts):
        # Return 384-dim vectors filled with zeros
        return [[0.0] * 384 for _ in texts]


def _minimal_llm_settings() -> LLMSettings:
    primary = ProviderSettings(
        api_base=None,
        api_key=None,
        models={"en": "ollama/llama-3.1-8b"},
        temperature=0.3,
    )
    return LLMSettings(providers=LLMProviders(primary=primary))


def _base_config(vs_type: str) -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="test", language="en"),
        llm=_minimal_llm_settings(),
        database=DatabaseSettings(url="postgresql://x:y@localhost:5432/db"),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type=vs_type, host="qdrant", port=6333, collection_prefix="test"),
            embeddings=EmbeddingSettings(model="paraphrase-multilingual-MiniLM-L12-v2", dimension=384),
            retention=RetentionSettings(),
        ),
        tokens=TokensSettings(default_budget=1000, per_node_budgets={}),
    )


def test_build_memory_backend_qdrant_returns_qdrant_backend():
    cfg = _base_config("qdrant")
    backend = build_memory_backend(cfg, client=_FakeClient(), embedder=_FakeEmbedder())
    assert isinstance(backend, QdrantMemoryBackend)


def test_build_memory_backend_non_qdrant_returns_in_memory():
    cfg = _base_config("otherstore")
    backend = build_memory_backend(cfg)
    assert isinstance(backend, InMemoryMemoryManager)
