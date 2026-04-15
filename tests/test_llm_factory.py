from universal_agentic_framework.config.schemas import CoreConfig, LLMSettings, LLMProviders, ProviderModelMap, ProviderSettings, ForkSettings, DatabaseSettings, MemorySettings, VectorStoreSettings, EmbeddingSettings, RetentionSettings, TokensSettings
from universal_agentic_framework.llm.factory import LLMFactory


class DummyModel:
    def __init__(self, label: str):
        self.label = label

    def invoke(self, text: str):  # pragma: no cover - not used
        return f"{self.label}:{text}"


def _core_config(primary_model: str = "local-model", fallback_model: str = "remote-model") -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="starter", language="en"),
        llm=LLMSettings(
            providers=LLMProviders(
                primary=ProviderSettings(
                    type="mock",
                    endpoint=None,
                    models=ProviderModelMap(en=primary_model),
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
                fallback=ProviderSettings(
                    type="mock",
                    endpoint=None,
                    models=ProviderModelMap(en=fallback_model),
                    temperature=0.2,
                    max_tokens=512,
                    timeout=30,
                ),
            )
        ),
        database=DatabaseSettings(url="postgresql://user:pass@localhost:5432/db", pool_size=10, echo=False),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type="qdrant", host="qdrant", port=6333, collection_prefix="starter"),
            embeddings=EmbeddingSettings(model="model", dimension=384, batch_size=32),
            retention=RetentionSettings(session_memory_days=90, user_memory_days=365),
        ),
        tokens=TokensSettings(default_budget=10000, per_node_budgets={}),
    )


def test_llm_factory_primary_selection():
    factory = LLMFactory(
        config=_core_config(),
        builders={"mock": lambda provider, model: DummyModel(f"{provider.type}:{model}")},
    )
    model = factory.get_model(language="en")
    assert isinstance(model, DummyModel)
    assert model.label == "mock:local-model"


def test_llm_factory_fallback_when_language_missing():
    factory = LLMFactory(
        config=_core_config(primary_model=None, fallback_model="remote-model"),
        builders={"mock": lambda provider, model: DummyModel(f"{provider.type}:{model}")},
    )
    model = factory.get_model(language="de")  # de missing in primary -> fallback used
    assert model.label == "mock:remote-model"


def test_llm_factory_model_selection_metadata():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model", fallback_model="remote-model"),
        builders={"mock": lambda provider, model: DummyModel(f"{provider.type}:{model}")},
    )

    selection = factory.get_model_selection(language="en")
    assert isinstance(selection.model, DummyModel)
    assert selection.provider_type == "mock"
    assert selection.model_name == "local-model"
    assert selection.source == "primary_default"


def test_llm_factory_candidate_order_with_preferred_model():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model", fallback_model="remote-model"),
        builders={"mock": lambda provider, model: DummyModel(f"{provider.type}:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="preferred-model",
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
        "primary_default",
        "fallback_preferred",
        "fallback_default",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "preferred-model",
        "local-model",
        "preferred-model",
        "remote-model",
    ]


def test_llm_factory_candidate_order_with_preferred_model_strict():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model", fallback_model="remote-model"),
        builders={"mock": lambda provider, model: DummyModel(f"{provider.type}:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="preferred-model",
        include_default_when_preferred=False,
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
        "fallback_preferred",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "preferred-model",
        "preferred-model",
    ]
