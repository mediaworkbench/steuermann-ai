from universal_agentic_framework.config.schemas import (
    CoreConfig,
    DatabaseSettings,
    EmbeddingSettings,
    ForkSettings,
    IngestionSettings,
    LLMRoleSettings,
    LLMRoles,
    LLMSettings,
    LLMProviders,
    ProviderModelMap,
    ProviderSettings,
    RetentionSettings,
    TokensSettings,
    VectorStoreSettings,
    MemorySettings,
)
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.llm.provider_registry import normalize_model_id, parse_model_id


class DummyModel:
    def __init__(self, label: str):
        self.label = label

    def invoke(self, text: str):  # pragma: no cover - not used
        return f"{self.label}:{text}"


def _core_config(primary_model: str = "local-model") -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="starter", language="en"),
        llm=LLMSettings(
            providers=LLMProviders(),
            roles=LLMRoles(
                chat=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
                embedding=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model="openai/text-embedding-granite-embedding-278m-multilingual",
                ),
                vision=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                ),
                auxiliary=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                ),
            ),
        ),
        database=DatabaseSettings(url="postgresql://user:pass@localhost:5432/db", pool_size=10, echo=False),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type="mem0", host="qdrant", port=6333, collection_prefix="starter"),
            embeddings=EmbeddingSettings(dimension=384, batch_size=32),
            retention=RetentionSettings(session_memory_days=90, user_memory_days=365),
        ),
        tokens=TokensSettings(default_budget=10000, per_node_budgets={}),
        ingestion=IngestionSettings(collection_name="starter-rag"),
    )


def test_llm_factory_primary_selection():
    factory = LLMFactory(
        config=_core_config(primary_model="openai/local-model"),
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )
    model = factory.get_model(language="en")
    assert isinstance(model, DummyModel)
    assert model.label == "mock:openai/local-model"


def test_llm_factory_uses_primary_for_any_language():
    factory = LLMFactory(
        config=_core_config(primary_model="openai/remote-model"),
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )
    model = factory.get_model(language="de")
    assert model.label == "mock:openai/remote-model"


def test_llm_factory_model_selection_metadata():
    factory = LLMFactory(
        config=_core_config(primary_model="openai/local-model"),
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    selection = factory.get_model_selection(language="en")
    assert isinstance(selection.model, DummyModel)
    assert selection.provider_type == "openai"
    assert selection.model_name == "openai/local-model"
    assert selection.source == "primary_default"


def test_llm_factory_candidate_order_with_preferred_model():
    factory = LLMFactory(
        config=_core_config(primary_model="openai/local-model"),
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="openai/preferred-model",
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
        "primary_default",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "openai/preferred-model",
        "openai/local-model",
    ]


def test_llm_factory_candidate_order_with_preferred_model_strict():
    factory = LLMFactory(
        config=_core_config(primary_model="openai/local-model"),
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="openai/preferred-model",
        include_default_when_preferred=False,
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "openai/preferred-model",
    ]


def test_llm_factory_prefers_role_level_model_override():
    config = _core_config(primary_model="openai/provider-default")
    config.llm.roles.chat.model = "openai/chat-role-override"

    factory = LLMFactory(
        config=config,
        builders={"mock": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    model = factory.get_model(language="en")
    assert isinstance(model, DummyModel)
    assert model.label == "mock:openai/chat-role-override"


def test_provider_settings_normalizes_known_provider_aliases():
    settings = ProviderSettings(
        api_base=None,
        api_key=None,
        models=ProviderModelMap(en="lmstudio/liquid/lfm2-24b-a2b"),
    )

    assert settings.models.en == "lm_studio/liquid/lfm2-24b-a2b"
    parsed = parse_model_id(settings.models.en)
    assert parsed.provider == "lm_studio"
    assert parsed.model == "liquid/lfm2-24b-a2b"


def test_normalize_model_id_accepts_openai_prefixed_model():
    assert normalize_model_id("openai/liquid/lfm2-24b-a2b") == "openai/liquid/lfm2-24b-a2b"


def test_parse_model_id_accepts_openai_prefixed_model():
    parsed = parse_model_id("openai/liquid/lfm2-24b-a2b")
    assert parsed.provider == "openai"
    assert parsed.model == "liquid/lfm2-24b-a2b"


# ---------------------------------------------------------------------------
# Phase J — LiteLLM router contract tests
# ---------------------------------------------------------------------------

def test_get_router_model_returns_chat_litellm_router():
    """get_router_model() must return a ChatLiteLLMRouter wrapping a litellm.Router."""
    from langchain_litellm import ChatLiteLLMRouter
    from litellm import Router

    factory = LLMFactory(config=_core_config(
        primary_model="openai/primary-model",
    ))
    router_model = factory.get_router_model(language="en")

    assert isinstance(router_model, ChatLiteLLMRouter), (
        f"Expected ChatLiteLLMRouter, got {type(router_model)}"
    )
    assert isinstance(router_model.router, Router), (
        f"Expected litellm.Router inside ChatLiteLLMRouter, got {type(router_model.router)}"
    )


def test_get_router_model_primary_only():
    """get_router_model() must work with single role provider."""
    from langchain_litellm import ChatLiteLLMRouter

    config = _core_config(primary_model="openai/primary-model")

    factory = LLMFactory(config=config)
    router_model = factory.get_router_model(language="en")

    assert isinstance(router_model, ChatLiteLLMRouter)
    # Only one model in the Router's model_list
    assert len(router_model.router.model_list) == 1
    assert router_model.router.model_list[0]["model_name"] == "primary"


def test_get_router_model_uses_profile_router_settings():
    factory = LLMFactory(config=_core_config(
        primary_model="openai/primary-model",
    ))
    factory.config.llm.router.num_retries = 7
    factory.config.llm.router.retry_after = 4
    factory.config.llm.router.routing_strategy = "latency-based-routing"
    factory.config.llm.router.default_max_parallel_requests = 9

    router_model = factory.get_router_model(language="en")

    assert router_model.router.num_retries == 7
    assert router_model.router.retry_after == 4
    assert router_model.router.routing_strategy == "latency-based-routing"
    assert router_model.router.model_list[0]["litellm_params"]["max_parallel_requests"] == 9


def test_no_legacy_llm_provider_imports():
    """Codebase must not import langchain_openai or langchain_ollama directly."""
    import ast
    import os

    forbidden = {"langchain_openai", "langchain_ollama"}
    violations = []

    for root, _dirs, files in os.walk("universal_agentic_framework"):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                tree = ast.parse(open(path).read(), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = (
                        node.module if isinstance(node, ast.ImportFrom) else None
                    )
                    for alias in getattr(node, "names", []):
                        top = (module or alias.name).split(".")[0]
                        if top in forbidden:
                            violations.append(f"{path}: imports {top}")

    assert not violations, "Legacy provider imports found:\n" + "\n".join(violations)
