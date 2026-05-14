from types import SimpleNamespace

import pytest

from universal_agentic_framework.config.schemas import (
    CoreConfig,
    DatabaseSettings,
    EmbeddingSettings,
    ForkSettings,
    IngestionSettings,
    LLMRoleSettings,
    LLMRoles,
    LLMProviders,
    LLMSettings,
    MemorySettings,
    ProviderModelMap,
    ProviderSettings,
    RetentionSettings,
    RoleProviderRef,
    TokensSettings,
    VectorStoreSettings,
)
from universal_agentic_framework.llm.factory import LLMFactory, ModelSelection
from universal_agentic_framework.orchestration.helpers.model_resolution import invoke_with_model_fallback


class _ModelInvokeError(RuntimeError):
    def __init__(self, message: str, provider: str, model_name: str):
        super().__init__(message)
        self.provider = provider
        self.model_name = model_name


class _FailingModel:
    def invoke(self, _payload):
        raise RuntimeError("primary invoke failed")


class _WorkingModel:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload):
        return SimpleNamespace(content=self.text)


class _ListContentModel:
    def invoke(self, _payload):
        return SimpleNamespace(
            content=[
                {"type": "text", "text": "Hallo"},
                {"type": "text", "text": "Welt"},
            ]
        )


def _core_config() -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="starter", language="en"),
        llm=LLMSettings(
            providers=LLMProviders(
                lmstudio=ProviderSettings(
                    api_base=None,
                    api_key=None,
                    models=ProviderModelMap(en="openai/primary-model"),
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
                openrouter=ProviderSettings(
                    api_base=None,
                    api_key=None,
                    models=ProviderModelMap(en="openai/fallback-model"),
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
            ),
            roles=LLMRoles(
                chat=LLMRoleSettings(
                    providers=[
                        RoleProviderRef(provider_id="lmstudio"),
                        RoleProviderRef(provider_id="openrouter"),
                    ],
                    config_only=False,
                ),
                embedding=LLMRoleSettings(providers=[RoleProviderRef(provider_id="lmstudio")], config_only=True),
                vision=LLMRoleSettings(providers=[RoleProviderRef(provider_id="lmstudio")], config_only=True),
                auxiliary=LLMRoleSettings(providers=[RoleProviderRef(provider_id="lmstudio")], config_only=True),
            ),
        ),
        database=DatabaseSettings(url="postgresql://user:pass@localhost:5432/db", pool_size=10, echo=False),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type="mem0", host="qdrant", port=6333, collection_prefix="starter"),
            embeddings=EmbeddingSettings(model="model", dimension=384, batch_size=32),
            retention=RetentionSettings(session_memory_days=90, user_memory_days=365),
        ),
        tokens=TokensSettings(default_budget=10000, per_node_budgets={}),
        ingestion=IngestionSettings(collection_name="starter-rag"),
    )


def test_invoke_with_model_fallback_uses_next_candidate(monkeypatch):
    config = _core_config()

    def _fake_candidates(self, language, preferred_model=None, prefer_local=True, include_default_when_preferred=True):
        del self, language, preferred_model, prefer_local, include_default_when_preferred
        return [
            ModelSelection(
                model=_WorkingModel("fallback response"),
                provider_type="mock",
                model_name="fallback-model",
                api_base=None,
                source="fallback_default",
            )
        ]

    monkeypatch.setattr(LLMFactory, "get_model_candidates", _fake_candidates)

    text, provider, model_name, _ = invoke_with_model_fallback(
        config=config,
        language="en",
        payload="hello",
        initial_model=_FailingModel(),
        initial_provider="mock",
        initial_model_name="primary-model",
        preferred_model=None,
    )

    assert text == "fallback response"
    assert provider == "mock"
    assert model_name == "fallback-model"


def test_invoke_with_model_fallback_raises_with_last_attempt_metadata(monkeypatch):
    config = _core_config()

    def _fake_candidates(self, language, preferred_model=None, prefer_local=True, include_default_when_preferred=True):
        del self, language, preferred_model, prefer_local, include_default_when_preferred
        return [
            ModelSelection(
                model=_FailingModel(),
                provider_type="mock",
                model_name="fallback-model",
                api_base=None,
                source="fallback_default",
            )
        ]

    monkeypatch.setattr(LLMFactory, "get_model_candidates", _fake_candidates)

    with pytest.raises(_ModelInvokeError) as exc_info:
        invoke_with_model_fallback(
            config=config,
            language="en",
            payload="hello",
            initial_model=_FailingModel(),
            initial_provider="mock",
            initial_model_name="primary-model",
            preferred_model=None,
            error_cls=_ModelInvokeError,
        )

    assert exc_info.value.provider == "mock"
    assert exc_info.value.model_name == "fallback-model"


def test_invoke_with_model_fallback_normalizes_list_content_blocks():
    config = _core_config()

    text, provider, model_name, _ = invoke_with_model_fallback(
        config=config,
        language="en",
        payload="hello",
        initial_model=_ListContentModel(),
        initial_provider="openrouter",
        initial_model_name="openrouter/poolside/laguna-m.1:free",
        preferred_model=None,
    )

    assert text == "Hallo\nWelt"
    assert provider == "openrouter"
    assert model_name == "openrouter/poolside/laguna-m.1:free"
