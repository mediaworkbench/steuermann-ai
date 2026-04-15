from types import SimpleNamespace

import pytest

from universal_agentic_framework.config.schemas import (
    CoreConfig,
    DatabaseSettings,
    EmbeddingSettings,
    ForkSettings,
    LLMProviders,
    LLMSettings,
    MemorySettings,
    ProviderModelMap,
    ProviderSettings,
    RetentionSettings,
    TokensSettings,
    VectorStoreSettings,
)
from universal_agentic_framework.llm.factory import LLMFactory, ModelSelection
from universal_agentic_framework.orchestration.graph_builder import (
    _ModelInvokeError,
    _invoke_with_model_fallback,
)


class _FailingModel:
    def invoke(self, _payload):
        raise RuntimeError("primary invoke failed")


class _WorkingModel:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload):
        return SimpleNamespace(content=self.text)


def _core_config() -> CoreConfig:
    return CoreConfig(
        fork=ForkSettings(name="starter", language="en"),
        llm=LLMSettings(
            providers=LLMProviders(
                primary=ProviderSettings(
                    type="mock",
                    endpoint=None,
                    models=ProviderModelMap(en="primary-model"),
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
                fallback=ProviderSettings(
                    type="mock",
                    endpoint=None,
                    models=ProviderModelMap(en="fallback-model"),
                    temperature=0.1,
                    max_tokens=256,
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


def test_invoke_with_model_fallback_uses_next_candidate(monkeypatch):
    config = _core_config()

    def _fake_candidates(self, language, preferred_model=None, prefer_local=True, include_default_when_preferred=True):
        del self, language, preferred_model, prefer_local, include_default_when_preferred
        return [
            ModelSelection(
                model=_WorkingModel("fallback response"),
                provider_type="mock",
                model_name="fallback-model",
                endpoint=None,
                source="fallback_default",
            )
        ]

    monkeypatch.setattr(LLMFactory, "get_model_candidates", _fake_candidates)

    text, provider, model_name, _ = _invoke_with_model_fallback(
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
                endpoint=None,
                source="fallback_default",
            )
        ]

    monkeypatch.setattr(LLMFactory, "get_model_candidates", _fake_candidates)

    with pytest.raises(_ModelInvokeError) as exc_info:
        _invoke_with_model_fallback(
            config=config,
            language="en",
            payload="hello",
            initial_model=_FailingModel(),
            initial_provider="mock",
            initial_model_name="primary-model",
            preferred_model=None,
        )

    assert exc_info.value.provider == "mock"
    assert exc_info.value.model_name == "fallback-model"
