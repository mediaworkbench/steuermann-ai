"""Tests for expanded probe coverage across roles (chat, vision, auxiliary)."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace

from backend.llm_capability_probe import LLMCapabilityProbeRunner


@pytest.fixture
def mock_core_config_with_roles():
    """Create a mock config with chat, vision, and auxiliary roles configured."""
    config = MagicMock()
    
    # Setup providers
    lmstudio_provider = MagicMock()
    lmstudio_provider.models = SimpleNamespace(
        en="openai/liquid/lfm2-24b-a2b",
        de="openai/liquid/lfm2-24b-a2b"
    )
    lmstudio_provider.get_tool_calling_mode = MagicMock(return_value="structured")
    
    ollama_provider = MagicMock()
    ollama_provider.models = SimpleNamespace(
        en="ollama/llama-3.1-8b",
        de="ollama/llama-3.1-8b"
    )
    ollama_provider.get_tool_calling_mode = MagicMock(return_value="native")
    
    config.llm.get_role_provider_chain_with_models = MagicMock(
        side_effect=lambda role_name, _lang: {
            "chat": [
                ("lmstudio", lmstudio_provider, "openai/liquid/lfm2-24b-a2b"),
                ("ollama", ollama_provider, "ollama/llama-3.1-8b"),
            ],
            "vision": [
                ("lmstudio", lmstudio_provider, "openai/liquid/lfm2-24b-a2b"),
            ],
            "auxiliary": [
                ("lmstudio", lmstudio_provider, "openai/liquid/lfm2-24b-a2b"),
            ],
        }.get(role_name, [])
    )
    config.llm.roles = SimpleNamespace(chat=object(), vision=object(), auxiliary=object())
    config.fork.language = "en"
    
    return config


def test_probe_runner_covers_all_roles_except_embedding(mock_core_config_with_roles):
    """Verify probe runner collects targets from chat, vision, and auxiliary roles."""
    runner = LLMCapabilityProbeRunner(
        core_config=mock_core_config_with_roles,
        profile_id="test-profile"
    )
    
    targets = runner._collect_targets()
    
    # Should have targets from:
    # - chat: lmstudio, ollama
    # - vision: lmstudio (duplicate, should be deduplicated)
    # - auxiliary: lmstudio (duplicate, should be deduplicated)
    # Total: 2 unique (provider_id, model_name) pairs
    assert len(targets) == 2
    
    provider_model_pairs = {(t.provider_id, t.model_name) for t in targets}
    assert ("lmstudio", "openai/liquid/lfm2-24b-a2b") in provider_model_pairs
    assert ("ollama", "ollama/llama-3.1-8b") in provider_model_pairs


def test_probe_runner_skips_embedding_role(mock_core_config_with_roles):
    """Verify embedding role is not included in probe coverage."""
    # Add embedding role marker to config; probe runner should still ignore it.
    mock_core_config_with_roles.llm.roles.embedding = object()
    
    runner = LLMCapabilityProbeRunner(
        core_config=mock_core_config_with_roles,
        profile_id="test-profile"
    )
    
    targets = runner._collect_targets()
    
    # Should still have only 2 targets (embedding should be skipped)
    assert len(targets) == 2
    
    provider_model_pairs = {(t.provider_id, t.model_name) for t in targets}
    assert ("lmstudio", "openai/liquid/lfm2-24b-a2b") in provider_model_pairs
    assert ("ollama", "ollama/llama-3.1-8b") in provider_model_pairs


def test_collect_all_models_extracts_distinct_models():
    """Verify _collect_all_models gets all distinct model names from a provider."""
    from types import SimpleNamespace as Namespace
    
    provider = MagicMock()
    provider.models = Namespace(
        en="model-en-v1",
        de="model-de-v1",
        fr="model-en-v1",  # Duplicate of English
        es=""  # Empty, should be skipped
    )
    
    models = LLMCapabilityProbeRunner._collect_all_models(provider)
    
    assert len(models) == 2
    assert "model-en-v1" in models
    assert "model-de-v1" in models


def test_probe_runner_handles_multi_language_models():
    """Verify probe runner creates targets for each distinct model across languages."""
    config = MagicMock()
    
    # Provider with different models per language
    lmstudio_provider = MagicMock()
    lmstudio_provider.models = SimpleNamespace(
        en="openai/liquid/lfm2-24b-a2b",
        de="openai/liquid/lfm2-40b-de",
        fr="openai/liquid/lfm2-40b-de"  # Same as German
    )
    lmstudio_provider.get_tool_calling_mode = MagicMock(return_value="structured")
    
    config.llm.get_role_provider_chain_with_models = MagicMock(
        side_effect=lambda role_name, _lang: {
            "chat": [("lmstudio", lmstudio_provider, "openai/liquid/lfm2-24b-a2b")],
            "vision": [("lmstudio", lmstudio_provider, "openai/liquid/lfm2-40b-de")],
            "auxiliary": [],
        }.get(role_name, [])
    )
    config.llm.roles = SimpleNamespace(chat=object(), vision=object(), auxiliary=object())
    config.fork.language = "en"
    
    runner = LLMCapabilityProbeRunner(core_config=config, profile_id="test-profile")
    targets = runner._collect_targets()
    
    # Should have 2 distinct models (duplicates deduplicated):
    # - openai/liquid/lfm2-24b-a2b (from en)
    # - openai/liquid/lfm2-40b-de (from de)
    assert len(targets) == 2
    
    models = {t.model_name for t in targets}
    assert "openai/liquid/lfm2-24b-a2b" in models
    assert "openai/liquid/lfm2-40b-de" in models


def test_probe_runner_includes_role_level_model_override():
    """Verify role provider ref model override is included in probe targets."""
    config = MagicMock()

    provider = MagicMock()
    provider.models = SimpleNamespace(en="openai/provider-default")
    provider.get_tool_calling_mode = MagicMock(return_value="structured")

    config.llm.get_role_provider_chain_with_models = MagicMock(
        side_effect=lambda role_name, _lang: {
            "chat": [("lmstudio", provider, "openai/role-specific")],
            "vision": [],
            "auxiliary": [],
        }.get(role_name, [])
    )
    config.llm.roles = SimpleNamespace(chat=object(), vision=object(), auxiliary=object())

    runner = LLMCapabilityProbeRunner(core_config=config, profile_id="test-profile")
    targets = runner._collect_targets()

    models = {t.model_name for t in targets}
    assert "openai/role-specific" in models
