"""Provider and model normalization helpers for LiteLLM routing."""
from __future__ import annotations

from dataclasses import dataclass


_PROVIDER_ALIASES = {
    "lmstudio": "lm_studio",
}


@dataclass(frozen=True)
class ProviderModelId:
    """Parsed provider/model identifier."""

    provider: str
    model: str


def normalize_provider_name(provider_name: str) -> str:
    normalized = provider_name.strip().lower()
    if not normalized:
        raise ValueError("provider name must not be empty")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def normalize_model_id(model_name: str) -> str:
    """Normalize model IDs while preserving LiteLLM provider prefixes."""
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError("model name must not be empty")

    if "/" not in normalized:
        raise ValueError(
            "model name must include a LiteLLM provider prefix like 'openai/...', 'lm_studio/...', or 'ollama/...'")

    provider_name, model_id = normalized.split("/", 1)
    provider_name = normalize_provider_name(provider_name)
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model name must include a model identifier after the provider prefix")
    return f"{provider_name}/{model_id}"


def parse_model_id(model_name: str) -> ProviderModelId:
    normalized = normalize_model_id(model_name)
    provider_name, model_id = normalized.split("/", 1)
    return ProviderModelId(provider=provider_name, model=model_id)
