"""LLM factory with language-aware provider selection and optional fallback."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from universal_agentic_framework.config import CoreConfig
from universal_agentic_framework.config.schemas import ProviderSettings

BuilderFn = Callable[[ProviderSettings, str], object]


@dataclass(frozen=True)
class ModelSelection:
    """Resolved runtime model selection metadata."""

    model: object
    provider_type: str
    model_name: str
    endpoint: Optional[str]
    source: str


def build_ollama_chat(provider: ProviderSettings, model_name: str):
    try:
        from langchain_ollama import ChatOllama
    except Exception as exc:  # pragma: no cover - import errors surfaced in tests
        raise RuntimeError("ChatOllama not available") from exc
    return ChatOllama(
        base_url=str(provider.endpoint) if provider.endpoint else None,
        model=model_name,
        temperature=provider.temperature,
        num_predict=provider.max_tokens,
        timeout=provider.timeout,
    )


def build_openai_chat(provider: ProviderSettings, model_name: str):
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ChatOpenAI not available") from exc
    model_kwargs = {}
    if provider.tool_calling != "native":
        model_kwargs["tool_choice"] = "none"
    return ChatOpenAI(
        model=model_name,
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
        timeout=provider.timeout,
        base_url=str(provider.endpoint) if provider.endpoint else None,
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        model_kwargs=model_kwargs,
    )


def build_anthropic_chat(provider: ProviderSettings, model_name: str):
    try:
        from langchain_anthropic import ChatAnthropic
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ChatAnthropic not available") from exc
    return ChatAnthropic(
        model=model_name,
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
        timeout=provider.timeout,
    )


DEFAULT_BUILDERS: Dict[str, BuilderFn] = {
    "ollama": build_ollama_chat,
    "openai": build_openai_chat,
    "anthropic": build_anthropic_chat,
}


class LLMFactory:
    def __init__(self, config: CoreConfig, builders: Optional[Dict[str, BuilderFn]] = None):
        self.config = config
        self.builders = builders or DEFAULT_BUILDERS

    def _select_model(self, provider: ProviderSettings, language: str) -> str:
        # exact language
        model = getattr(provider.models, language, None)
        if model:
            return model
        # fallback to first non-empty
        for value in provider.models.__dict__.values():
            if value:
                return value
        raise ValueError(f"No model configured for language '{language}' and no fallback available")

    def _build(self, provider: ProviderSettings, model_name: str):
        builder = self.builders.get(provider.type)
        if not builder:
            raise ValueError(f"Unsupported provider type: {provider.type}")
        return builder(provider, model_name)

    def _build_selection(
        self,
        provider: ProviderSettings,
        model_name: str,
        source: str,
    ) -> ModelSelection:
        model = self._build(provider, model_name)
        endpoint = str(provider.endpoint) if provider.endpoint else None
        return ModelSelection(
            model=model,
            provider_type=provider.type,
            model_name=model_name,
            endpoint=endpoint,
            source=source,
        )

    def get_model_candidates(
        self,
        language: str,
        preferred_model: Optional[str] = None,
        prefer_local: bool = True,
        include_default_when_preferred: bool = True,
    ) -> list[ModelSelection]:
        """Build ordered runtime model candidates.

        Order:
        - preferred on primary (if provided)
        - default primary
        - preferred on fallback (if provided)
        - default fallback
        """
        providers = self.config.llm.providers
        candidates: list[ModelSelection] = []
        errors = []
        seen: set[tuple[str, str, str]] = set()

        def try_add(provider: Optional[ProviderSettings], source: str, model_override: Optional[str] = None):
            if not provider:
                return
            try:
                model_name = model_override or self._select_model(provider, language)
                key = (provider.type, model_name, source)
                if key in seen:
                    return
                selection = self._build_selection(provider, model_name, source)
                candidates.append(selection)
                seen.add(key)
            except Exception as exc:  # pragma: no cover - fallback path
                errors.append(exc)

        if prefer_local:
            if preferred_model:
                try_add(providers.primary, "primary_preferred", preferred_model)
            if (not preferred_model) or include_default_when_preferred:
                try_add(providers.primary, "primary_default")

        if preferred_model:
            try_add(providers.fallback, "fallback_preferred", preferred_model)
        if (not preferred_model) or include_default_when_preferred:
            try_add(providers.fallback, "fallback_default")

        if not candidates and errors:
            raise RuntimeError(f"All providers failed: {errors}")
        if not candidates:
            raise RuntimeError("No provider available")
        return candidates

    def get_model_selection(
        self,
        language: str,
        prefer_local: bool = True,
    ) -> ModelSelection:
        """Return the first available runtime model selection."""
        return self.get_model_candidates(
            language=language,
            preferred_model=None,
            prefer_local=prefer_local,
        )[0]

    def get_model(self, language: str, prefer_local: bool = True):
        return self.get_model_selection(language=language, prefer_local=prefer_local).model

    def get_model_with_config(
        self, language: str, prefer_local: bool = True
    ) -> Tuple[object, ProviderSettings]:
        """Return (model, provider_settings) so callers can inspect tool_calling mode."""
        selection = self.get_model_selection(language=language, prefer_local=prefer_local)
        providers = self.config.llm.providers
        provider = providers.primary
        if selection.source.startswith("fallback") and providers.fallback:
            provider = providers.fallback
        return selection.model, provider
