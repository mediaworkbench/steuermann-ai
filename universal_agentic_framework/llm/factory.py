"""LLM factory with language-aware provider selection and optional fallback."""
from __future__ import annotations

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
    api_base: Optional[str]
    source: str


def build_litellm_chat(provider: ProviderSettings, model_name: str):
    try:
        from langchain_litellm import ChatLiteLLM
    except Exception as exc:  # pragma: no cover - import errors surfaced in tests
        raise RuntimeError("ChatLiteLLM not available") from exc

    model_kwargs = {}
    if provider.tool_calling != "native":
        model_kwargs["tool_choice"] = "none"

    chat_kwargs = {
        "model": model_name,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "timeout": provider.timeout,
        "model_kwargs": model_kwargs,
    }
    if provider.api_base:
        chat_kwargs["api_base"] = str(provider.api_base)
    if provider.api_key:
        chat_kwargs["api_key"] = provider.api_key

    return ChatLiteLLM(
        **chat_kwargs,
    )


DEFAULT_BUILDERS: Dict[str, BuilderFn] = {
    "litellm": build_litellm_chat,
}


class LLMFactory:
    def __init__(self, config: CoreConfig, builders: Optional[Dict[str, BuilderFn]] = None):
        self.config = config
        self.builders = builders or DEFAULT_BUILDERS
        self.default_builder_name = next(iter(self.builders.keys()))

    @staticmethod
    def _provider_from_model(model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/", 1)[0]
        return "unknown"

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
        builder = self.builders.get(self.default_builder_name)
        if not builder:
            raise ValueError("No LLM builder configured")
        return builder(provider, model_name)

    def _build_selection(
        self,
        provider: ProviderSettings,
        model_name: str,
        source: str,
    ) -> ModelSelection:
        model = self._build(provider, model_name)
        api_base = str(provider.api_base) if provider.api_base else None
        return ModelSelection(
            model=model,
            provider_type=self._provider_from_model(model_name),
            model_name=model_name,
            api_base=api_base,
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
                key = (self._provider_from_model(model_name), model_name, source)
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

    def get_router_model(
        self,
        language: str,
        preferred_model: Optional[str] = None,
    ):
        """Return a ChatLiteLLMRouter model with primary->fallback routing."""
        try:
            from litellm import Router
            from langchain_litellm import ChatLiteLLMRouter
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("LiteLLM router dependencies not available") from exc

        providers = self.config.llm.providers
        primary = providers.primary
        fallback = providers.fallback

        primary_model = preferred_model or self._select_model(primary, language)
        model_list = [
            {
                "model_name": "primary",
                "litellm_params": self._to_router_params(primary, primary_model),
            }
        ]

        if fallback:
            fallback_model = preferred_model or self._select_model(fallback, language)
            model_list.append(
                {
                    "model_name": "fallback",
                    "litellm_params": self._to_router_params(fallback, fallback_model),
                }
            )

        router = Router(
            model_list=model_list,
            num_retries=3,
            retry_after=1,
        )

        model_kwargs = {}
        if primary.tool_calling != "native":
            model_kwargs["tool_choice"] = "none"
        return ChatLiteLLMRouter(
            router=router,
            model_name="primary",
            temperature=primary.temperature,
            max_tokens=primary.max_tokens,
            model_kwargs=model_kwargs,
        )

    def _to_router_params(self, provider: ProviderSettings, model_name: str) -> dict:
        params: dict = {"model": model_name}
        if provider.api_base:
            params["api_base"] = str(provider.api_base)
        if provider.api_key:
            params["api_key"] = provider.api_key
        if provider.timeout is not None:
            params["timeout"] = provider.timeout
        if provider.max_tokens is not None:
            params["max_tokens"] = provider.max_tokens
        if provider.order is not None:
            params["order"] = provider.order
        if provider.weight is not None:
            params["weight"] = provider.weight
        if provider.rpm is not None:
            params["rpm"] = provider.rpm
        if provider.tpm is not None:
            params["tpm"] = provider.tpm
        if provider.max_parallel_requests is not None:
            params["max_parallel_requests"] = provider.max_parallel_requests
        if provider.region_name:
            params["region_name"] = provider.region_name
        return params

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
