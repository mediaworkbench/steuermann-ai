"""LLM factory with language-aware provider selection and optional fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from universal_agentic_framework.config import CoreConfig
from universal_agentic_framework.config.schemas import ProviderSettings
from universal_agentic_framework.llm.provider_registry import parse_model_id

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
    getter = getattr(provider, "get_tool_calling_mode", None)
    tool_mode = getter(model_name) if callable(getter) else getattr(provider, "tool_calling", "structured")
    if tool_mode != "native":
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
        try:
            return parse_model_id(model_name).provider
        except Exception:
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

    def _chat_provider_chain(self) -> List[Tuple[str, ProviderSettings]]:
        registry = self.config.llm.providers.get_registry()
        role_refs = self.config.llm.roles.chat.providers
        chain: List[Tuple[str, ProviderSettings]] = []
        for ref in role_refs:
            provider = registry.get(ref.provider_id)
            if provider:
                chain.append((ref.provider_id, provider))
        if not chain:
            raise RuntimeError("No providers configured for chat role")
        return chain

    @staticmethod
    def _source_label_for(provider_id: str, index: int) -> str:
        if provider_id in {"primary", "fallback"}:
            return provider_id
        if index == 0:
            return "primary"
        if index == 1:
            return "fallback"
        return provider_id

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
        provider_chain = self._chat_provider_chain()
        if not prefer_local and len(provider_chain) > 1:
            provider_chain = provider_chain[1:]
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

        for index, (_provider_id, provider) in enumerate(provider_chain):
            label = self._source_label_for(_provider_id, index)
            if preferred_model:
                try_add(provider, f"{label}_preferred", preferred_model)
            if (not preferred_model) or include_default_when_preferred:
                try_add(provider, f"{label}_default")

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

        provider_chain = self._chat_provider_chain()
        primary_provider_id, primary = provider_chain[0]
        primary_label = self._source_label_for(primary_provider_id, 0)
        primary_model = preferred_model or self._select_model(primary, language)
        model_list = [
            {
                "model_name": primary_label,
                "litellm_params": self._to_router_params(primary, primary_model),
            }
        ]

        for index, (provider_id, provider) in enumerate(provider_chain[1:], start=1):
            fallback_label = self._source_label_for(provider_id, index)
            fallback_model = preferred_model or self._select_model(provider, language)
            model_list.append(
                {
                    "model_name": fallback_label,
                    "litellm_params": self._to_router_params(provider, fallback_model),
                }
            )

        router_config = self.config.llm.router
        router_kwargs = {
            "model_list": model_list,
            "routing_strategy": router_config.routing_strategy,
            "num_retries": router_config.num_retries,
            "retry_after": router_config.retry_after,
            "disable_cooldowns": router_config.disable_cooldowns,
            "enable_pre_call_checks": router_config.enable_pre_call_checks,
            "set_verbose": router_config.set_verbose,
        }
        if router_config.allowed_fails is not None:
            router_kwargs["allowed_fails"] = router_config.allowed_fails
        if router_config.cooldown_time is not None:
            router_kwargs["cooldown_time"] = router_config.cooldown_time
        if router_config.debug_level is not None:
            router_kwargs["debug_level"] = router_config.debug_level
        if router_config.fallbacks:
            router_kwargs["fallbacks"] = router_config.fallbacks
        if router_config.default_fallbacks:
            router_kwargs["default_fallbacks"] = router_config.default_fallbacks
        if router_config.context_window_fallbacks:
            router_kwargs["context_window_fallbacks"] = router_config.context_window_fallbacks
        if router_config.content_policy_fallbacks:
            router_kwargs["content_policy_fallbacks"] = router_config.content_policy_fallbacks
        router = Router(**router_kwargs)

        model_kwargs = {}
        getter = getattr(primary, "get_tool_calling_mode", None)
        tool_mode = getter(primary_model) if callable(getter) else getattr(primary, "tool_calling", "structured")
        if tool_mode != "native":
            model_kwargs["tool_choice"] = "none"
        return ChatLiteLLMRouter(
            router=router,
            model_name=primary_label,
            temperature=primary.temperature,
            max_tokens=primary.max_tokens,
            model_kwargs=model_kwargs,
        )

    def _to_router_params(self, provider: ProviderSettings, model_name: str) -> dict:
        params: dict = {"model": model_name}
        router_defaults = self.config.llm.router
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
        elif router_defaults.default_max_parallel_requests is not None:
            params["max_parallel_requests"] = router_defaults.default_max_parallel_requests
        if provider.region_name:
            params["region_name"] = provider.region_name
        return params

    def get_model_with_config(
        self, language: str, prefer_local: bool = True
    ) -> Tuple[object, ProviderSettings]:
        """Return (model, provider_settings) so callers can inspect tool_calling mode."""
        selection = self.get_model_selection(language=language, prefer_local=prefer_local)
        provider_chain = self._chat_provider_chain()
        source_prefix = selection.source.split("_", 1)[0]
        for index, (_provider_id, provider) in enumerate(provider_chain):
            label = self._source_label_for(_provider_id, index)
            if label == source_prefix:
                return selection.model, provider
        return selection.model, provider_chain[0][1]
