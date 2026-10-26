"""LLM factory with language-aware provider selection and optional fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from universal_agentic_framework.config import CoreConfig
from universal_agentic_framework.config.schemas import ProviderSettings

BuilderFn = Callable[[ProviderSettings, str], object]


@dataclass(frozen=True)
class ModelSelection:
    """Resolved runtime model selection metadata."""

    model: object
    provider_type: str  # the role's provider_id, e.g. "lmstudio"
    model_name: str
    api_base: Optional[str]
    source: str


def build_openai_compatible_chat(provider: ProviderSettings, model_name: str):
    """Chat model for an OpenAI-compatible endpoint (LM Studio, Ollama, OpenRouter, gateways)."""
    from universal_agentic_framework.llm.chat_model import OpenAICompatibleChat

    return OpenAICompatibleChat(
        model=model_name,
        base_url=str(provider.api_base) if provider.api_base else None,
        # The OpenAI SDK insists on a key; local servers ignore it.
        api_key=provider.api_key or "not-needed",
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
        timeout=provider.timeout,
        max_retries=provider.max_retries,
        streaming=True,
        stream_usage=True,
        # Gateways may not implement the Responses API; never route there by model name.
        use_responses_api=False,
    )


DEFAULT_BUILDERS: Dict[str, BuilderFn] = {
    "openai_compatible": build_openai_compatible_chat,
}


def build_chat_model(
    provider: ProviderSettings,
    model_name: str,
    builders: Optional[Dict[str, BuilderFn]] = None,
):
    """Build the chat model for ``provider`` with the builder registered for its api_type."""
    registry = builders or DEFAULT_BUILDERS
    builder = registry.get(provider.api_type)
    if builder is None:
        raise ValueError(f"No LLM builder registered for api_type '{provider.api_type}'")
    return builder(provider, model_name)


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
        return build_chat_model(provider, model_name, self.builders)

    def _build_selection(
        self,
        provider_id: str,
        provider: ProviderSettings,
        model_name: str,
        source: str,
    ) -> ModelSelection:
        model = self._build(provider, model_name)
        api_base = str(provider.api_base) if provider.api_base else None
        return ModelSelection(
            model=model,
            provider_type=provider_id,
            model_name=model_name,
            api_base=api_base,
            source=source,
        )

    def _chat_provider_chain(self, language: str) -> List[Tuple[str, ProviderSettings, str]]:
        chain: List[Tuple[str, ProviderSettings, str]] = []
        try:
            chain = self.config.llm.get_role_provider_chain_with_models("chat", language)
        except Exception:
            chain = []
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
        provider_chain = self._chat_provider_chain(language)
        if not prefer_local and len(provider_chain) > 1:
            provider_chain = provider_chain[1:]
        candidates: list[ModelSelection] = []
        errors = []
        seen: set[tuple[str, str, str]] = set()

        def try_add(provider_id: str, provider: Optional[ProviderSettings], default_model: Optional[str], source: str, model_override: Optional[str] = None):
            if not provider:
                return
            try:
                model_name = model_override or default_model or self._select_model(provider, language)
                key = (provider_id, model_name, source)
                if key in seen:
                    return
                selection = self._build_selection(provider_id, provider, model_name, source)
                candidates.append(selection)
                seen.add(key)
            except Exception as exc:  # pragma: no cover - fallback path
                errors.append(exc)

        for index, (_provider_id, provider, default_model_name) in enumerate(provider_chain):
            label = self._source_label_for(_provider_id, index)
            if preferred_model:
                try_add(_provider_id, provider, default_model_name, f"{label}_preferred", preferred_model)
            if (not preferred_model) or include_default_when_preferred:
                try_add(_provider_id, provider, default_model_name, f"{label}_default")

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

    def get_model(
        self,
        language: str,
        prefer_local: bool = True,
        preferred_model: Optional[str] = None,
    ):
        """Return the chat model for ``language``, using ``preferred_model`` when given.

        Retries are handled by the client (``llm.max_retries``); there is no router.
        """
        if preferred_model:
            return self.get_model_candidates(
                language=language,
                preferred_model=preferred_model,
                prefer_local=prefer_local,
                include_default_when_preferred=False,
            )[0].model
        return self.get_model_selection(language=language, prefer_local=prefer_local).model

    def create_auxiliary_llm(self):
        """Return the chat model for the auxiliary role.

        Falls back to the chat role if auxiliary is not configured.
        """
        role_name = "auxiliary" if getattr(getattr(self.config.llm, "roles", None), "auxiliary", None) is not None else "chat"
        provider = self.config.llm.get_role_provider(role_name)
        model_name = self.config.llm.get_role_model_name(role_name, "en")
        return self._build(provider, model_name)

    def get_model_with_config(
        self, language: str, prefer_local: bool = True
    ) -> Tuple[object, ProviderSettings]:
        """Return (model, provider_settings) so callers can inspect tool_calling mode."""
        selection = self.get_model_selection(language=language, prefer_local=prefer_local)
        provider_chain = self._chat_provider_chain(language)
        source_prefix = selection.source.split("_", 1)[0]
        for index, (_provider_id, provider, _model_name) in enumerate(provider_chain):
            label = self._source_label_for(_provider_id, index)
            if label == source_prefix:
                return selection.model, provider
        return selection.model, provider_chain[0][1]
