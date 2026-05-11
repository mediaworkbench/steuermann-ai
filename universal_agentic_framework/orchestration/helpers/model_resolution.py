"""Model resolution and fallback logic helpers."""

from typing import Any, Optional, Tuple

from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.llm.provider_registry import normalize_model_id


def safe_get_model(config, language: str, preferred_model: Optional[str] = None):
    """Return an LLM model; fallback to simple echo if providers fail.

    Args:
        config: Core configuration
        language: Language code (e.g., 'en', 'de')
        preferred_model: Optional preferred model name to override config
    """
    try:
        factory = LLMFactory(config)
        if preferred_model:
            selection = factory.get_model_candidates(
                language=language,
                preferred_model=preferred_model,
                prefer_local=True,
                include_default_when_preferred=False,
            )[0]
            return selection.model
        return factory.get_router_model(language=language)
    except Exception:
        class _EchoModel:
            def invoke(self, prompt: str):
                class _Out:
                    content = f"LLM: {prompt}"
                return _Out()
        return _EchoModel()


def resolve_initial_model_metadata(config: Any, language: str, preferred_model: Optional[str]) -> Tuple[str, str]:
    """Best-effort metadata for the initial selected model."""
    provider = "unknown"
    model_name = preferred_model or "unknown"
    try:
        primary = config.llm.providers.primary
        if not preferred_model:
            factory = LLMFactory(config)
            model_name = factory._select_model(primary, language)
        provider = model_name.split("/", 1)[0] if "/" in model_name else "unknown"
    except Exception:
        pass
    return provider, model_name


def invoke_with_model_fallback(
    *,
    config: Any,
    language: str,
    payload: Any,
    initial_model: object,
    initial_provider: str,
    initial_model_name: str,
    preferred_model: Optional[str] = None,
    logger: Optional[Any] = None,
    error_cls: Optional[type] = None,
) -> Tuple[str, str, str, object]:
    """Invoke a model with ordered runtime fallback and return response text + metadata.

    This helper mirrors graph_builder behavior and supports custom error classes
    to preserve backward compatibility for existing tests and callers.
    """

    def _normalize_response_text(raw: Any) -> str:
        if raw is None:
            return ""

        if isinstance(raw, str):
            return raw

        if isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item)
                    continue

                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                        continue

                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        parts.append(content)
                        continue

            return "\n".join(parts).strip()

        if isinstance(raw, dict):
            text = raw.get("text")
            if isinstance(text, str):
                return text
            content = raw.get("content")
            if isinstance(content, str):
                return content
            return str(raw)

        return str(raw)

    attempts: List[Tuple[object, str, str, str]] = [
        (initial_model, initial_provider, initial_model_name, "initial"),
    ]
    # Deduplicate by effective provider/model identity; source labels may differ
    # for the same concrete candidate (e.g., initial vs primary_preferred).
    seen: set[Tuple[str, str]] = {(initial_provider, initial_model_name)}
    expanded_fallbacks = False
    idx = 0
    last_error: Optional[Exception] = None
    last_provider = initial_provider
    last_model_name = initial_model_name

    while idx < len(attempts):
        candidate_model, provider, model_name, source = attempts[idx]
        idx += 1
        try:
            invoke = getattr(candidate_model, "invoke", None)
            out = invoke(payload) if callable(invoke) else (
                candidate_model(payload) if callable(candidate_model) else str(payload)
            )
            raw_text = out.content if hasattr(out, "content") else out
            text = _normalize_response_text(raw_text)
            if not text.strip():
                raise ValueError("LLM returned empty response content")
            if logger is not None:
                logger.info(
                    "LLM invoke succeeded",
                    provider=provider,
                    model=model_name,
                    source=source,
                )
            return text, provider, model_name, candidate_model
        except Exception as exc:
            last_error = exc
            last_provider = provider
            last_model_name = model_name
            if logger is not None:
                logger.warning(
                    "LLM invoke failed; trying fallback if available",
                    provider=provider,
                    model=model_name,
                    source=source,
                    error=str(exc),
                )

            if not expanded_fallbacks:
                expanded_fallbacks = True
                factory = LLMFactory(config)
                try:
                    for selection in factory.get_model_candidates(
                        language=language,
                        preferred_model=preferred_model,
                        prefer_local=True,
                        # If a preferred model fails, include configured defaults
                        # so we can reliably fall back to provider-native models.
                        include_default_when_preferred=True,
                    ):
                        key = (selection.provider_type, selection.model_name)
                        if key in seen:
                            continue
                        attempts.append(
                            (
                                selection.model,
                                selection.provider_type,
                                selection.model_name,
                                selection.source,
                            )
                        )
                        seen.add(key)
                except Exception as candidate_exc:
                    if logger is not None:
                        logger.warning("Failed to build model fallback candidates", error=str(candidate_exc))

    error_msg = "LLM invoke failed for all candidates"
    if last_error is not None:
        error_msg = f"{error_msg}: {last_error}"

    if error_cls is not None:
        raise error_cls(error_msg, provider=last_provider, model_name=last_model_name)
    raise RuntimeError(error_msg)
