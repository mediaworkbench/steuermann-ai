"""Model resolution and fallback logic helpers."""

from typing import Any, Optional, Tuple

from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.llm.provider_registry import normalize_model_id

try:
    from litellm.exceptions import (
        ContextWindowExceededError as _CWE,
        RateLimitError as _RLE,
        AuthenticationError as _AE,
        ServiceUnavailableError as _SUE,
    )
except ImportError:
    _CWE = _RLE = _AE = _SUE = None  # type: ignore[assignment,misc]


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


def _classify_error(exc: Exception) -> str:
    if _CWE and isinstance(exc, _CWE):
        return "context_window_exceeded"
    if _RLE and isinstance(exc, _RLE):
        return "rate_limit"
    if _AE and isinstance(exc, _AE):
        return "auth_error"
    if _SUE and isinstance(exc, _SUE):
        return "service_unavailable"
    return "error"


def safe_get_model(config, language: str, preferred_model: Optional[str] = None):
    """Return a ChatLiteLLMRouter; fallback to echo model if providers fail.

    Always uses the Router so that LiteLLM's native retry and fallback semantics
    apply regardless of whether a preferred_model override is in effect.
    """
    try:
        factory = LLMFactory(config)
        return factory.get_router_model(language=language, preferred_model=preferred_model)
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
        factory = LLMFactory(config)
        if preferred_model:
            selection = factory.get_model_candidates(
                language=language,
                preferred_model=preferred_model,
                prefer_local=True,
                include_default_when_preferred=False,
            )[0]
        else:
            selection = factory.get_model_selection(language=language, prefer_local=True)
        provider = selection.provider_type
        model_name = selection.model_name
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
) -> Tuple[str, str, str, object, Optional[dict]]:
    """Invoke a model (expected to be ChatLiteLLMRouter) and normalize the response.

    Returns (text, provider, model_name, model, usage_metadata).

    The Router handles retries and provider fallbacks internally. This function's
    responsibility is response normalization and error classification/wrapping.
    """
    try:
        invoke_fn = getattr(initial_model, "invoke", None)
        out = invoke_fn(payload) if callable(invoke_fn) else (
            initial_model(payload) if callable(initial_model) else str(payload)  # type: ignore[operator]
        )
        _meta = getattr(out, "usage_metadata", None)
        usage_metadata = _meta if isinstance(_meta, dict) and _meta else None
        raw_text = out.content if hasattr(out, "content") else out
        text = _normalize_response_text(raw_text)
        if not text.strip():
            raise ValueError("LLM returned empty response content")
        if logger is not None:
            logger.info("LLM invoke succeeded", provider=initial_provider, model=initial_model_name)
        return text, initial_provider, initial_model_name, initial_model, usage_metadata
    except Exception as exc:
        error_type = _classify_error(exc)
        if logger is not None:
            logger.warning(
                "LLM invoke failed",
                provider=initial_provider,
                model=initial_model_name,
                error=str(exc),
                error_type=error_type,
            )
        if error_cls is not None:
            raise error_cls(str(exc), provider=initial_provider, model_name=initial_model_name, error_type=error_type)
        raise
