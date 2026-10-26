"""Model resolution and fallback logic helpers."""

import re
from typing import Any, Optional, Tuple

import openai
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

from universal_agentic_framework.llm.factory import LLMFactory, build_chat_model
from universal_agentic_framework.llm.provider_registry import normalize_model_id

# Local servers report an overflowing prompt as a plain 400 text, not with OpenAI's
# ``context_length_exceeded`` code — match the usual phrasings as well.
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _](length|window)|maximum context|context_length_exceeded|n_ctx",
    re.IGNORECASE,
)


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
    """Map a model error onto the error types the respond node reports to users.

    LangChain's provider-agnostic error classes come first: langchain-openai raises them
    (e.g. a context overflow reported as a non-400 ``APIError``) and any other chat-model
    builder would too. The raw OpenAI-SDK classes cover errors raised outside a chat model.
    """
    if isinstance(exc, ContextOverflowError):
        return "context_window_exceeded"
    if isinstance(exc, openai.BadRequestError):
        # Local servers (LM Studio, llama.cpp) report an overflow as plain 400 text that
        # langchain-openai does not recognise.
        code = str(getattr(exc, "code", "") or "")
        if code == "context_length_exceeded" or _CONTEXT_OVERFLOW_RE.search(str(exc)):
            return "context_window_exceeded"
        return "error"
    if isinstance(exc, (ModelRateLimitError, openai.RateLimitError)):
        return "rate_limit"
    if isinstance(
        exc,
        (ModelAuthenticationError, ModelPermissionDeniedError, openai.AuthenticationError, openai.PermissionDeniedError),
    ):
        return "auth_error"
    # APITimeoutError subclasses APIConnectionError; InternalServerError covers 5xx.
    if isinstance(
        exc,
        (ModelConnectionError, ModelTimeoutError, ModelAPIError, openai.APIConnectionError, openai.InternalServerError),
    ):
        return "service_unavailable"
    return "error"


def get_auxiliary_model(config, language: str = "en"):
    """Return (chat model, provider_id, model_name) for the auxiliary role.

    Falls back to the chat role if auxiliary is not configured.
    Language accepted for API consistency; ignored by the flat per-role schema.
    """
    role_name = "auxiliary" if getattr(getattr(config.llm, "roles", None), "auxiliary", None) is not None else "chat"
    provider = config.llm.get_role_provider(role_name)
    model_name = config.llm.get_role_model_name(role_name, language)
    llm = build_chat_model(provider, model_name)
    provider_id = getattr(provider, "provider_id", role_name)
    return llm, provider_id, model_name


def get_vision_model(config, language: str = "en"):
    """Return (chat model, provider_id, model_name) for the vision role.

    Raises ValueError if llm.roles.vision is not configured — callers should
    check for vision role availability before calling this.
    """
    provider = config.llm.get_role_provider("vision")
    model_name = config.llm.get_role_model_name("vision", language)
    llm = build_chat_model(provider, model_name)
    provider_id = getattr(provider, "provider_id", "vision")
    return llm, provider_id, model_name


def get_model(config, language: str, preferred_model: Optional[str] = None):
    """Return the chat model for the given language/model preference.

    Retries happen in the client (``llm.max_retries``).

    Raises:
        Exception: Propagates any error from LLMFactory (provider misconfiguration,
            unreachable service, etc.) — callers must not silently swallow this.
    """
    factory = LLMFactory(config)
    return factory.get_model(language=language, preferred_model=preferred_model)


def resolve_initial_model_metadata(config: Any, language: str, preferred_model: Optional[str]) -> Tuple[str, str]:
    """Best-effort metadata for the initial selected model."""
    provider = "unknown"
    model_name = "unknown"
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
    """Invoke a chat model and normalize the response.

    Returns (text, provider, model_name, model, usage_metadata).

    The model's client retries transient failures itself (``llm.max_retries``). This
    function's responsibility is response normalization and error classification/wrapping.
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
