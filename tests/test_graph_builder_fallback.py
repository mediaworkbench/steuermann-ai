from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.exceptions import ContextOverflowError, ModelRateLimitError, ModelTimeoutError
from langchain_openai.chat_models.base import OpenAIAPIContextOverflowError

from universal_agentic_framework.orchestration.helpers.model_resolution import invoke_with_model_fallback


class _ModelInvokeError(RuntimeError):
    def __init__(self, message: str, provider: str, model_name: str, error_type: str = "error"):
        super().__init__(message)
        self.provider = provider
        self.model_name = model_name
        self.error_type = error_type


class _FailingModel:
    def invoke(self, _payload):
        raise RuntimeError("primary invoke failed")


class _WorkingModel:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload):
        return SimpleNamespace(content=self.text)


class _ListContentModel:
    def invoke(self, _payload):
        return SimpleNamespace(
            content=[
                {"type": "text", "text": "Hallo"},
                {"type": "text", "text": "Welt"},
            ]
        )



def test_invoke_with_model_fallback_generic_error_uses_initial_metadata():
    """A plain RuntimeError from the initial model is re-raised as _ModelInvokeError
    with the initial provider/model metadata and error_type='error'."""
    with pytest.raises(_ModelInvokeError) as exc_info:
        invoke_with_model_fallback(
            config=None,
            language="en",
            payload="hello",
            initial_model=_FailingModel(),
            initial_provider="mock",
            initial_model_name="primary-model",
            preferred_model=None,
            error_cls=_ModelInvokeError,
        )

    assert exc_info.value.provider == "mock"
    assert exc_info.value.model_name == "primary-model"
    assert exc_info.value.error_type == "error"


def _status_error(cls, status: int, message: str, body=None):
    request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=body)


_REQUEST = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        # OpenAI/OpenRouter flag an overflowing prompt with a code …
        (_status_error(openai.BadRequestError, 400, "too long", {"code": "context_length_exceeded"}),
         "context_window_exceeded"),
        # … local servers only say so in the message text.
        (_status_error(openai.BadRequestError, 400,
                       "The number of tokens to keep from the initial prompt is greater than the context length"),
         "context_window_exceeded"),
        (_status_error(openai.BadRequestError, 400, "invalid 'temperature'"), "error"),
        (_status_error(openai.RateLimitError, 429, "slow down"), "rate_limit"),
        (_status_error(openai.AuthenticationError, 401, "bad key"), "auth_error"),
        (_status_error(openai.PermissionDeniedError, 403, "forbidden"), "auth_error"),
        (_status_error(openai.InternalServerError, 500, "boom"), "service_unavailable"),
        (openai.APIConnectionError(request=_REQUEST), "service_unavailable"),
        (openai.APITimeoutError(request=_REQUEST), "service_unavailable"),
        # langchain-openai reports some overflows as a non-400 APIError …
        (OpenAIAPIContextOverflowError(message="input exceeds the context window", request=_REQUEST, body=None),
         "context_window_exceeded"),
        # … and any chat-model builder raises LangChain's provider-agnostic classes.
        (ContextOverflowError("prompt too long"), "context_window_exceeded"),
        (ModelRateLimitError("slow down"), "rate_limit"),
        (ModelTimeoutError("timed out"), "service_unavailable"),
    ],
)
def test_invoke_with_model_fallback_classifies_openai_errors(exc, expected):
    """OpenAI-SDK exceptions map onto the error types the respond node reports to users."""

    class _RaisingModel:
        def invoke(self, _payload):
            raise exc

    with pytest.raises(_ModelInvokeError) as exc_info:
        invoke_with_model_fallback(
            config=None,
            language="en",
            payload="hello",
            initial_model=_RaisingModel(),
            initial_provider="lmstudio",
            initial_model_name="test-model",
            preferred_model=None,
            error_cls=_ModelInvokeError,
        )

    assert exc_info.value.error_type == expected
    assert exc_info.value.provider == "lmstudio"
    assert exc_info.value.model_name == "test-model"


def test_invoke_with_model_fallback_normalizes_list_content_blocks():
    text, provider, model_name, _, _usage = invoke_with_model_fallback(
        config=None,
        language="en",
        payload="hello",
        initial_model=_ListContentModel(),
        initial_provider="openrouter",
        initial_model_name="openrouter/poolside/laguna-m.1:free",
        preferred_model=None,
    )

    assert text == "Hallo\nWelt"
    assert provider == "openrouter"
    assert model_name == "openrouter/poolside/laguna-m.1:free"
