from types import SimpleNamespace

import pytest

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


def test_invoke_with_model_fallback_classifies_litellm_errors():
    """LiteLLM-specific exceptions are classified into the appropriate error_type string."""
    try:
        from litellm.exceptions import ContextWindowExceededError
    except ImportError:
        pytest.skip("litellm not installed")

    class _ContextWindowModel:
        def invoke(self, _payload):
            raise ContextWindowExceededError(
                message="context window exceeded",
                model="test-model",
                llm_provider="openai",
            )

    with pytest.raises(_ModelInvokeError) as exc_info:
        invoke_with_model_fallback(
            config=None,
            language="en",
            payload="hello",
            initial_model=_ContextWindowModel(),
            initial_provider="openai",
            initial_model_name="test-model",
            preferred_model=None,
            error_cls=_ModelInvokeError,
        )

    assert exc_info.value.error_type == "context_window_exceeded"
    assert exc_info.value.provider == "openai"
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
