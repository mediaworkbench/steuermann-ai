"""OpenAICompatibleChat: reasoning text survives the OpenAI-compatible client.

These exercise the langchain-openai internal hooks the subclass overrides, so a
langchain-openai release that changes them fails here instead of silently dropping
the thinking stream.
"""
from __future__ import annotations

import openai
from langchain_core.messages import AIMessageChunk

from universal_agentic_framework.llm.chat_model import OpenAICompatibleChat


def _model() -> OpenAICompatibleChat:
    return OpenAICompatibleChat(model="m", base_url="http://localhost:1234/v1", api_key="k")


def _stream_chunk(delta: dict) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _completion(message: dict) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_stream_chunk_keeps_reasoning_content():
    gen = _model()._convert_chunk_to_generation_chunk(
        _stream_chunk({"role": "assistant", "content": "", "reasoning_content": "let me think"}),
        AIMessageChunk,
        {},
    )

    assert gen is not None
    assert gen.message.additional_kwargs["reasoning_content"] == "let me think"


def test_stream_chunk_maps_reasoning_alias():
    """OpenRouter-style gateways send the thinking text as ``reasoning``."""
    gen = _model()._convert_chunk_to_generation_chunk(
        _stream_chunk({"content": "", "reasoning": "step one"}), AIMessageChunk, {}
    )

    assert gen.message.additional_kwargs["reasoning_content"] == "step one"


def test_plain_content_chunk_is_unchanged():
    gen = _model()._convert_chunk_to_generation_chunk(
        _stream_chunk({"content": "Hello"}), AIMessageChunk, {}
    )

    assert gen.message.content == "Hello"
    assert "reasoning_content" not in gen.message.additional_kwargs


def test_usage_only_chunk_without_choices_is_handled():
    chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    gen = _model()._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

    assert gen is not None
    assert gen.message.usage_metadata["total_tokens"] == 5
    assert "reasoning_content" not in gen.message.additional_kwargs


def test_non_streaming_result_keeps_reasoning_from_dict():
    result = _model()._create_chat_result(
        _completion({"role": "assistant", "content": "391 = 17 × 23", "reasoning_content": "factor it"})
    )

    message = result.generations[0].message
    assert message.content == "391 = 17 × 23"
    assert message.additional_kwargs["reasoning_content"] == "factor it"


def test_non_streaming_result_keeps_reasoning_from_sdk_object():
    response = openai.types.chat.ChatCompletion.model_validate(
        _completion({"role": "assistant", "content": "done", "reasoning": "short thought"})
    )
    result = _model()._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs["reasoning_content"] == "short thought"


def test_non_streaming_result_without_reasoning_is_unchanged():
    result = _model()._create_chat_result(_completion({"role": "assistant", "content": "plain"}))

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs
