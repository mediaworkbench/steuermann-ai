"""Chat model for OpenAI-compatible endpoints (LM Studio, Ollama, OpenRouter, gateways).

``langchain_openai.ChatOpenAI`` targets the official OpenAI API only, which bites
OpenAI-compatible servers in two ways:

- it drops non-standard response fields, so a reasoning model's thinking text
  (``reasoning_content``; ``reasoning`` on OpenRouter-style gateways) never reaches
  ``additional_kwargs`` — the LangGraph stream relies on it for ``thinking`` events;
- it rewrites ``max_tokens`` to ``max_completion_tokens``, a field these servers do
  not document, so the output cap could be silently ignored.

``OpenAICompatibleChat`` therefore builds on ``BaseChatOpenAI`` (which sends plain
``max_tokens``) and copies the reasoning text into
``additional_kwargs["reasoning_content"]`` — the same approach as ``ChatDeepSeek``.
"""

from __future__ import annotations

from typing import Any, Optional

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


def _reasoning_text(message: Any) -> Optional[str]:
    """Return the reasoning text of a message/delta dict, if the provider sent any."""
    if not isinstance(message, dict):
        return None
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    return reasoning if isinstance(reasoning, str) and reasoning else None


class OpenAICompatibleChat(BaseChatOpenAI):
    """``BaseChatOpenAI`` that keeps the provider's reasoning text."""

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-chat"

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or []
        if generation_chunk and choices and isinstance(generation_chunk.message, AIMessageChunk):
            reasoning = _reasoning_text(choices[0].get("delta"))
            if reasoning is not None:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response.model_dump() if isinstance(response, openai.BaseModel) else response
        choices = (response_dict or {}).get("choices") or []
        if choices and result.generations:
            reasoning = _reasoning_text(choices[0].get("message"))
            if reasoning is not None:
                result.generations[0].message.additional_kwargs["reasoning_content"] = reasoning
        return result
