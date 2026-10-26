"""Model ID normalization."""
from __future__ import annotations


def normalize_model_id(model_name: str) -> str:
    """Return the model ID exactly as the endpoint knows it, trimmed.

    IDs are the provider's own (``google/gemma-4-e4b``, ``llama3.1:8b``, OpenRouter's
    ``openai/gpt-5``): no transport prefix is required, stripped or case-folded, since
    a leading segment such as ``openai/`` can be part of the real ID.
    """
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError("model name must not be empty")
    return normalized
