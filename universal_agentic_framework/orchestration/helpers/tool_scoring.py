"""Tool scoring and semantic selection helpers."""

from typing import Any, Dict

import numpy as np

# Cache of tool-description embeddings, keyed on (embedding model, tool name, desc hash).
# Cleared by tests via the imported symbol; populated lazily on first scoring pass.
_tool_embedding_cache: Dict[str, Any] = {}


def score_tool_similarity(
    *,
    embedding_model_name: str,
    tool_name: str,
    tool_desc: str,
    embedding_provider: Any,
    query_embedding: Any,
) -> float:
    """Cosine similarity between the query embedding and a tool's description embedding.

    The tool-description embedding is cached per (embedding model, tool name, desc hash),
    so it is encoded once per tool rather than on every prefilter pass. Intent boosting is
    applied by the caller (node_prefilter_tools), not here.
    """
    cache_key = f"{embedding_model_name}:{tool_name}:{hash(tool_desc)}"
    if cache_key not in _tool_embedding_cache:
        _tool_embedding_cache[cache_key] = embedding_provider.encode(tool_desc)
    tool_embedding = _tool_embedding_cache[cache_key]
    return float(
        np.dot(query_embedding, tool_embedding)
        / (np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding))
    )
