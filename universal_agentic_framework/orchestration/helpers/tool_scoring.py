"""Tool scoring and semantic selection helpers."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from langchain.tools import BaseTool
_tool_embedding_cache: Dict[str, Any] = {}


def score_tool_similarity(*args, **kwargs) -> float:
    """Score tool relevance via embedding similarity + intent-boosted scoring.

    Combines semantic (embedding) and syntactic (keyword) signals to rank tools
    for prefiltering. Higher scores = more relevant for the query.
    """
    # Compatibility path for graph_builder's existing call contract.
    if "embedding_model_name" in kwargs and "tool_name" in kwargs and "query_embedding" in kwargs:
        embedding_model_name = kwargs["embedding_model_name"]
        tool_name = kwargs["tool_name"]
        tool_desc = kwargs["tool_desc"]
        embedding_provider = kwargs["embedding_provider"]
        query_embedding = kwargs["query_embedding"]

        cache_key = f"{embedding_model_name}:{tool_name}:{hash(tool_desc)}"
        if cache_key not in _tool_embedding_cache:
            _tool_embedding_cache[cache_key] = embedding_provider.encode(tool_desc)
        tool_embedding = _tool_embedding_cache[cache_key]
        return float(np.dot(query_embedding, tool_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
        ))

    # Generic helper path (original helper contract).
    user_msg_lower = kwargs.get("user_msg_lower", args[0] if len(args) > 0 else "")
    tool = kwargs.get("tool", args[1] if len(args) > 1 else None)
    embedding_provider = kwargs.get("embedding_provider", args[2] if len(args) > 2 else None)
    embedding_config_key = kwargs.get("embedding_config_key", args[3] if len(args) > 3 else "")
    intent_hints = kwargs.get("intent_hints", args[4] if len(args) > 4 else None)

    if tool is None or embedding_provider is None:
        return 0.0

    intent_hints = intent_hints or {}
    base_score = embedding_provider.similarity_score(
        user_msg_lower,
        tool.description or "",
        embedding_config_key,
    )
    base_score = max(0.0, min(1.0, base_score))

    # Intent boost: boost tools that match detected intents
    tool_name_lower = (tool.name or "").lower()
    if intent_hints.get("mentions_web_search") and "web" in tool_name_lower:
        base_score = min(1.0, base_score + 0.15)
    if intent_hints.get("mentions_datetime") and any(w in tool_name_lower for w in ["time", "date", "datetime", "now"]):
        base_score = min(1.0, base_score + 0.15)
    if intent_hints.get("mentions_calculation") and "calcul" in tool_name_lower:
        base_score = min(1.0, base_score + 0.15)

    # Penalty for file operations on production environments where file access is restricted
    if "file" in tool_name_lower or "read" in tool_name_lower:
        base_score = max(0.0, base_score - 0.05)

    return base_score


def build_semantic_tool_kwargs(
    state: Any,
    embedding_provider,
    embedding_config_key: str,
    all_tools: List[BaseTool],
    intent_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build tool-calling kwargs for semantic-scored tool selection."""
    intent_hints = intent_hints or {}
    user_msg_lower = state.get("user_msg_lower", "").lower()

    # Rank all tools by semantic + intent relevance
    tool_scores = [
        (tool, score_tool_similarity(user_msg_lower, tool, embedding_provider, embedding_config_key, intent_hints))
        for tool in all_tools
    ]
    tool_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top-K tools (default 5) for presentation to model
    top_k = 5
    selected_tools = [tool for tool, _ in tool_scores[:top_k]]

    return {
        "tools": selected_tools,
        "scoring_metadata": {
            f"{tool.name}": {"score": score, "rank": i + 1}
            for i, (tool, score) in enumerate(tool_scores[:10])
        },
    }
