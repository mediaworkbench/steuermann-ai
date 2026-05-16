"""Tool preparation helpers for graph routing."""

from typing import Any, List, Optional, Tuple


def apply_top_k_scored_tools(
    scored_tools: List[Tuple[Any, float]],
    top_k: Optional[int],
) -> List[Tuple[Any, float]]:
    """Apply optional top-k selection to scored tools."""
    if top_k is None or top_k <= 0:
        return scored_tools
    return sorted(scored_tools, key=lambda x: x[1], reverse=True)[:top_k]
