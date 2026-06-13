"""Tool scoring and semantic selection helpers."""

from typing import Any, Callable, Dict, Tuple

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


# Declarative intent-boost rules (mirrors the table in CLAUDE.md). Each predicate receives
# (intents, image_attachment_present, csv_workspace_doc_present) and returns True when the
# tool matches a detected intent and should get the prefilter score boost. Each tool name has
# exactly one rule, so the original if/elif "first match" semantics are preserved by lookup.
_IntentPredicate = Callable[[Dict[str, Any], bool, bool], Any]
_INTENT_BOOST_RULES: Dict[str, _IntentPredicate] = {
    "datetime_tool": lambda i, img, csv: i.get("mentions_datetime"),
    "calculator_tool": lambda i, img, csv: i.get("mentions_calculation"),
    "extract_webpage_mcp": lambda i, img, csv: i.get("url_in_query"),
    "web_search_mcp": lambda i, img, csv: i.get("mentions_web_search"),
    "analyze_image_tool": lambda i, img, csv: i.get("image_url_in_query") or img,
    "ocr_tool": lambda i, img, csv: (i.get("image_in_query") or img) and i.get("mentions_ocr"),
    "analyze_document_tool": lambda i, img, csv: (i.get("image_in_query") or img) and i.get("mentions_document"),
    "analyze_chart_tool": lambda i, img, csv: (i.get("image_in_query") or img) and i.get("mentions_chart"),
    "image_metadata_tool": lambda i, img, csv: (i.get("image_in_query") or img) and i.get("mentions_image_metadata"),
    "read_barcodes_tool": lambda i, img, csv: (i.get("image_in_query") or img) and i.get("mentions_barcode"),
    "map_tool": lambda i, img, csv: i.get("mentions_map"),
    "csv_analyze_tool": lambda i, img, csv: i.get("mentions_csv_analysis") and csv,
}

# Tools that get a hard floor (not just a +boost) when their intent is explicitly signalled,
# so they survive both threshold gates and Layer 2 can make the final call.
_INTENT_OVERRIDE_FLOOR: Dict[str, str] = {
    "web_search_mcp": "mentions_web_search",
    "map_tool": "mentions_map",
}


def intent_boost_applies(
    tool_name: str,
    intents: Dict[str, Any],
    *,
    image_attachment_present: bool,
    csv_workspace_doc_present: bool,
) -> bool:
    """Return True if ``tool_name`` matches a detected intent and should get the score boost."""
    rule = _INTENT_BOOST_RULES.get(tool_name)
    if rule is None:
        return False
    return bool(rule(intents, image_attachment_present, csv_workspace_doc_present))


def apply_intent_override_floor(
    tool_name: str,
    similarity: float,
    intents: Dict[str, Any],
    *,
    similarity_threshold: float,
    min_top_score: float,
) -> Tuple[float, bool]:
    """Raise ``similarity`` to a forced floor for floor-eligible tools whose intent is signalled.

    Returns ``(similarity, applied)`` — ``applied`` is True only when the floor actually lifted
    the score (so the caller can log it).
    """
    intent_key = _INTENT_OVERRIDE_FLOOR.get(tool_name)
    if not intent_key or not intents.get(intent_key):
        return similarity, False
    forced_floor = max(similarity_threshold, min_top_score) + 0.01
    if similarity < forced_floor:
        return forced_floor, True
    return similarity, False
