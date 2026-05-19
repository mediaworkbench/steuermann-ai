"""Pure utility functions for RAG knowledge retrieval.

No LangGraph state, no config loading — all inputs are explicit parameters.
Used by orchestration/rag_node.py.
"""

import re
import httpx
from typing import Union

from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)

_RAG_STOPWORDS: frozenset[str] = frozenset({
    "kennst", "kannst", "weißt", "wissen", "sagen", "bitte",
    "about", "tell", "what", "which", "with", "have", "this",
    "that", "from", "oder", "aber", "denn", "dann", "doch",
})


def extract_rag_keyword(query: str) -> str | None:
    """Return the longest non-stopword token from query, or None if none found."""
    tokens = [t.lower() for t in re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", query)]
    candidates = [t for t in tokens if t not in _RAG_STOPWORDS]
    return max(candidates, key=len) if candidates else None


def search_qdrant(
    qdrant_url: str,
    collection_name: str,
    embedding_vector: list[float],
    top_k: int,
    with_payload: Union[bool, list[str]],
    with_vector: bool,
    score_threshold: float | None,
    timeout_seconds: int,
    label: str,
) -> list[dict]:
    """Execute a vector search against the Qdrant REST API."""
    payload: dict = {
        "vector": embedding_vector,
        "limit": top_k,
        "with_payload": with_payload,
        "with_vector": with_vector,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold

    logger.info("RAG: Searching Qdrant", url=qdrant_url, collection=collection_name, query_label=label)
    response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/points/search",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    result = response.json().get("result", [])
    logger.info("RAG: Search completed", query_label=label, results_count=len(result))
    return result


def filter_and_deduplicate(
    search_results: list[dict],
    min_relevance_score: float,
    top_k: int,
) -> list[dict]:
    """Client-side safety floor: drops results below min_relevance_score,
    deduplicates by document ID, then limits to top_k by score.

    Server-side score_threshold handles filtering when configured; this
    floor catches noise when no threshold is set (fallback: 0.6).
    """
    seen: set = set()
    docs = []
    for result in search_results:
        if result.get("score", 0.0) < min_relevance_score:
            continue
        payload = result.get("payload", {})
        # file_name is the pre-migration field; file_path is canonical
        file_path = payload.get("file_name") or payload.get("file_path") or "Unknown"
        file_name = file_path.split("/")[-1] if isinstance(file_path, str) else "Unknown"
        doc_id = result.get("id") or f"{file_path}:{payload.get('chunk_index')}"
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append({
            "text": payload.get("text", ""),
            "file_name": file_name,
            "file_path": file_path,
            "score": result["score"],
        })
    if len(docs) > top_k:
        docs = sorted(docs, key=lambda d: d["score"], reverse=True)[:top_k]
    return docs


def resolve_rag_config(
    user_rag_config: dict,
    system_rag_config,  # RagSettings | None
) -> dict:
    """Merge user overrides on top of system config baseline.

    Returns a flat dict with keys: collection_name, top_k, score_threshold,
    with_payload, with_vector, timeout_seconds.
    """
    resolved: dict = {
        "collection_name": "framework",
        "top_k": 5,
        "score_threshold": None,
        "with_payload": True,
        "with_vector": False,
        "timeout_seconds": 30,
    }

    if system_rag_config is not None:
        if system_rag_config.collection_name:
            resolved["collection_name"] = system_rag_config.collection_name
        resolved["top_k"] = system_rag_config.top_k
        resolved["score_threshold"] = system_rag_config.score_threshold
        resolved["with_payload"] = system_rag_config.with_payload
        resolved["with_vector"] = system_rag_config.with_vectors
        resolved["timeout_seconds"] = system_rag_config.timeout_seconds

    # User overrides on top of system baseline
    if user_rag_config:
        if user_rag_config.get("collection"):
            resolved["collection_name"] = user_rag_config["collection"]
        if user_rag_config.get("top_k") is not None:
            resolved["top_k"] = user_rag_config["top_k"]
        if user_rag_config.get("score_threshold") is not None:
            resolved["score_threshold"] = user_rag_config["score_threshold"]
        if user_rag_config.get("timeout_seconds") is not None:
            resolved["timeout_seconds"] = user_rag_config["timeout_seconds"]

    return resolved
