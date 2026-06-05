"""Admin RAG knowledge-base explorer.

Lets an administrator search a Qdrant collection by keyword and review the matching
chunks (text + file + similarity score) for evaluation. Reuses the same embedding
provider and Qdrant search helper that the LangGraph ``node_retrieve_knowledge`` node
uses, so scores match production.

A single semantic search returns *all* hits sorted by score, with no threshold cut.
Each hit is annotated ``above_cutoff`` relative to the production ``pill_score_threshold``
so the UI can mark exactly which chunks the chat would actually keep, while borderline
and below-cutoff chunks stay visible for inspection.

Endpoints are sync ``def`` on purpose: ``search_qdrant`` and ``embedder.encode`` use
blocking ``httpx``, so FastAPI runs them in its threadpool (matching the other admin
endpoints in ``settings.py``). An ``async def`` here would block the event loop.

Auth note: protected by the shared ``require_api_access`` token like every other router.
Admin-only access is enforced at the Next.js page layer (``/admin/`` middleware prefix +
``AdminOnly``); the backend does not do per-request role checks today — same posture as
the existing ``/api/admin/*`` endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.single_user import require_api_access
from universal_agentic_framework.config import load_core_config
from universal_agentic_framework.embeddings import EmbeddingProviderUnavailableError
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.orchestration.helpers.embedding_provider import (
    get_routing_embedding_provider,
)
from universal_agentic_framework.orchestration.helpers.rag_retrieval import (
    resolve_rag_config,
    search_qdrant,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/rag",
    tags=["rag"],
    dependencies=[Depends(require_api_access)],
)

# Fallback production threshold, matching node_retrieve_knowledge (rag_node.py).
_DEFAULT_PRODUCTION_THRESHOLD = 0.72
_DEFAULT_COLLECTION = "framework"


def _qdrant_url(config: Any) -> str:
    """Build the Qdrant REST base URL exactly like node_retrieve_knowledge does."""
    vs = config.memory.vector_store
    return f"http://{vs.host}:{vs.port}"


def _resolve_collection(override: Optional[str], system_cfg: dict) -> str:
    """collection query param -> profile rag.collection_name -> 'framework' fallback."""
    if override:
        return override
    return system_cfg.get("collection_name") or _DEFAULT_COLLECTION


def _serialize_hit(hit: dict, production_threshold: float) -> Dict[str, Any]:
    """Flatten a raw Qdrant search result into the API item shape."""
    payload = dict(hit.get("payload") or {})
    score = float(hit.get("score", 0.0))
    # file_name is the pre-migration field; file_path is canonical (see rag_retrieval).
    file_path = payload.get("file_name") or payload.get("file_path") or "Unknown"
    file_name = file_path.split("/")[-1] if isinstance(file_path, str) else "Unknown"
    text = payload.get("text", "")
    # Remaining payload fields (minus the ones surfaced as first-class) become metadata.
    metadata = {
        k: v
        for k, v in payload.items()
        if k not in {"text", "file_name", "file_path", "chunk_index", "chunk_count",
                     "detected_language", "language_confidence"}
    }
    return {
        "id": hit.get("id"),
        "score": score,
        "text": text,
        "file_name": file_name,
        "file_path": file_path,
        "chunk_index": payload.get("chunk_index"),
        "chunk_count": payload.get("chunk_count"),
        "detected_language": payload.get("detected_language"),
        "language_confidence": payload.get("language_confidence"),
        "above_cutoff": score >= production_threshold,
        "metadata": metadata,
    }


@router.get("/search")
def search_knowledge_base(
    q: str = Query(..., min_length=1, description="Keyword or phrase to search the knowledge base"),
    top_k: int = Query(10, ge=1, le=50),
    collection: Optional[str] = Query(None, description="Override the configured collection"),
    score_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Optional score floor; default shows all hits"
    ),
) -> Dict[str, Any]:
    """Search the RAG knowledge base and return matching chunks for evaluation."""
    config = load_core_config()
    rag_config = getattr(config, "rag", None)

    # System/profile baseline only — the explorer must not inherit a user's session overrides.
    system_cfg = resolve_rag_config(user_rag_config={}, system_rag_config=rag_config)
    production_threshold = system_cfg.get("pill_score_threshold") or _DEFAULT_PRODUCTION_THRESHOLD

    coll = _resolve_collection(collection, system_cfg)
    qdrant_url = _qdrant_url(config)
    timeout = system_cfg.get("timeout_seconds", 30)

    embedder, _model = get_routing_embedding_provider(config)

    # Embed up front so embedding-provider failures are attributed to the embedder,
    # not mis-reported as a Qdrant error.
    try:
        query_vector = embedder.encode(q)
    except (EmbeddingProviderUnavailableError, httpx.HTTPError) as exc:
        logger.error("RAG explorer: embedding failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Embedding provider unavailable") from exc

    try:
        # Show everything, no threshold cut (unless the admin sets a floor). Each hit is
        # flagged above/below the production cutoff so the UI can mark what the chat keeps.
        raw_results = search_qdrant(
            qdrant_url, coll, query_vector, top_k, True, False,
            score_threshold, timeout, "admin_search",
        )
        items = [_serialize_hit(h, production_threshold) for h in raw_results]
        items.sort(key=lambda i: i["score"], reverse=True)

    except httpx.TimeoutException as exc:
        logger.error("RAG explorer: Qdrant timed out", error=str(exc), collection=coll)
        raise HTTPException(status_code=504, detail="Knowledge base search timed out") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 404:
            raise HTTPException(
                status_code=404, detail=f"Collection '{coll}' not found"
            ) from exc
        logger.error("RAG explorer: Qdrant error", status=status, collection=coll)
        raise HTTPException(status_code=502, detail="Knowledge base search failed") from exc
    except httpx.RequestError as exc:
        # Connection/transport errors (e.g. Qdrant unreachable) — not TimeoutException
        # and not HTTPStatusError, so they must be handled explicitly or they'd 500.
        logger.error("RAG explorer: Qdrant unreachable", error=str(exc), collection=coll)
        raise HTTPException(status_code=502, detail="Could not reach the knowledge base") from exc

    return {
        "items": items,
        "count": len(items),
        "query": q,
        "collection": coll,
        "top_k": top_k,
        "production_threshold": production_threshold,
    }


@router.get("/collections")
def list_collections() -> Dict[str, Any]:
    """List Qdrant collections with point counts so the admin can pick + verify a target."""
    config = load_core_config()
    rag_config = getattr(config, "rag", None)
    system_cfg = resolve_rag_config(user_rag_config={}, system_rag_config=rag_config)
    default_collection = system_cfg.get("collection_name") or _DEFAULT_COLLECTION
    qdrant_url = _qdrant_url(config)
    timeout = system_cfg.get("timeout_seconds", 30)

    try:
        resp = httpx.get(f"{qdrant_url}/collections", timeout=timeout)
        resp.raise_for_status()
        names = [c["name"] for c in resp.json().get("result", {}).get("collections", [])]
    except httpx.HTTPError as exc:
        logger.error("RAG explorer: failed to list collections", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not reach the knowledge base") from exc

    collections: List[Dict[str, Any]] = []
    for name in names:
        points_count: Optional[int] = None
        try:
            detail = httpx.get(f"{qdrant_url}/collections/{name}", timeout=timeout)
            detail.raise_for_status()
            points_count = detail.json().get("result", {}).get("points_count")
        except httpx.HTTPError:
            # Best-effort — degrade to a name-only entry rather than failing the whole list.
            pass
        collections.append({"name": name, "points_count": points_count})

    return {
        "collections": collections,
        "default_collection": default_collection,
    }
