"""LangGraph node for RAG knowledge retrieval.

Extracts node_retrieve_knowledge from graph_builder.py following the same
pattern as crew_nodes.py and performance_nodes.py.
"""

from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from universal_agentic_framework.config import load_core_config, load_features_config
from universal_agentic_framework.embeddings import build_embedding_provider
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.monitoring.metrics import track_node_execution
from universal_agentic_framework.orchestration.helpers.rag_retrieval import (
    extract_rag_keyword,
    filter_and_deduplicate,
    resolve_rag_config,
    search_qdrant,
)

if TYPE_CHECKING:
    from universal_agentic_framework.orchestration.graph_builder import GraphState

logger = get_logger(__name__)


def node_retrieve_knowledge(state: GraphState) -> GraphState:
    """Retrieve relevant documents from the knowledge base (RAG)."""
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    rag_config = getattr(config, "rag", None)
    features_config = load_features_config()

    user_msg = state["messages"][-1]["content"] if state.get("messages") else ""

    logger.info("RAG node started", fork_name=fork_name, has_query=bool(user_msg))

    # --- Skip conditions (priority order) ---
    if not user_msg:
        logger.info("No query for knowledge retrieval", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    if not getattr(features_config, "rag_retrieval", True):
        logger.info("RAG disabled via features flag", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    if rag_config is not None and not rag_config.enabled:
        logger.info("RAG disabled via config", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    # Per-session user toggle: checked before intent skip so explicit user override takes priority
    user_settings = state.get("user_settings", {})
    user_rag_config = user_settings.get("rag_config", {})
    if user_rag_config and not user_rag_config.get("enabled", True):
        logger.info("RAG disabled by user setting", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    # Intent-based short-circuit: skip Qdrant entirely for trivial queries
    intents = state.get("prefilter_intents") or {}
    if intents.get("skip_rag"):
        logger.info("RAG skipped (trivial intent)", fork_name=fork_name)
        state["knowledge_context"] = []
        return state

    # All skip paths passed — Qdrant will be queried this turn
    state["rag_attempted"] = True
    logger.info("Retrieving knowledge", fork_name=fork_name, query_length=len(user_msg))

    with track_node_execution(fork_name, "retrieve_knowledge"):
        try:
            embedder = build_embedding_provider(
                model_name=config.llm.get_role_model_name("embedding", config.fork.language),
                dimension=config.memory.embeddings.dimension,
                provider_type=config.llm.get_embedding_provider_type(),
                remote_endpoint=config.llm.get_embedding_remote_endpoint(),
            )
            query_embedding = embedder.encode(user_msg)

            logger.info("RAG: Generated embedding", embedding_size=len(query_embedding))

            # Resolve effective config: system baseline, then user overrides on top
            cfg = resolve_rag_config(user_rag_config, rag_config)
            if cfg["collection_name"] == "framework":
                logger.warning(
                    "RAG: collection_name not set in config, falling back to 'framework'",
                    fork_name=fork_name,
                )

            qdrant_url = f"http://{config.memory.vector_store.host}:{config.memory.vector_store.port}"

            # Thin local adapter — captures qdrant_url and cfg so call sites stay concise.
            # The actual search logic lives in the module-level search_qdrant helper.
            def _search(vector: list[float], label: str) -> list[dict]:
                return search_qdrant(
                    qdrant_url, cfg["collection_name"], vector,
                    cfg["top_k"], cfg["with_payload"], cfg["with_vector"],
                    cfg["score_threshold"], cfg["timeout_seconds"], label,
                )

            raw_results = _search(query_embedding, "full_query")

            # Dual-query: focused keyword search merged with full-query results
            keyword = extract_rag_keyword(user_msg)
            if keyword and keyword != user_msg.lower():
                kw_results = _search(embedder.encode(keyword), "keyword")
                if kw_results:
                    raw_results = raw_results + kw_results

            # Client-side safety floor: server-side score_threshold runs when configured;
            # 0.6 is the fallback floor when no threshold is set, to drop noise.
            min_score = cfg["score_threshold"] if cfg["score_threshold"] is not None else 0.6
            knowledge_docs = filter_and_deduplicate(raw_results, min_score, cfg["top_k"])

            state["knowledge_context"] = knowledge_docs
            state["rag_doc_count"] = len(knowledge_docs)
            logger.info(
                "Knowledge retrieved successfully",
                fork_name=fork_name,
                results_count=len(knowledge_docs),
                top_score=knowledge_docs[0]["score"] if knowledge_docs else 0.0,
            )

        except httpx.TimeoutException as e:
            logger.error("RAG: Qdrant request timed out", error=str(e), fork_name=fork_name)
            state["knowledge_context"] = []
            state["rag_doc_count"] = 0
        except httpx.HTTPStatusError as e:
            logger.error(
                "RAG: Qdrant returned HTTP error",
                status=e.response.status_code,
                fork_name=fork_name,
            )
            state["knowledge_context"] = []
            state["rag_doc_count"] = 0
        except Exception as e:
            logger.error("RAG: Knowledge retrieval failed", error=str(e), fork_name=fork_name, exc_info=True)
            state["knowledge_context"] = []
            state["rag_doc_count"] = 0

    return state
