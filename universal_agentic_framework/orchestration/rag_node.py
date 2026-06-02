"""LangGraph node for RAG knowledge retrieval.

Extracts node_retrieve_knowledge from graph_builder.py following the same
pattern as crew_nodes.py and performance_nodes.py.
"""

from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from universal_agentic_framework.config import load_core_config, load_features_config
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.monitoring.metrics import track_node_execution
from universal_agentic_framework.orchestration.helpers.embedding_provider import get_routing_embedding_provider
from universal_agentic_framework.orchestration.helpers.rag_retrieval import (
    extract_rag_keyword,
    filter_and_deduplicate,
    resolve_rag_config,
    search_qdrant,
)

if TYPE_CHECKING:
    from universal_agentic_framework.orchestration.graph_builder import GraphState

logger = get_logger(__name__)


def _rewrite_query_for_rag(user_message: str, config, lang: str, num_variants: int = 1) -> list[str]:
    """Rewrite/expand user query for better semantic retrieval via auxiliary model. Fails open."""
    try:
        provider = config.llm.get_role_provider("auxiliary")
        api_base = str(provider.api_base or "").rstrip("/")
        model_name = config.llm.get_role_model_name("auxiliary", lang)
        bare_model = model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
        if num_variants <= 1:
            prompt = (
                "Rewrite this query to improve semantic document retrieval. "
                "Resolve pronouns, expand abbreviations, keep it concise. "
                "Output ONLY the rewritten query.\n\n"
                f"Query: {user_message}\nRewritten:"
            )
            max_tokens = 128
        else:
            prompt = (
                f"Generate exactly {num_variants} semantically varied search queries for document retrieval. "
                "Resolve pronouns, expand abbreviations. "
                f"Output ONLY the {num_variants} queries, one per line, no numbering or extra text.\n\n"
                f"Query: {user_message}"
            )
            max_tokens = 80 * num_variants
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {provider.api_key or 'no-key'}"},
                json={"model": bare_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if num_variants <= 1:
                return [content] if content else [user_message]
            variants = [line.strip() for line in content.splitlines() if line.strip()]
            return variants[:num_variants] if variants else [user_message]
    except Exception:
        return [user_message]


def node_retrieve_knowledge(state: GraphState) -> GraphState:
    """Retrieve relevant documents from the knowledge base (RAG)."""
    config = load_core_config()
    profile_name = getattr(config.profile, "name", "default-profile")
    rag_config = getattr(config, "rag", None)
    features_config = load_features_config()

    user_msg = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""

    logger.info("RAG node started", profile_name=profile_name, has_query=bool(user_msg))

    # --- Skip conditions (priority order) ---
    if not user_msg:
        logger.info("No query for knowledge retrieval", profile_name=profile_name)
        state["knowledge_context"] = []
        return state

    if not getattr(features_config, "rag_retrieval", True):
        logger.info("RAG disabled via features flag", profile_name=profile_name)
        state["knowledge_context"] = []
        return state

    if rag_config is not None and not rag_config.enabled:
        logger.info("RAG disabled via config", profile_name=profile_name)
        state["knowledge_context"] = []
        return state

    # Per-session user toggle: checked before intent skip so explicit user override takes priority
    user_settings = state.get("user_settings", {})
    user_rag_config = user_settings.get("rag_config", {})
    if user_rag_config and not user_rag_config.get("enabled", True):
        logger.info("RAG disabled by user setting", profile_name=profile_name)
        state["knowledge_context"] = []
        return state

    # Intent-based short-circuit: skip Qdrant entirely for trivial queries
    intents = state.get("prefilter_intents") or {}
    if intents.get("skip_rag"):
        logger.info("RAG skipped (trivial intent)", profile_name=profile_name)
        state["knowledge_context"] = []
        return state

    # All skip paths passed — Qdrant will be queried this turn
    state["rag_attempted"] = True

    _qr_cfg = getattr(rag_config, "query_rewriting", None)
    queries = [user_msg]  # default: single original query
    if _qr_cfg is not None and _qr_cfg.enabled:
        lang = state.get("language") or getattr(config.profile, "language", "en")
        num_variants = getattr(_qr_cfg, "num_variants", 1)
        queries = _rewrite_query_for_rag(user_msg, config, lang, num_variants)
        state.pop("query_embedding", None)  # invalidate prefilter cache; force re-embed of rewritten queries
        logger.info("RAG: query rewritten for retrieval", num_variants=len(queries), first_query_length=len(queries[0]))

    logger.info("Retrieving knowledge", profile_name=profile_name, num_queries=len(queries))

    with track_node_execution(profile_name, "retrieve_knowledge"):
        try:
            # Reuse the module-level cached provider (same instance as node_prefilter_tools).
            embedder, _ = get_routing_embedding_provider(config)

            # Embed query variants. Single variant reuses the prefilter cache when available.
            if len(queries) == 1:
                single_embedding = state.get("query_embedding") or embedder.encode(queries[0])
                logger.info("RAG: Got embedding", embedding_size=len(single_embedding), from_cache=bool(state.get("query_embedding")))
                query_embeddings = [single_embedding]
            else:
                # Batch encode all variants; encoder accepts list[str] → list[list[float]]
                query_embeddings = embedder.encode(queries)
                logger.info("RAG: Got batch embeddings", num_embeddings=len(query_embeddings))

            # Resolve effective config: system baseline, then user overrides on top
            cfg = resolve_rag_config(user_rag_config, rag_config)
            if cfg["collection_name"] is None:
                logger.warning(
                    "RAG: collection_name not set in config, falling back to 'framework'",
                    profile_name=profile_name,
                )
                cfg["collection_name"] = "framework"

            qdrant_url = f"http://{config.memory.vector_store.host}:{config.memory.vector_store.port}"

            threshold = cfg.get("pill_score_threshold") or 0.72

            # Thin local adapter — captures qdrant_url and cfg so call sites stay concise.
            # The actual search logic lives in the module-level search_qdrant helper.
            def _search(vector: list[float], label: str) -> list[dict]:
                return search_qdrant(
                    qdrant_url, cfg["collection_name"], vector,
                    cfg["top_k"], cfg["with_payload"], cfg["with_vector"],
                    threshold, cfg["timeout_seconds"], label,
                )

            # Union results from all query variants
            raw_results: list[dict] = []
            for i, embedding in enumerate(query_embeddings):
                label = f"variant_{i}" if len(query_embeddings) > 1 else "full_query"
                raw_results.extend(_search(embedding, label))

            # Dual-query: focused keyword search merged with variant results (keyed on primary query)
            keyword = extract_rag_keyword(queries[0])
            if keyword and keyword != queries[0].lower():
                kw_results = _search(embedder.encode(keyword), "keyword")
                if kw_results:
                    raw_results = raw_results + kw_results

            knowledge_docs = filter_and_deduplicate(raw_results, threshold, cfg["top_k"])

            state["knowledge_context"] = knowledge_docs
            state["rag_doc_count"] = len(knowledge_docs)
            logger.info(
                "Knowledge retrieved successfully",
                profile_name=profile_name,
                results_count=len(knowledge_docs),
                top_score=knowledge_docs[0]["score"] if knowledge_docs else 0.0,
            )

        except httpx.TimeoutException as e:
            logger.error("RAG: Qdrant request timed out", error=str(e), profile_name=profile_name)
            state["knowledge_context"] = []
            state["rag_doc_count"] = 0
        except httpx.HTTPStatusError as e:
            logger.error(
                "RAG: Qdrant returned HTTP error",
                status=e.response.status_code,
                profile_name=profile_name,
            )
            state["knowledge_context"] = []
            state["rag_doc_count"] = 0
        # EmbeddingProviderUnavailableError and other non-Qdrant exceptions propagate
        # so that provider outages surface as hard errors rather than silent empty context.

    return state
