"""Tool preparation and scoring for graph routing."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .semantic_execution import extract_calculator_expression, run_forced_tool


logger = logging.getLogger(__name__)


def prepare_scored_tools_with_forced_execution(
    *,
    loaded_tools: List[Any],
    config: Any,
    user_msg: str,
    embedding_model_name: str,
    embedding_provider: Any,
    query_embedding: Any,
    similarity_threshold: float,
    mentions_datetime: bool,
    mentions_calculation: bool,
    mentions_file_ops: bool,
    mentions_web_search: bool,
    enhanced_web_query: str,
    requested_web_results: int,
    search_region: str,
    url_in_query: Optional[str],
    wants_save_to_rag: bool,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
    executed_forced: set,
    score_tool_func: Any,
) -> List[Tuple[Any, float]]:
    """Handle forced tool paths and collect semantically scored candidates.
    
    Forced execution applies to tools that match explicit intent patterns
    (datetime, calculator, file ops, URL extraction, explicit web search).
    
    Semantic scoring is applied to remaining tools for ranking.
    """
    scored_tools: List[Tuple[Any, float]] = []

    for tool in loaded_tools:
        tool_name = getattr(tool, "name", "unknown")
        tool_desc = getattr(tool, "description", "")

        if not tool_desc:
            logger.info("Tool skipped (no description)", extra={"tool": tool_name})
            continue

        # Force-run datetime tool when the query mentions date/time
        if tool_name == "datetime_tool" and mentions_datetime:
            run_forced_tool(
                tool=tool,
                tool_name=tool_name,
                run_kwargs={"timezone": getattr(getattr(config, "fork", None), "timezone", None)},
                reason="date/time pattern detected",
                log_label="forced datetime",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                logger=logger,
            )
            continue

        # Force-run calculator when math expression detected
        if tool_name == "calculator_tool" and mentions_calculation:
            run_forced_tool(
                tool=tool,
                tool_name=tool_name,
                run_kwargs={"operation": "evaluate", "expression": extract_calculator_expression(user_msg)},
                reason="mathematical expression detected",
                log_label="forced calculator",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                logger=logger,
            )
            continue

        # Force-run file_ops when file operation keywords detected
        if tool_name == "file_ops_tool" and mentions_file_ops:
            run_forced_tool(
                tool=tool,
                tool_name=tool_name,
                run_kwargs={"operation": "list", "path": "."},
                reason="file operation keywords detected",
                log_label="forced file_ops",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                logger=logger,
            )
            continue

        # Force-run extract tool when a URL is present
        if tool_name == "extract_webpage_mcp" and url_in_query:
            run_forced_tool(
                tool=tool,
                tool_name=tool_name,
                run_kwargs={"query": url_in_query, "save_to_rag": wants_save_to_rag},
                reason=f"URL detected: {url_in_query}",
                log_label="forced URL",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                logger=logger,
            )
            continue

        # Force-run web search when user explicitly asks to search the web
        if tool_name == "web_search_mcp" and mentions_web_search and not url_in_query:
            run_forced_tool(
                tool=tool,
                tool_name=tool_name,
                run_kwargs={
                    "query": enhanced_web_query,
                    "region": search_region,
                    "max_results": requested_web_results,
                    "save_to_rag": wants_save_to_rag,
                },
                reason="explicit web-search request detected",
                log_label="forced web_search",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
                executed_forced=executed_forced,
                logger=logger,
            )
            continue

        # Skip web search if the user provided a URL
        if tool_name == "web_search_mcp" and url_in_query:
            logger.info("Tool skipped (URL query)", extra={"tool": tool_name})
            continue

        # Skip extract tool if no URL is present
        if tool_name == "extract_webpage_mcp" and not url_in_query:
            logger.info("Tool skipped (no URL provided)", extra={"tool": tool_name})
            continue

        # Score remaining tools for semantic relevance
        similarity = score_tool_func(
            user_msg_lower=user_msg.lower(),
            tool_name=tool_name,
            tool_desc=tool_desc,
            embedding_provider=embedding_provider,
            query_embedding=query_embedding,
            embedding_model_name=embedding_model_name,
        )

        logger.info(
            "Tool scored",
            extra={
                "tool": tool_name,
                "similarity": round(similarity, 4),
                "threshold": similarity_threshold,
            },
        )
        scored_tools.append((tool, similarity))

    return scored_tools


def apply_top_k_scored_tools(
    scored_tools: List[Tuple[Any, float]],
    top_k: Optional[int],
) -> List[Tuple[Any, float]]:
    """Apply optional top-k selection to scored tools."""
    if top_k is None or top_k <= 0:
        return scored_tools
    return sorted(scored_tools, key=lambda x: x[1], reverse=True)[:top_k]
