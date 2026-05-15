"""Semantic tool execution and scoring helpers."""

import re
from typing import Any, Dict, List, Optional, Tuple

from .tool_payload import record_tool_error, record_tool_success


def extract_calculator_expression(user_msg: str) -> str:
    """Extract a likely calculator expression from user input."""
    expr_match = re.search(
        r"(\d[\d\s\+\-\*/\^\(\)\.]*\d\)*|\b(?:sqrt|log|sin|cos|tan|factorial)\s*\([^)]+\))",
        user_msg,
    )
    return expr_match.group(0).strip() if expr_match else user_msg


def build_semantic_tool_kwargs(
    *,
    tool: Any,
    tool_name: str,
    user_msg: str,
    url_in_query: Optional[str],
    wants_save_to_rag: bool,
    enhanced_web_query: str,
    web_max_results: int,
    search_language: str,
    search_region: str,
    timezone: Optional[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Build kwargs for semantic tool execution and indicate if tool should be skipped."""
    tool_kwargs: Dict[str, Any] = {}

    if tool_name == "datetime_tool":
        tool_kwargs["timezone"] = timezone
    elif tool_name == "calculator_tool":
        tool_kwargs["operation"] = "evaluate"
        tool_kwargs["expression"] = user_msg
    elif tool_name == "file_ops_tool":
        tool_kwargs["operation"] = "list"
        tool_kwargs["path"] = "."
    elif tool_name == "extract_webpage_mcp":
        if not url_in_query:
            return True, {}
        tool_kwargs["query"] = url_in_query
        tool_kwargs["tool"] = "fetch_content"
        if wants_save_to_rag:
            tool_kwargs["save_to_rag"] = True
    # MCP server tools need the user query
    elif hasattr(tool, "server_url"):
        if tool_name == "web_search_mcp":
            tool_kwargs["query"] = enhanced_web_query
            tool_kwargs["region"] = search_region
            tool_kwargs["max_results"] = web_max_results
        else:
            tool_kwargs["query"] = user_msg
        if wants_save_to_rag:
            tool_kwargs["save_to_rag"] = True

    return False, tool_kwargs


def run_forced_tool(
    *,
    tool: Any,
    tool_name: str,
    run_kwargs: Dict[str, Any],
    reason: str,
    log_label: str,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
    executed_forced: set,
    logger: Any = None,
) -> None:
    """Execute a force-routed tool and record success/error envelopes."""
    try:
        result = tool._run(**run_kwargs)
        record_tool_success(
            tool_name=tool_name,
            result=result,
            reason=reason,
            tool_results=tool_results,
            tool_execution_results=tool_execution_results,
            routing_metadata=routing_metadata,
        )
        if logger:
            logger.info(f"Tool executed ({log_label})", tool=tool_name, result_length=len(str(result)))
        executed_forced.add(tool_name)
    except Exception as e:
        record_tool_error(
            tool_name=tool_name,
            error=e,
            tool_results=tool_results,
            tool_execution_results=tool_execution_results,
        )
        if logger:
            logger.error(f"Tool execution failed ({log_label})", tool=tool_name, error=str(e))


def execute_semantic_scored_tools(
    *,
    scored_tools: List[Tuple[Any, float]],
    similarity_threshold: float,
    executed_forced: set,
    mentions_calculation: bool,
    mentions_datetime: bool,
    mentions_file_ops: bool,
    user_msg: str,
    url_in_query: Optional[str],
    wants_save_to_rag: bool,
    enhanced_web_query: str,
    requested_web_results: int,
    search_language: str,
    search_region: str,
    timezone: Optional[str],
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
    logger: Any = None,
) -> None:
    """Execute scored tools that pass thresholds and intent gating."""
    for tool, similarity in scored_tools:
        tool_name = getattr(tool, "name", "unknown")
        if similarity < similarity_threshold or tool_name in executed_forced:
            continue

        # Intent gating for semantic execution (forced paths already handled above)
        if tool_name == "calculator_tool" and not mentions_calculation:
            continue
        if tool_name == "file_ops_tool" and not mentions_file_ops:
            continue
        if tool_name == "datetime_tool" and not mentions_datetime:
            continue
        # Skip web search if utility tools (calculator, datetime, file_ops) are already handling the query
        if tool_name == "web_search_mcp" and (mentions_calculation or mentions_datetime or mentions_file_ops):
            if logger:
                logger.info("Tool skipped (utility tool already matched)", tool=tool_name)
            continue

        try:
            should_skip, tool_kwargs = build_semantic_tool_kwargs(
                tool=tool,
                tool_name=tool_name,
                user_msg=user_msg,
                url_in_query=url_in_query,
                wants_save_to_rag=wants_save_to_rag,
                enhanced_web_query=enhanced_web_query,
                web_max_results=requested_web_results,
                search_language=search_language,
                search_region=search_region,
                timezone=timezone,
            )
            if should_skip:
                continue

            result = tool._run(**tool_kwargs)
            record_tool_success(
                tool_name=tool_name,
                result=result,
                reason=f"semantic match (similarity: {similarity:.2f})",
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
                routing_metadata=routing_metadata,
            )
            if logger:
                logger.info("Tool executed", tool=tool_name, result_length=len(str(result)))
        except Exception as e:
            record_tool_error(
                tool_name=tool_name,
                error=e,
                tool_results=tool_results,
                tool_execution_results=tool_execution_results,
            )
            if logger:
                logger.error("Tool execution failed", tool=tool_name, error=str(e))
