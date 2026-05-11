"""Tool execution result handling and envelope helpers."""

import ast
from typing import Any, Dict


def normalize_tool_payload(result: Any) -> Dict[str, Any]:
    """Create a structured, backward-compatible envelope for tool outputs.

    The framework still uses string-based `tool_results` for prompt injection,
    while this envelope enables typed handling for future planner/executor steps.
    """
    output_text = str(result)
    parsed_data: Any = None

    # Best-effort parsing for tools that return Python-dict-like strings.
    if isinstance(result, (dict, list)):
        parsed_data = result
    elif output_text.startswith("{") or output_text.startswith("["):
        try:
            parsed_data = ast.literal_eval(output_text)
        except Exception:
            parsed_data = None

    summary = output_text.replace("\n", " ").strip()
    if len(summary) > 300:
        summary = summary[:297] + "..."

    return {
        "status": "success",
        "summary": summary,
        "data": parsed_data,
        "output_text": output_text,
        "error": None,
        "sources": [],
    }


def error_tool_payload(error_message: str) -> Dict[str, Any]:
    """Create an error envelope for tool execution failures."""
    return {
        "status": "error",
        "summary": error_message,
        "data": None,
        "output_text": error_message,
        "error": error_message,
        "sources": [],
    }


def record_tool_success(
    *,
    tool_name: str,
    result: Any,
    reason: str,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
) -> None:
    """Record successful tool execution in all tracking dictionaries."""
    tool_results[tool_name] = str(result)
    tool_execution_results[tool_name] = normalize_tool_payload(result)
    routing_metadata[tool_name] = reason


def record_tool_error(
    *,
    tool_name: str,
    error: Exception,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
) -> None:
    """Record tool execution error in all tracking dictionaries."""
    error_msg = f"Tool execution failed: {str(error)}"
    tool_results[tool_name] = error_msg
    tool_execution_results[tool_name] = error_tool_payload(error_msg)
