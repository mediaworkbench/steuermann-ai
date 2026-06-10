"""Tool execution result handling and envelope helpers."""

import ast
import json
from typing import Any, Dict, List, Optional

# Frontend-facing payload bounds. Tool results can be large (OCR dumps, web
# extractions); the Outputs tab only needs a readable preview, so truncate
# aggressively before the data ever leaves the graph.
_ARGS_VALUE_MAX = 200      # per-string-value cap inside args
_ARGS_JSON_MAX = 600       # whole sanitized-args JSON cap
_OUTPUT_MAX = 1500         # tool output preview cap
_ERROR_MAX = 500           # error message cap

# Argument keys whose values are redacted before they reach the client. Matched
# as a case-insensitive substring, so e.g. ``api_key`` and ``x-auth-token`` hit.
_SECRET_KEY_HINTS = (
    "api_key", "apikey", "token", "secret", "password", "passwd",
    "authorization", "auth", "access_key", "bearer", "credential",
)

_REDACTED = "[redacted]"


def _truncate(value: Optional[str], limit: int) -> Optional[str]:
    """Truncate a string to ``limit`` chars with an ellipsis marker."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _is_secret_key(key: Any) -> bool:
    k = str(key).lower()
    return any(hint in k for hint in _SECRET_KEY_HINTS)


def _sanitize_args(args: Any) -> Optional[Dict[str, Any]]:
    """Produce a bounded, secret-redacted, JSON-safe copy of tool call args.

    Secret-looking keys are redacted; string values are truncated; the whole
    structure is capped so an oversized arg can't bloat the metadata payload.
    Returns ``None`` when there are no meaningful args to show.
    """
    if not args or not isinstance(args, dict):
        return None

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return _truncate(value, _ARGS_VALUE_MAX)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): (_REDACTED if _is_secret_key(k) else _clean(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value][:20]
        return _truncate(repr(value), _ARGS_VALUE_MAX)

    cleaned = {str(k): (_REDACTED if _is_secret_key(k) else _clean(v)) for k, v in args.items()}

    # Hard cap on the serialized size so a huge arg payload can't slip through.
    try:
        if len(json.dumps(cleaned, ensure_ascii=False)) > _ARGS_JSON_MAX:
            return {"_": _truncate(json.dumps(cleaned, ensure_ascii=False), _ARGS_JSON_MAX)}
    except (TypeError, ValueError):
        return {"_": _truncate(repr(cleaned), _ARGS_JSON_MAX)}
    return cleaned or None


def normalize_tool_payload(result: Any, args: Any = None) -> Dict[str, Any]:
    """Create a structured, backward-compatible envelope for tool outputs.

    The framework still uses string-based `tool_results` for prompt injection,
    while this envelope enables typed handling for future planner/executor steps.
    ``args`` (the invocation arguments) is captured in sanitized form for the
    Inspector/Outputs provenance surfaces.
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
        "args": _sanitize_args(args),
        "data": parsed_data,
        "output_text": output_text,
        "error": None,
        "sources": [],
    }


def error_tool_payload(error_message: str, args: Any = None) -> Dict[str, Any]:
    """Create an error envelope for tool execution failures."""
    return {
        "status": "error",
        "summary": error_message,
        "args": _sanitize_args(args),
        "data": None,
        "output_text": error_message,
        "error": error_message,
        "sources": [],
    }


def build_tool_results_detail(
    tool_execution_results: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Project the internal envelope into a bounded, client-safe detail list.

    Used by the streaming `metadata` event to surface real tool invocations
    (args + result preview + status) in the workspace Outputs tab. Heavy fields
    (`data`, full `output_text`) are dropped/truncated so the metadata payload
    stays small and free of secrets.
    """
    detail: List[Dict[str, Any]] = []
    for name, env in (tool_execution_results or {}).items():
        if not isinstance(env, dict):
            continue
        is_error = env.get("status") == "error"
        item: Dict[str, Any] = {
            "name": name,
            "status": "error" if is_error else "success",
            "summary": _truncate(env.get("summary"), 300),
        }
        if env.get("args"):
            item["args"] = env["args"]
        if is_error:
            # For errors `output_text` is just the error message — carry it once as `error`.
            item["error"] = _truncate(env.get("error") or env.get("output_text"), _ERROR_MAX)
        else:
            item["output"] = _truncate(env.get("output_text"), _OUTPUT_MAX)
        detail.append(item)
    return detail


def record_tool_success(
    *,
    tool_name: str,
    result: Any,
    reason: str,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
    args: Any = None,
) -> None:
    """Record successful tool execution in all tracking dictionaries."""
    tool_results[tool_name] = str(result)
    tool_execution_results[tool_name] = normalize_tool_payload(result, args=args)
    routing_metadata[tool_name] = reason


def record_tool_error(
    *,
    tool_name: str,
    error: Exception,
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    args: Any = None,
) -> None:
    """Record tool execution error in all tracking dictionaries."""
    error_msg = f"Tool execution failed: {str(error)}"
    tool_results[tool_name] = error_msg
    tool_execution_results[tool_name] = error_tool_payload(error_msg, args=args)
