"""Pure argument-preparation helpers shared by the three Layer 2 tool-calling nodes.

`node_call_tools_native` / `_structured` / `_react` each had byte-for-byte copies of the
web-search ``max_results`` patch and the schema-validation step; the native node additionally
infers a missing ``request_url`` for webpage extraction. These are pure (no model, no state),
so they extract cleanly without touching the test suite's ``graph_builder.get_model`` patch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

_URL_PREFIX_RE = re.compile(r"^(https?://|www\.)")


def apply_web_search_max_results(
    tool_name: str,
    tool_args: Any,
    requested_results: Optional[int],
) -> Any:
    """Inject ``max_results`` into a web_search call when the user asked for N results and the
    model omitted it. Returns a new dict (never mutates the input) or the args unchanged."""
    if tool_name == "web_search_mcp" and requested_results and "max_results" not in (tool_args or {}):
        tool_args = dict(tool_args)
        tool_args["max_results"] = requested_results
    return tool_args


def _contains_url_arg(value: Any) -> bool:
    """True if any (possibly nested) string value looks like a URL."""
    if isinstance(value, str):
        candidate = value.strip().strip('"\'')
        return bool(_URL_PREFIX_RE.match(candidate))
    if isinstance(value, dict):
        return any(_contains_url_arg(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_url_arg(v) for v in value)
    return False


def infer_extract_webpage_url(tool_name: str, tool_args: Any, url_in_query: Optional[str]) -> Any:
    """For extract_webpage_mcp calls that carry no URL argument, fall back to the URL detected
    in the user message. Returns a new dict (never mutates) or the args unchanged."""
    if tool_name == "extract_webpage_mcp" and isinstance(tool_args, dict):
        if not _contains_url_arg(tool_args) and url_in_query:
            tool_args = dict(tool_args)
            tool_args["request_url"] = url_in_query
    return tool_args


def coerce_tool_args(tool_obj: Any, tool_args: Any) -> Tuple[Any, Optional[str]]:
    """Validate/coerce args against the tool's ``args_schema`` (also strips unknown fields).

    Returns ``(coerced_args, None)`` on success (or when the tool has no schema), or
    ``(original_args, error_message)`` on validation failure — the caller decides how to react
    (re-prompt, mark a parse error, or ignore).
    """
    schema = getattr(tool_obj, "args_schema", None)
    if not schema:
        return tool_args, None
    try:
        return schema(**tool_args).model_dump(), None
    except Exception as val_err:  # noqa: BLE001 - surfaced to the caller as a string
        return tool_args, str(val_err)
