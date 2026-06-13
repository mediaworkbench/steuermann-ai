"""Unit tests for the Layer 2 argument-prep helpers extracted from the tool-calling nodes (W3.4b)."""

from universal_agentic_framework.orchestration.helpers.tool_call_args import (
    apply_web_search_max_results,
    infer_extract_webpage_url,
    coerce_tool_args,
)


# ── apply_web_search_max_results ──────────────────────────────────────

def test_web_search_max_results_injected():
    out = apply_web_search_max_results("web_search_mcp", {"query": "x"}, 5)
    assert out == {"query": "x", "max_results": 5}


def test_web_search_max_results_not_overwritten():
    out = apply_web_search_max_results("web_search_mcp", {"max_results": 3}, 5)
    assert out == {"max_results": 3}


def test_web_search_max_results_other_tool_unchanged():
    args = {"query": "x"}
    out = apply_web_search_max_results("calculator_tool", args, 5)
    assert out == {"query": "x"}


def test_web_search_max_results_no_request_count():
    out = apply_web_search_max_results("web_search_mcp", {"query": "x"}, None)
    assert out == {"query": "x"}


def test_web_search_max_results_does_not_mutate_input():
    args = {"query": "x"}
    apply_web_search_max_results("web_search_mcp", args, 5)
    assert args == {"query": "x"}  # original untouched


# ── infer_extract_webpage_url ─────────────────────────────────────────

def test_extract_url_inferred_when_missing():
    out = infer_extract_webpage_url("extract_webpage_mcp", {"selector": "#a"}, "https://e.com")
    assert out == {"selector": "#a", "request_url": "https://e.com"}


def test_extract_url_not_inferred_when_url_present():
    args = {"request_url": "https://given.com"}
    out = infer_extract_webpage_url("extract_webpage_mcp", args, "https://e.com")
    assert out == {"request_url": "https://given.com"}


def test_extract_url_detects_nested_url():
    # A URL nested in a list value counts as present — no inference.
    args = {"links": ["www.foo.com"]}
    out = infer_extract_webpage_url("extract_webpage_mcp", args, "https://e.com")
    assert "request_url" not in out


def test_extract_url_other_tool_unchanged():
    out = infer_extract_webpage_url("web_search_mcp", {"query": "x"}, "https://e.com")
    assert out == {"query": "x"}


def test_extract_url_does_not_mutate_input():
    args = {"selector": "#a"}
    infer_extract_webpage_url("extract_webpage_mcp", args, "https://e.com")
    assert args == {"selector": "#a"}


# ── coerce_tool_args ──────────────────────────────────────────────────

class _Schema:
    def __init__(self, **kwargs):
        if "bad" in kwargs:
            raise ValueError("bad arg")
        self._data = {k: v for k, v in kwargs.items() if k in ("x", "y")}

    def model_dump(self):
        return self._data


class _Tool:
    def __init__(self, schema=None):
        self.args_schema = schema


def test_coerce_no_schema_passthrough():
    args, err = coerce_tool_args(_Tool(None), {"a": 1})
    assert args == {"a": 1}
    assert err is None


def test_coerce_strips_unknown_fields():
    args, err = coerce_tool_args(_Tool(_Schema), {"x": 1, "unknown": 2})
    assert args == {"x": 1}
    assert err is None


def test_coerce_returns_error_and_original_on_failure():
    original = {"bad": 1}
    args, err = coerce_tool_args(_Tool(_Schema), original)
    assert args == original  # unchanged on failure
    assert err is not None and "bad arg" in err
