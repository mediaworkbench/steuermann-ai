"""Unit tests for tool-result provenance envelope (args capture, sanitization, detail builder)."""

from universal_agentic_framework.orchestration.helpers.tool_payload import (
    build_tool_results_detail,
    error_tool_payload,
    normalize_tool_payload,
    record_tool_error,
    record_tool_success,
)


def test_normalize_captures_sanitized_args():
    env = normalize_tool_payload("ok", args={"query": "weather", "max_results": 5})
    assert env["args"] == {"query": "weather", "max_results": 5}
    assert env["status"] == "success"


def test_secret_args_are_redacted():
    env = normalize_tool_payload("ok", args={"api_key": "sk-123", "Authorization": "Bearer t", "q": "hi"})
    assert env["args"]["api_key"] == "[redacted]"
    assert env["args"]["Authorization"] == "[redacted]"
    assert env["args"]["q"] == "hi"


def test_long_arg_value_is_truncated():
    env = normalize_tool_payload("ok", args={"q": "x" * 500})
    assert len(env["args"]["q"]) <= 200
    assert env["args"]["q"].endswith("…")


def test_oversized_args_are_capped():
    big = {f"k{i}": "y" * 100 for i in range(50)}
    env = normalize_tool_payload("ok", args=big)
    # Collapsed into a single bounded blob rather than the full structure.
    assert set(env["args"].keys()) == {"_"}
    assert len(env["args"]["_"]) <= 600


def test_none_args_yield_none():
    assert normalize_tool_payload("ok", args=None)["args"] is None
    assert normalize_tool_payload("ok", args={})["args"] is None


def test_build_detail_projects_bounded_client_payload():
    exec_results = {
        "web_search_mcp": normalize_tool_payload("Sunny, 21C", args={"query": "weather"}),
        "broken_tool": error_tool_payload("Tool execution failed: boom", args={"x": 1}),
    }
    detail = build_tool_results_detail(exec_results)
    assert [d["name"] for d in detail] == ["web_search_mcp", "broken_tool"]

    ok = detail[0]
    assert ok["status"] == "success"
    assert ok["args"] == {"query": "weather"}
    assert ok["output"] == "Sunny, 21C"
    assert "error" not in ok
    # Heavy internal fields are dropped from the client payload.
    assert "data" not in ok
    assert "output_text" not in ok

    err = detail[1]
    assert err["status"] == "error"
    assert err["error"] == "Tool execution failed: boom"
    assert err["args"] == {"x": 1}
    # Errors carry the message once (as `error`), not duplicated into `output`.
    assert "output" not in err


def test_build_detail_truncates_large_output():
    env = normalize_tool_payload("z" * 5000)
    detail = build_tool_results_detail({"dump_tool": env})
    assert len(detail[0]["output"]) <= 1500


def test_build_detail_handles_empty_or_none():
    assert build_tool_results_detail(None) == []
    assert build_tool_results_detail({}) == []


def test_record_helpers_thread_args_into_envelope():
    tool_results: dict = {}
    tool_exec: dict = {}
    routing: dict = {}
    record_tool_success(
        tool_name="calc",
        result="42",
        reason="native",
        tool_results=tool_results,
        tool_execution_results=tool_exec,
        routing_metadata=routing,
        args={"expression": "6*7"},
    )
    assert tool_exec["calc"]["args"] == {"expression": "6*7"}
    assert tool_results["calc"] == "42"

    record_tool_error(
        tool_name="calc",
        error=ValueError("bad"),
        tool_results=tool_results,
        tool_execution_results=tool_exec,
        args={"expression": "1/0"},
    )
    assert tool_exec["calc"]["status"] == "error"
    assert tool_exec["calc"]["args"] == {"expression": "1/0"}
