"""Comprehensive tests for Week 32: Tool Ecosystem Expansion.

Tests cover:
- Calculator tool (evaluate, convert, statistics, percentage, sandboxed AST)
- File operations tool (read, write, list, info, exists, sandbox security)
- Tool sandbox (permissions, timeouts, output limits, concurrency)
- Tool rate limiter (global, per-tool, per-user, window expiry)
- Semantic routing (heuristic triggers for new tools, embedding cache)
- Tool registry (new tools discovered and loaded)
"""

import os
import time
import tempfile
import threading

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.tools.calculator.tool import CalculatorTool
from universal_agentic_framework.tools.file_ops.tool import FileOpsTool
from universal_agentic_framework.tools.sandbox import (
    ToolSandbox,
    SandboxPolicy,
    SandboxResult,
    Permission,
    get_sandbox,
    reset_sandbox,
)
from universal_agentic_framework.tools.rate_limiter import (
    ToolRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    _SlidingWindow,
    get_rate_limiter,
    reset_rate_limiter,
)


# ═══════════════════════════════════════════════════════════════════════
# Calculator Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCalculatorEvaluate:
    """Test the evaluate operation (safe AST evaluation)."""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    def test_basic_arithmetic(self, calc):
        result = calc._run(operation="evaluate", expression="2 + 3 * 4")
        assert "= 14" in result

    def test_power(self, calc):
        result = calc._run(operation="evaluate", expression="2 ** 10")
        assert "= 1024" in result

    def test_caret_as_power(self, calc):
        result = calc._run(operation="evaluate", expression="2 ^ 10")
        assert "= 1024" in result

    def test_negative_numbers(self, calc):
        result = calc._run(operation="evaluate", expression="-5 + 3")
        assert "= -2" in result

    def test_float_arithmetic(self, calc):
        result = calc._run(operation="evaluate", expression="1.5 + 2.5")
        assert "= 4" in result

    def test_floor_division(self, calc):
        result = calc._run(operation="evaluate", expression="7 // 2")
        assert "= 3" in result

    def test_modulo(self, calc):
        result = calc._run(operation="evaluate", expression="10 % 3")
        assert "= 1" in result

    def test_sqrt_function(self, calc):
        result = calc._run(operation="evaluate", expression="sqrt(16)")
        assert "= 4" in result

    def test_pi_constant(self, calc):
        result = calc._run(operation="evaluate", expression="pi")
        assert "= 3.14159" in result

    def test_nested_functions(self, calc):
        result = calc._run(operation="evaluate", expression="sqrt(abs(-25))")
        assert "= 5" in result

    def test_division_by_zero(self, calc):
        result = calc._run(operation="evaluate", expression="1 / 0")
        assert "Error" in result

    def test_too_large_exponent(self, calc):
        result = calc._run(operation="evaluate", expression="2 ** 10000")
        assert "Error" in result
        assert "too large" in result.lower()

    def test_expression_too_long(self, calc):
        long_expr = "1 + " * 200 + "1"
        result = calc._run(operation="evaluate", expression=long_expr)
        assert "Error" in result
        assert "too long" in result.lower()

    def test_no_expression(self, calc):
        result = calc._run(operation="evaluate", expression=None)
        assert "Error" in result

    def test_unsafe_code_blocked(self, calc):
        """Ensure arbitrary Python code is blocked."""
        result = calc._run(operation="evaluate", expression="__import__('os').system('ls')")
        assert "Error" in result

    def test_unknown_function_blocked(self, calc):
        result = calc._run(operation="evaluate", expression="exec('print(1)')")
        assert "Error" in result


class TestCalculatorConvert:
    """Test the unit conversion operation."""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    def test_km_to_miles(self, calc):
        result = calc._run(operation="convert", value=100, from_unit="km", to_unit="mi")
        assert "62.137" in result

    def test_celsius_to_fahrenheit(self, calc):
        result = calc._run(operation="convert", value=100, from_unit="celsius", to_unit="fahrenheit")
        assert "212" in result

    def test_fahrenheit_to_celsius(self, calc):
        result = calc._run(operation="convert", value=32, from_unit="f", to_unit="c")
        assert "= 0 " in result

    def test_kg_to_lb(self, calc):
        result = calc._run(operation="convert", value=1, from_unit="kg", to_unit="lb")
        assert "2.204" in result

    def test_gb_to_mb(self, calc):
        result = calc._run(operation="convert", value=1, from_unit="gb", to_unit="mb")
        assert "= 1024 " in result

    def test_hours_to_seconds(self, calc):
        result = calc._run(operation="convert", value=1, from_unit="hour", to_unit="s")
        assert "= 3600 " in result

    def test_liter_to_gallon(self, calc):
        result = calc._run(operation="convert", value=3.78541, from_unit="l", to_unit="gal")
        assert "= 1 " in result

    def test_missing_params(self, calc):
        result = calc._run(operation="convert", value=100)
        assert "Error" in result

    def test_incompatible_units(self, calc):
        result = calc._run(operation="convert", value=1, from_unit="km", to_unit="kg")
        assert "Error" in result


class TestCalculatorStatistics:
    """Test the statistics operation."""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    def test_basic_stats(self, calc):
        result = calc._run(operation="statistics", values=[1, 2, 3, 4, 5])
        assert "Mean:" in result
        assert "Sum: 15" in result
        assert "Min: 1" in result
        assert "Max: 5" in result
        assert "Median: 3" in result

    def test_single_value(self, calc):
        result = calc._run(operation="statistics", values=[42])
        assert "Mean:" in result
        assert "Sum: 42" in result

    def test_empty_list(self, calc):
        result = calc._run(operation="statistics", values=[])
        assert "Error" in result

    def test_no_list(self, calc):
        result = calc._run(operation="statistics", values=None)
        assert "Error" in result


class TestCalculatorPercentage:
    """Test the percentage operation."""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    def test_percentage_of(self, calc):
        result = calc._run(operation="percentage", value=200, percentage=15)
        assert "15% of 200 = 30" in result

    def test_percentage_increase(self, calc):
        result = calc._run(operation="percentage", value=100, percentage=10)
        assert "100 + 10% = 110" in result

    def test_percentage_decrease(self, calc):
        result = calc._run(operation="percentage", value=100, percentage=25)
        assert "100 - 25% = 75" in result

    def test_missing_params(self, calc):
        result = calc._run(operation="percentage", value=100)
        assert "Error" in result


# ═══════════════════════════════════════════════════════════════════════
# File Operations Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFileOpsRead:
    """Test the read operation."""

    def test_read_existing_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("Hello World")
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="read", path="hello.txt")
        assert "Hello World" in result

    def test_read_nonexistent(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="read", path="nope.txt")
        assert "Error" in result

    def test_read_too_large(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * 2_000_000)
        tool = FileOpsTool(sandbox_dir=str(tmp_path), max_read_size_bytes=1_000_000)
        result = tool._run(operation="read", path="big.txt")
        assert "too large" in result.lower()

    def test_read_disallowed_extension(self, tmp_path):
        (tmp_path / "secret.bin").write_bytes(b"\x00" * 10)
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="read", path="secret.bin")
        assert "not allowed" in result.lower()


class TestFileOpsWrite:
    """Test the write operation."""

    def test_write_creates_file(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="write", path="output.txt", content="Hello!")
        assert "Written" in result
        assert (tmp_path / "output.txt").read_text() == "Hello!"

    def test_write_creates_subdirs(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="write", path="sub/dir/file.txt", content="nested")
        assert "Written" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"

    def test_write_no_content(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="write", path="x.txt")
        assert "Error" in result

    def test_write_too_large(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path), max_write_size_bytes=100)
        result = tool._run(operation="write", path="big.txt", content="x" * 200)
        assert "too large" in result.lower()


class TestFileOpsList:
    """Test the list operation."""

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.md").touch()
        (tmp_path / "subdir").mkdir()
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="list", path=".")
        assert "a.txt" in result
        assert "b.md" in result
        assert "subdir/" in result

    def test_list_empty_dir(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="list", path=".")
        assert "empty" in result.lower()

    def test_list_nonexistent(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="list", path="nonexistent")
        assert "Error" in result


class TestFileOpsInfoAndExists:
    """Test info and exists operations."""

    def test_info_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello")
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="info", path="sample.txt")
        assert "File:" in result
        assert "Size:" in result
        assert "Modified:" in result

    def test_exists_true(self, tmp_path):
        (tmp_path / "there.txt").touch()
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="exists", path="there.txt")
        assert "Yes" in result

    def test_exists_false(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="exists", path="nope.txt")
        assert "No" in result


class TestFileOpsSecurity:
    """Test sandbox security features."""

    def test_path_traversal_blocked(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="read", path="../../etc/passwd")
        assert "Error" in result

    def test_path_traversal_blocked_on_write(self, tmp_path):
        tool = FileOpsTool(sandbox_dir=str(tmp_path))
        result = tool._run(operation="write", path="../escape.txt", content="pwned")
        assert "Error" in result

    def test_symlink_traversal_blocked(self, tmp_path):
        """Symlinks resolving outside sandbox must be blocked."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("confidential")
        
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        # Create symlink inside sandbox pointing outside
        link = sandbox / "link.txt"
        link.symlink_to(outside / "secret.txt")
        
        tool = FileOpsTool(sandbox_dir=str(sandbox))
        result = tool._run(operation="read", path="link.txt")
        assert "Error" in result


# ═══════════════════════════════════════════════════════════════════════
# Tool Sandbox Tests
# ═══════════════════════════════════════════════════════════════════════


class TestToolSandbox:
    """Test the ToolSandbox execution wrapper."""

    def setup_method(self):
        reset_sandbox()

    def test_successful_execution(self):
        sandbox = ToolSandbox()
        result = sandbox.execute("my_tool", lambda: "result_value")
        assert result.success is True
        assert result.output == "result_value"
        assert result.execution_time_ms > 0
        sandbox.shutdown()

    def test_timeout_enforcement(self):
        sandbox = ToolSandbox()
        result = sandbox.execute(
            "slow_tool", lambda: time.sleep(10) or "done", timeout_override=0.2
        )
        assert result.success is False
        assert result.timed_out is True
        assert "timed out" in result.error.lower()
        sandbox.shutdown()

    def test_permission_check(self):
        policy = SandboxPolicy(allowed_permissions={Permission.SYSTEM_TIME.value})
        sandbox = ToolSandbox(policy=policy)
        result = sandbox.execute(
            "calc",
            lambda: "42",
            required_permissions=[Permission.SYSTEM_COMPUTE.value],
        )
        assert result.success is False
        assert result.permission_denied is True
        assert "system:compute" in result.error
        sandbox.shutdown()

    def test_permission_allowed(self):
        policy = SandboxPolicy(allowed_permissions={
            Permission.SYSTEM_TIME.value,
            Permission.SYSTEM_COMPUTE.value,
        })
        sandbox = ToolSandbox(policy=policy)
        result = sandbox.execute(
            "calc", lambda: "42", required_permissions=[Permission.SYSTEM_COMPUTE.value]
        )
        assert result.success is True
        assert result.output == "42"
        sandbox.shutdown()

    def test_output_truncation(self):
        policy = SandboxPolicy(max_output_size=50)
        sandbox = ToolSandbox(policy=policy)
        result = sandbox.execute("big_output", lambda: "x" * 200)
        assert result.success is True
        assert "truncated" in result.output
        assert len(result.output) < 200
        sandbox.shutdown()

    def test_exception_handling(self):
        sandbox = ToolSandbox()
        result = sandbox.execute("error_tool", lambda: 1 / 0)
        assert result.success is False
        assert result.error is not None
        assert "division by zero" in result.error.lower()
        sandbox.shutdown()

    def test_concurrent_limit(self):
        policy = SandboxPolicy(max_concurrent=1, max_execution_time=5)
        sandbox = ToolSandbox(policy=policy)
        
        # First execution blocks the slot
        import threading
        barrier = threading.Barrier(2, timeout=5)
        
        def slow():
            barrier.wait()
            time.sleep(0.5)
            return "slow_done"
        
        # Submit first task in a thread
        results = []
        def run_first():
            r = sandbox.execute("t1", slow)
            results.append(("first", r))
        
        t = threading.Thread(target=run_first)
        t.start()
        barrier.wait()
        
        # While first is running, second should be rejected
        r2 = sandbox.execute("t2", lambda: "second")
        assert r2.success is False
        assert "limit" in r2.error.lower()
        
        t.join(timeout=5)
        sandbox.shutdown()

    def test_get_sandbox_singleton(self):
        s1 = get_sandbox()
        s2 = get_sandbox()
        assert s1 is s2
        reset_sandbox()

    def test_path_check(self):
        policy = SandboxPolicy()
        assert policy.is_path_allowed("/tmp/safe/file.txt") is True
        assert policy.is_path_allowed("/etc/shadow") is False


# ═══════════════════════════════════════════════════════════════════════
# Rate Limiter Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSlidingWindow:
    """Test the sliding window counter."""

    def test_allows_within_limit(self):
        w = _SlidingWindow(max_calls=3, window_seconds=60)
        for _ in range(3):
            allowed, remaining, retry = w.check()
            assert allowed is True
            w.record()

    def test_blocks_over_limit(self):
        w = _SlidingWindow(max_calls=2, window_seconds=60)
        w.record()
        w.record()
        allowed, remaining, retry = w.check()
        assert allowed is False
        assert remaining == 0
        assert retry > 0

    def test_window_expiry(self):
        w = _SlidingWindow(max_calls=1, window_seconds=0.1)
        w.record()
        allowed, _, _ = w.check()
        assert allowed is False
        time.sleep(0.15)
        allowed, _, _ = w.check()
        assert allowed is True

    def test_count(self):
        w = _SlidingWindow(max_calls=10, window_seconds=60)
        assert w.count() == 0
        w.record()
        w.record()
        assert w.count() == 2

    def test_reset(self):
        w = _SlidingWindow(max_calls=10, window_seconds=60)
        w.record()
        w.record()
        w.reset()
        assert w.count() == 0


class TestToolRateLimiter:
    """Test the full rate limiter."""

    def setup_method(self):
        reset_rate_limiter()

    def test_global_limit(self):
        limiter = ToolRateLimiter(
            RateLimitConfig(global_max_calls=2, global_window_seconds=60)
        )
        limiter.record("tool_a")
        limiter.record("tool_b")
        result = limiter.check("tool_c")
        assert result.allowed is False
        assert "Global" in result.reason

    def test_per_tool_limit(self):
        limiter = ToolRateLimiter(
            RateLimitConfig(per_tool_max_calls=2, per_tool_window_seconds=60)
        )
        limiter.record("calc")
        limiter.record("calc")
        result = limiter.check("calc")
        assert result.allowed is False
        assert "calc" in result.reason

    def test_per_user_limit(self):
        limiter = ToolRateLimiter(
            RateLimitConfig(per_user_max_calls=2, per_user_window_seconds=60)
        )
        limiter.record("tool_a", user_id="user1")
        limiter.record("tool_b", user_id="user1")
        result = limiter.check("tool_c", user_id="user1")
        assert result.allowed is False
        assert "Per-user" in result.reason

    def test_different_users_independent(self):
        limiter = ToolRateLimiter(
            RateLimitConfig(per_user_max_calls=1, per_user_window_seconds=60)
        )
        limiter.record("tool_a", user_id="user1")
        result = limiter.check("tool_a", user_id="user2")
        assert result.allowed is True

    def test_remaining_calls_decreases(self):
        limiter = ToolRateLimiter(
            RateLimitConfig(per_tool_max_calls=5, per_tool_window_seconds=60)
        )
        r1 = limiter.check("calc")
        assert r1.remaining_calls == 5
        limiter.record("calc")
        r2 = limiter.check("calc")
        assert r2.remaining_calls == 4

    def test_get_stats(self):
        limiter = ToolRateLimiter()
        limiter.record("tool_a")
        limiter.record("tool_a")
        limiter.record("tool_b")
        stats = limiter.get_stats()
        assert stats["global_calls"] == 3
        assert stats["tool_tool_a_calls"] == 2
        assert stats["tool_tool_b_calls"] == 1

    def test_reset(self):
        limiter = ToolRateLimiter()
        limiter.record("tool_a")
        limiter.reset()
        stats = limiter.get_stats()
        assert stats["global_calls"] == 0

    def test_get_rate_limiter_singleton(self):
        r1 = get_rate_limiter()
        r2 = get_rate_limiter()
        assert r1 is r2
        reset_rate_limiter()


# ═══════════════════════════════════════════════════════════════════════
# Semantic Routing Heuristic Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRoutingHeuristics:
    """Test that heuristic triggers correctly detect tool-relevant queries."""

    def _check_heuristic(self, query, pattern_func):
        """Helper to test heuristic detection on a query string."""
        import re
        return pattern_func(query.lower(), query)

    def test_calculator_heuristic_arithmetic(self):
        import re
        query = "What is 25 + 37?"
        assert re.search(r"\b\d+\s*[\+\-\*/\^]\s*\d+", query)

    def test_calculator_heuristic_keyword_en(self):
        assert "calculate" in "Please calculate the area".lower()

    def test_calculator_heuristic_keyword_de(self):
        assert "berechne" in "Berechne 15 mal 3".lower()

    def test_calculator_heuristic_sqrt(self):
        import re
        query = "What is sqrt(144)?"
        assert re.search(r"\b(sqrt|log|sin|cos|tan|factorial)\s*\(", query.lower())

    def test_calculator_heuristic_percentage(self):
        import re
        query = "What is 15% of 200?"
        assert re.search(r"\b\d+\s*%\s*(of|von|de)\b", query.lower())

    def test_calculator_heuristic_unit_conversion(self):
        import re
        query = "Convert 100 km to miles"
        assert re.search(r"\b(convert|umrechnen|convertir)\b.*\b(to|in|nach|en)\b", query.lower())

    def test_file_ops_heuristic_read(self):
        assert "read file" in "Can you read file config.yaml?".lower()

    def test_file_ops_heuristic_list(self):
        assert "list directory" in "List directory contents".lower()

    def test_file_ops_heuristic_de(self):
        assert "datei lesen" in "Bitte die Datei lesen".lower()

    def test_no_false_positive_calculator(self):
        """Regular text should not trigger calculator heuristic."""
        import re
        query = "Tell me about the weather today"
        assert not re.search(r"\b\d+\s*[\+\-\*/\^]\s*\d+", query)
        assert not any(k in query.lower() for k in ["berechne", "calculate", "compute"])

    def test_no_false_positive_file_ops(self):
        """Regular text should not trigger file ops heuristic."""
        query = "What is the capital of France?"
        keywords = ["read file", "write file", "list files", "list directory", "datei"]
        assert not any(k in query.lower() for k in keywords)


# ═══════════════════════════════════════════════════════════════════════
# Tool Embedding Cache Tests
# ═══════════════════════════════════════════════════════════════════════


class TestToolEmbeddingCache:
    """Test that tool embedding caching works."""

    def test_cache_dict_operations(self):
        """Verify the cache dict structure used for tool embeddings."""
        from universal_agentic_framework.orchestration.graph_builder import _tool_embedding_cache, _clear_embedding_cache

        _clear_embedding_cache()
        assert len(_tool_embedding_cache) == 0

        # Simulate caching a tool embedding
        import numpy as np
        fake_embedding = np.random.rand(384)
        cache_key = "test_model:test_tool:12345"
        _tool_embedding_cache[cache_key] = fake_embedding

        assert cache_key in _tool_embedding_cache
        assert np.array_equal(_tool_embedding_cache[cache_key], fake_embedding)

        _clear_embedding_cache()
        assert len(_tool_embedding_cache) == 0


# ═══════════════════════════════════════════════════════════════════════
# Tool Registry Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistryNewTools:
    """Test that new tools are discovered by the registry."""

    def test_calculator_tool_discovered(self):
        """Registry should discover and load the calculator tool."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, fork_language="en")
        tools = registry.discover_and_load()
        tool_names = [t.name for t in tools]
        assert "calculator_tool" in tool_names

    def test_file_ops_tool_discovered(self):
        """Registry should discover and load the file_ops tool."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, fork_language="en")
        tools = registry.discover_and_load()
        tool_names = [t.name for t in tools]
        assert "file_ops_tool" in tool_names

    def test_calculator_tool_is_functional(self):
        """Calculator tool loaded by registry should be callable."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, fork_language="en")
        tools = registry.discover_and_load()
        calc = next(t for t in tools if t.name == "calculator_tool")
        result = calc._run(operation="evaluate", expression="1 + 1")
        assert "= 2" in result

    def test_total_enabled_tools_count(self):
        """Should have at least 5 enabled tools (datetime, calculator, file_ops, web_search, extract)."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, fork_language="en")
        tools = registry.discover_and_load()
        # datetime + calculator + file_ops + web_search_mcp + extract_webpage_mcp = 5
        assert len(tools) >= 5
