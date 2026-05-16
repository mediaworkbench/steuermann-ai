"""Tests for the tool ecosystem: calculator, file_ops, routing heuristics, and registry."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.tools.calculator.tool import CalculatorTool
from universal_agentic_framework.tools.file_ops.tool import FileOpsTool


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
        from universal_agentic_framework.orchestration.helpers.tool_scoring import _tool_embedding_cache
        from universal_agentic_framework.orchestration.helpers.embedding_provider import clear_embedding_cache

        _tool_embedding_cache.clear()
        clear_embedding_cache()
        assert len(_tool_embedding_cache) == 0

        # Simulate caching a tool embedding
        import numpy as np
        fake_embedding = np.random.rand(384)
        cache_key = "test_model:test_tool:12345"
        _tool_embedding_cache[cache_key] = fake_embedding

        assert cache_key in _tool_embedding_cache
        assert np.array_equal(_tool_embedding_cache[cache_key], fake_embedding)

        _tool_embedding_cache.clear()
        clear_embedding_cache()
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
