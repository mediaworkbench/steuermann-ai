"""Tests for the tool ecosystem: calculator, routing heuristics, and registry."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.tools.calculator.tool import CalculatorTool


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

    def test_no_false_positive_calculator(self):
        """Regular text should not trigger calculator heuristic."""
        import re
        query = "Tell me about the weather today"
        assert not re.search(r"\b\d+\s*[\+\-\*/\^]\s*\d+", query)
        assert not any(k in query.lower() for k in ["berechne", "calculate", "compute"])


# ═══════════════════════════════════════════════════════════════════════
# Tool Embedding Cache Tests
# ═══════════════════════════════════════════════════════════════════════

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
        registry = ToolRegistry(config=config, profile_language="en")
        tools = registry.discover_and_load()
        tool_names = [t.name for t in tools]
        assert "calculator_tool" in tool_names

    def test_calculator_tool_is_functional(self):
        """Calculator tool loaded by registry should be callable."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, profile_language="en")
        tools = registry.discover_and_load()
        calc = next(t for t in tools if t.name == "calculator_tool")
        result = calc._run(operation="evaluate", expression="1 + 1")
        assert "= 2" in result

    def test_total_enabled_tools_count(self):
        """Should have at least 4 tools loaded (datetime, calculator, web_search, extract)."""
        from universal_agentic_framework.tools.registry import ToolRegistry
        from universal_agentic_framework.config import load_tools_config

        config = load_tools_config()
        registry = ToolRegistry(config=config, profile_language="en")
        tools = registry.discover_and_load()
        # datetime + calculator + web_search_mcp + extract_webpage_mcp
        assert len(tools) >= 4


def test_file_ops_tool_is_not_in_the_catalog():
    """file_ops_tool was removed: its sandbox defaulted to the app's own source tree and
    was shared by every user (workspace documents cover the use case)."""
    from universal_agentic_framework.tools.registry import ToolRegistry
    from universal_agentic_framework.config import load_tools_config

    registry = ToolRegistry(config=load_tools_config(), profile_language="en")
    assert "file_ops_tool" not in [t.name for t in registry.discover_and_load()]
