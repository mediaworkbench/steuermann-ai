"""Calculator tool for mathematical operations."""

import ast
import math
import operator
import statistics
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


# Safe operator mapping for AST evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions available in expressions
_SAFE_MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Unit conversion tables
_UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    # Length (base: meters)
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "inch": 0.0254,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "yd": 0.9144,
        "yard": 0.9144,
        "mi": 1609.344,
        "mile": 1609.344,
    },
    # Weight (base: grams)
    "weight": {
        "mg": 0.001,
        "g": 1.0,
        "kg": 1000.0,
        "oz": 28.3495,
        "lb": 453.592,
        "pound": 453.592,
        "t": 1_000_000.0,
        "ton": 1_000_000.0,
    },
    # Temperature (special handling)
    "temperature": {
        "c": "celsius",
        "celsius": "celsius",
        "f": "fahrenheit",
        "fahrenheit": "fahrenheit",
        "k": "kelvin",
        "kelvin": "kelvin",
    },
    # Volume (base: liters)
    "volume": {
        "ml": 0.001,
        "l": 1.0,
        "liter": 1.0,
        "gal": 3.78541,
        "gallon": 3.78541,
        "qt": 0.946353,
        "quart": 0.946353,
        "pt": 0.473176,
        "pint": 0.473176,
        "cup": 0.236588,
        "tbsp": 0.0147868,
        "tsp": 0.00492892,
    },
    # Time (base: seconds)
    "time": {
        "ms": 0.001,
        "s": 1.0,
        "sec": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hour": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
        "month": 2_592_000.0,
        "year": 31_536_000.0,
    },
    # Data (base: bytes)
    "data": {
        "b": 1,
        "byte": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    },
}


class CalculatorInput(BaseModel):
    """Input for calculator operations."""

    operation: Literal["evaluate", "convert", "statistics", "percentage"] = Field(
        default="evaluate",
        description="Operation to perform: 'evaluate' (math expression), 'convert' (unit conversion), 'statistics' (stats on a list), 'percentage' (percentage calculations)",
    )
    expression: Optional[str] = Field(
        default=None,
        description="Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', '2**10')",
    )
    value: Optional[float] = Field(
        default=None,
        description="Numeric value for conversion or percentage operations",
    )
    from_unit: Optional[str] = Field(
        default=None, description="Source unit for conversion (e.g., 'km', 'lb', 'celsius')"
    )
    to_unit: Optional[str] = Field(
        default=None, description="Target unit for conversion (e.g., 'mi', 'kg', 'fahrenheit')"
    )
    values: Optional[List[float]] = Field(
        default=None, description="List of numbers for statistical operations"
    )
    percentage: Optional[float] = Field(
        default=None, description="Percentage value for percentage calculations"
    )


class CalculatorTool(BaseTool):
    """Perform mathematical calculations, unit conversions, and statistics."""

    name: str = "calculator_tool"
    description: str = (
        "Perform mathematical calculations, unit conversions, and statistical operations. "
        "Use for arithmetic, algebra, percentages, unit conversions, or statistical analysis."
    )
    args_schema: type[BaseModel] = CalculatorInput

    # Config: injected by registry from tool.yaml / tools.yaml
    max_expression_length: int = 500
    precision: int = 10

    def _run(
        self,
        operation: str = "evaluate",
        expression: Optional[str] = None,
        value: Optional[float] = None,
        from_unit: Optional[str] = None,
        to_unit: Optional[str] = None,
        values: Optional[List[float]] = None,
        percentage: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Execute calculator operation."""
        try:
            if operation == "evaluate":
                return self._evaluate(expression)
            elif operation == "convert":
                return self._convert(value, from_unit, to_unit)
            elif operation == "statistics":
                return self._statistics(values)
            elif operation == "percentage":
                return self._percentage(value, percentage)
            else:
                return f"Unknown operation '{operation}'. Available: evaluate, convert, statistics, percentage"
        except Exception as e:
            logger.error("Calculator operation failed", error=str(e), operation=operation)
            return f"Error: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Async execution (uses sync implementation)."""
        return self._run(**kwargs)

    # ── Expression evaluation (sandboxed AST) ─────────────────────────

    def _evaluate(self, expression: Optional[str]) -> str:
        """Safely evaluate a mathematical expression using AST parsing."""
        if not expression:
            return "Error: No expression provided. Example: '2 + 3 * 4' or 'sqrt(16)'"

        if len(expression) > self.max_expression_length:
            return f"Error: Expression too long ({len(expression)} chars, max {self.max_expression_length})"

        logger.info("Evaluating expression", expression=expression)

        try:
            result = self._safe_eval(expression)
            # Format result
            if isinstance(result, float):
                # Remove trailing zeros but keep precision
                formatted = f"{result:.{self.precision}f}".rstrip("0").rstrip(".")
            elif isinstance(result, complex):
                formatted = f"{result.real:.{self.precision}f} + {result.imag:.{self.precision}f}i"
                formatted = formatted.replace(".0 ", " ").rstrip("0").rstrip(".")
            else:
                formatted = str(result)

            return f"Result: {expression} = {formatted}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (ValueError, TypeError, OverflowError) as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.warning("Expression evaluation failed", expression=expression, error=str(e))
            return f"Error: Could not evaluate '{expression}': {str(e)}"

    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Parse and evaluate expression using AST — no exec/eval."""
        # Normalize common math notation
        expr = expression.strip()
        expr = expr.replace("^", "**")  # Caret → power
        expr = expr.replace("×", "*").replace("÷", "/")  # Unicode operators

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate AST nodes."""
        # Numeric literal
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # Unary operator (e.g., -5)
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(self._eval_node(node.operand))

        # Binary operator (e.g., 2 + 3)
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            # Safety: limit power to prevent huge computations
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
                raise ValueError(f"Exponent too large: {right} (max 1000)")
            return op(left, right)

        # Function call (e.g., sqrt(16))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls are allowed")
            func_name = node.func.id
            if func_name not in _SAFE_MATH_FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")
            func = _SAFE_MATH_FUNCTIONS[func_name]
            # Constants like pi, e are not callable
            if not callable(func):
                raise ValueError(f"'{func_name}' is a constant, not a function. Use it directly: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)

        # Name (constants: pi, e, tau, inf)
        if isinstance(node, ast.Name):
            if node.id in _SAFE_MATH_FUNCTIONS:
                val = _SAFE_MATH_FUNCTIONS[node.id]
                if not callable(val):
                    return val
                raise ValueError(f"'{node.id}' is a function — use it as {node.id}()")
            raise ValueError(f"Unknown variable: {node.id}")

        # List literal (for functions like max, min, sum)
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        # Tuple (treated as list)
        if isinstance(node, ast.Tuple):
            return [self._eval_node(elt) for elt in node.elts]

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    # ── Unit conversion ───────────────────────────────────────────────

    def _convert(
        self, value: Optional[float], from_unit: Optional[str], to_unit: Optional[str]
    ) -> str:
        """Convert between units."""
        if value is None or from_unit is None or to_unit is None:
            return "Error: Conversion requires 'value', 'from_unit', and 'to_unit'. Example: value=100, from_unit='km', to_unit='mi'"

        from_unit_lower = from_unit.lower().strip()
        to_unit_lower = to_unit.lower().strip()

        logger.info("Unit conversion", value=value, from_unit=from_unit_lower, to_unit=to_unit_lower)

        # Find the conversion category
        for category, units in _UNIT_CONVERSIONS.items():
            if from_unit_lower in units and to_unit_lower in units:
                if category == "temperature":
                    return self._convert_temperature(value, from_unit_lower, to_unit_lower)
                # Standard ratio-based conversion
                from_factor = units[from_unit_lower]
                to_factor = units[to_unit_lower]
                result = value * from_factor / to_factor
                formatted = f"{result:.{self.precision}f}".rstrip("0").rstrip(".")
                return f"Result: {value} {from_unit} = {formatted} {to_unit}"

        return f"Error: Cannot convert from '{from_unit}' to '{to_unit}'. Units must be in the same category."

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> str:
        """Handle temperature conversions (non-linear)."""
        temps = _UNIT_CONVERSIONS["temperature"]
        from_type = temps[from_unit]
        to_type = temps[to_unit]

        if from_type == to_type:
            return f"Result: {value} {from_unit} = {value} {to_unit} (same unit)"

        # Convert to Celsius first
        if from_type == "celsius":
            celsius = value
        elif from_type == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        elif from_type == "kelvin":
            celsius = value - 273.15
        else:
            return f"Error: Unknown temperature unit: {from_unit}"

        # Convert from Celsius to target
        if to_type == "celsius":
            result = celsius
        elif to_type == "fahrenheit":
            result = celsius * 9 / 5 + 32
        elif to_type == "kelvin":
            result = celsius + 273.15
        else:
            return f"Error: Unknown temperature unit: {to_unit}"

        formatted = f"{result:.{self.precision}f}".rstrip("0").rstrip(".")
        return f"Result: {value} {from_unit} = {formatted} {to_unit}"

    # ── Statistics ────────────────────────────────────────────────────

    def _statistics(self, values: Optional[List[float]]) -> str:
        """Compute descriptive statistics for a list of numbers."""
        if not values or len(values) == 0:
            return "Error: Provide a list of numbers. Example: values=[1, 2, 3, 4, 5]"

        logger.info("Computing statistics", count=len(values))

        n = len(values)
        mean = statistics.mean(values)
        total = sum(values)
        minimum = min(values)
        maximum = max(values)

        lines = [
            f"Statistics for {n} values:",
            f"• Sum: {total}",
            f"• Mean: {mean:.{self.precision}f}".rstrip("0").rstrip("."),
            f"• Min: {minimum}",
            f"• Max: {maximum}",
            f"• Range: {maximum - minimum}",
        ]

        if n >= 2:
            median = statistics.median(values)
            stdev = statistics.stdev(values)
            variance = statistics.variance(values)
            lines.extend(
                [
                    f"• Median: {median}",
                    f"• Std Dev: {stdev:.{self.precision}f}".rstrip("0").rstrip("."),
                    f"• Variance: {variance:.{self.precision}f}".rstrip("0").rstrip("."),
                ]
            )

        if n >= 3:
            try:
                mode = statistics.mode(values)
                lines.append(f"• Mode: {mode}")
            except statistics.StatisticsError:
                lines.append("• Mode: No unique mode")

        return "\n".join(lines)

    # ── Percentage calculations ───────────────────────────────────────

    def _percentage(self, value: Optional[float], percentage: Optional[float]) -> str:
        """Perform percentage calculations."""
        if value is None or percentage is None:
            return "Error: Provide both 'value' and 'percentage'. Example: value=200, percentage=15"

        logger.info("Percentage calculation", value=value, percentage=percentage)

        pct_of = value * percentage / 100
        increase = value * (1 + percentage / 100)
        decrease = value * (1 - percentage / 100)

        def _fmt(v: float) -> str:
            return f"{v:.{self.precision}f}".rstrip("0").rstrip(".")

        return (
            f"Percentage calculations for {percentage}% of {value}:\n"
            f"• {percentage}% of {value} = {_fmt(pct_of)}\n"
            f"• {value} + {percentage}% = {_fmt(increase)}\n"
            f"• {value} - {percentage}% = {_fmt(decrease)}"
        )
