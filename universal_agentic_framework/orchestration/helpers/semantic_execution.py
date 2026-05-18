"""Semantic tool execution helpers."""

import re


def extract_calculator_expression(user_msg: str) -> str:
    """Extract a likely calculator expression from user input."""
    expr_match = re.search(
        r"(\d[\d\s\+\-\*/\^\(\)\.]*\d\)*|\b(?:sqrt|log|sin|cos|tan|factorial)\s*\([^)]+\))",
        user_msg,
    )
    return expr_match.group(0).strip() if expr_match else user_msg
