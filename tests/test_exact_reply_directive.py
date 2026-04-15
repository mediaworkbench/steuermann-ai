"""Regression tests for _extract_exact_reply_directive in graph_builder."""

import pytest
from universal_agentic_framework.orchestration.graph_builder import (
    _extract_exact_reply_directive,
)


# ── Basic English patterns ────────────────────────────────────────────

def test_reply_with_exactly_unquoted():
    assert _extract_exact_reply_directive("Reply with exactly: FOO") == "FOO"


def test_respond_with_exactly_unquoted():
    assert _extract_exact_reply_directive("Respond with exactly: BAR") == "BAR"


def test_reply_with_exactly_double_quoted():
    assert _extract_exact_reply_directive('Reply with exactly: "HELLO WORLD"') == "HELLO WORLD"


def test_reply_with_exactly_single_quoted():
    assert _extract_exact_reply_directive("Reply with exactly: 'SINGLE'") == "SINGLE"


def test_burst_style_token():
    assert _extract_exact_reply_directive("Reply with exactly: BURST20_7") == "BURST20_7"


# ── German pattern ────────────────────────────────────────────────────

def test_german_unquoted():
    assert _extract_exact_reply_directive("Antworte genau mit: HALLO") == "HALLO"


def test_german_double_quoted():
    assert _extract_exact_reply_directive('Antworte genau mit: "GENAU SO"') == "GENAU SO"


# ── Case insensitivity ────────────────────────────────────────────────

def test_case_insensitive_reply():
    assert _extract_exact_reply_directive("REPLY WITH EXACTLY: TEST") == "TEST"


def test_case_insensitive_respond():
    assert _extract_exact_reply_directive("RESPOND WITH EXACTLY: VALUE") == "VALUE"


# ── Whitespace trimming ───────────────────────────────────────────────

def test_trailing_whitespace_trimmed():
    assert _extract_exact_reply_directive("Reply with exactly: TOKEN   ") == "TOKEN"


# ── No directive → None ───────────────────────────────────────────────

def test_no_directive_plain_message():
    assert _extract_exact_reply_directive("Hello, how are you?") is None


def test_no_directive_partial_match():
    assert _extract_exact_reply_directive("Reply with something else") is None


def test_empty_string():
    assert _extract_exact_reply_directive("") is None


def test_none_input():
    assert _extract_exact_reply_directive(None) is None


# ── Multi-line: directive must be at end ──────────────────────────────

def test_directive_at_end_of_multiline():
    msg = "Please process this.\nReply with exactly: DONE"
    assert _extract_exact_reply_directive(msg) == "DONE"


def test_directive_in_middle_not_at_end_returns_none():
    # A second line after the directive means $ won't match
    msg = "Reply with exactly: MIDDLE\nBut also do this."
    assert _extract_exact_reply_directive(msg) is None
