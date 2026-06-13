"""Unit tests for the pure text-cleanup helpers extracted from node_generate_response (W3)."""

from universal_agentic_framework.orchestration.respond.text_cleanup import (
    strip_control_tokens,
    filter_untrusted_urls,
)


def test_strip_control_tokens_removes_blocks_tags_and_placeholder():
    raw = "Hello <|tool_call_start|>{json}<|tool_call_end|> there <|channel|>\n[Tool Result]\n"
    assert strip_control_tokens(raw) == "Hello  there"


def test_strip_control_tokens_noop_on_clean_text():
    assert strip_control_tokens("just a normal answer") == "just a normal answer"
    assert strip_control_tokens("") == ""
    assert strip_control_tokens(None) is None


def test_filter_untrusted_urls_replaces_unknown_only():
    text = "See https://allowed.com/p and https://evil.com/x"
    out, removed = filter_untrusted_urls(text, {"https://allowed.com/p"})
    assert "https://allowed.com/p" in out
    assert "https://evil.com/x" not in out
    assert "source omitted" in out
    assert removed == 1


def test_filter_untrusted_urls_normalizes_trailing_punctuation():
    # The model emits a trailing period; the allowed form has none — still a match.
    text = "Read https://allowed.com/page."
    out, removed = filter_untrusted_urls(text, {"https://allowed.com/page"})
    assert removed == 0
    assert "source omitted" not in out


def test_filter_untrusted_urls_no_urls():
    out, removed = filter_untrusted_urls("no links here", set())
    assert out == "no links here"
    assert removed == 0
