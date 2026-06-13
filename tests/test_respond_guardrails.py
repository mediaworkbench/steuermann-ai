"""Unit tests for the post-generation guardrail helpers extracted from the respond node (W3.3)."""

from types import SimpleNamespace

from universal_agentic_framework.orchestration.respond.guardrails import (
    retry_synthesis_if_empty,
    retry_on_attachment_refusal,
    retry_on_web_extract_contradiction,
    format_tool_based_fallback,
)


class _Model:
    def __init__(self, output="recovered answer"):
        self.output = output
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.output)


# ── synthesis retry ───────────────────────────────────────────────────

def test_synthesis_retry_skipped_when_response_present():
    model = _Model()
    out = retry_synthesis_if_empty(
        "already have an answer", model=model, lang="en", user_msg="q",
        knowledge_context=[], context_text="", tool_results={}, collected_sources=[],
    )
    assert out == "already have an answer"
    assert model.calls == 0


def test_synthesis_retry_runs_when_empty():
    model = _Model("synthesized from tools")
    out = retry_synthesis_if_empty(
        "", model=model, lang="en", user_msg="what is X?",
        knowledge_context=[], context_text="", tool_results={"web_search_mcp": "{...}"},
        collected_sources=[{"label": "src", "url": "https://e.com", "index": 1}],
    )
    assert out == "synthesized from tools"
    assert model.calls == 1


def test_synthesis_retry_returns_empty_on_model_failure():
    class _Boom:
        def invoke(self, messages):
            raise RuntimeError("down")

    out = retry_synthesis_if_empty(
        "", model=_Boom(), lang="en", user_msg="q",
        knowledge_context=[], context_text="", tool_results={}, collected_sources=[],
    )
    assert out == ""


# ── attachment-refusal retry ──────────────────────────────────────────

def test_attachment_refusal_retry_fires_on_refusal():
    model = _Model("Based on the attachment, the answer is 42.")
    state = {"attachment_context": [{"text": "doc"}], "profile_name": "p"}
    out = retry_on_attachment_refusal(
        "I cannot access the attached document.", model=model, state=state,
        system_prompt="SYS", all_msgs=[{"role": "user", "content": "q"}],
    )
    assert out == "Based on the attachment, the answer is 42."
    assert model.calls == 1


def test_attachment_refusal_retry_skipped_without_refusal():
    model = _Model()
    state = {"attachment_context": [{"text": "doc"}], "profile_name": "p"}
    out = retry_on_attachment_refusal(
        "Here is the summary of your file.", model=model, state=state,
        system_prompt="SYS", all_msgs=[],
    )
    assert out == "Here is the summary of your file."
    assert model.calls == 0


def test_attachment_refusal_retry_skipped_without_attachments():
    model = _Model()
    out = retry_on_attachment_refusal(
        "I cannot access the attached document.", model=model,
        state={"attachment_context": []}, system_prompt="SYS", all_msgs=[],
    )
    assert model.calls == 0


# ── web-extract contradiction retry ───────────────────────────────────

def test_web_extract_retry_fires_on_contradiction():
    model = _Model("The page says hello.")
    out = retry_on_web_extract_contradiction(
        "I am unable to access the website.", model=model,
        tool_results={"extract_webpage_mcp": "Extracted page body"}, user_msg="q",
    )
    assert out == "The page says hello."
    assert model.calls == 1


def test_web_extract_retry_skipped_when_extract_failed():
    model = _Model()
    out = retry_on_web_extract_contradiction(
        "I am unable to access the website.", model=model,
        tool_results={"extract_webpage_mcp": "Error: timeout"}, user_msg="q",
    )
    assert model.calls == 0


# ── tool-based final fallback ─────────────────────────────────────────

def test_fallback_noop_when_response_present():
    assert format_tool_based_fallback("answer", tool_results={}, lang="en") == "answer"


def test_fallback_formats_web_search_results():
    raw = "{'results': [{'title': 'T1', 'url': 'https://a.com', 'snippet': 'snip'}]}"
    out = format_tool_based_fallback("", tool_results={"web_search_mcp": raw}, lang="en")
    assert "T1" in out and "https://a.com" in out
    # Parseable web results are a complete answer — returned with their own intro and NOT
    # double-wrapped in the generic "Here is the result from the executed tools:" prefix.
    assert out.startswith("Here are the most relevant web results I found:")
    assert "Here is the result from the executed tools:" not in out


def test_fallback_wraps_unparseable_web_search_with_prefix():
    # An unparseable web-search payload is raw output, so it DOES get the generic prefix.
    out = format_tool_based_fallback("", tool_results={"web_search_mcp": "not-a-dict"}, lang="en")
    assert out.startswith("Here is the result from the executed tools:")
    assert "not-a-dict" in out


def test_fallback_uses_utility_tool_result():
    out = format_tool_based_fallback("", tool_results={"calculator_tool": "42"}, lang="en")
    assert "42" in out
    assert out.startswith("Here is the result from the executed tools")


def test_fallback_polite_message_when_nothing_usable():
    out = format_tool_based_fallback("", tool_results={}, lang="en")
    assert "rephrase" in out.lower()
    out_de = format_tool_based_fallback("", tool_results={}, lang="de")
    assert "neu" in out_de.lower()
