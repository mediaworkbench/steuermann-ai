"""Unit tests for the CrewSpec registry + node/route factories (W4.1 + W4.2)."""

from unittest.mock import patch

from universal_agentic_framework.orchestration import crew_nodes
from universal_agentic_framework.orchestration.crew_nodes import (
    CREW_SPECS,
    CrewSpec,
    make_crew_node,
    make_route,
    _format_crew_message,
)


def _msg(text):
    return {"messages": [{"role": "user", "content": text}]}


def test_registry_has_all_four_crews_with_expected_prefixes():
    assert set(CREW_SPECS) == {"research", "analytics", "code_generation", "planning"}
    assert CREW_SPECS["research"].message_prefix == "Research Result:"
    assert CREW_SPECS["analytics"].message_prefix == "Analysis Result:"  # not "Analytics Result:"
    assert CREW_SPECS["code_generation"].message_prefix == "Code Generation Result:"
    assert CREW_SPECS["planning"].message_prefix == "Planning Result:"


def test_simple_crews_have_no_step_sections():
    assert CREW_SPECS["research"].step_sections is None
    assert CREW_SPECS["analytics"].step_sections is None
    assert CREW_SPECS["code_generation"].step_sections is not None
    assert CREW_SPECS["planning"].step_sections is not None


def test_format_simple_result():
    out = _format_crew_message(CREW_SPECS["research"], {"success": True, "result": "found it"})
    assert out == "Research Result:\nfound it"


def test_format_simple_result_missing_falls_back():
    out = _format_crew_message(CREW_SPECS["analytics"], {"success": True})
    assert out == "Analysis Result:\nNo result"


def test_format_sectioned_result():
    result = {
        "success": True,
        "steps": {
            "design": "D", "implementation": "I", "qa_testing": "Q",
        },
    }
    out = _format_crew_message(CREW_SPECS["code_generation"], result)
    assert out.startswith("Code Generation Result:\n\n")
    assert "## Technical Design\nD" in out
    assert "## Implementation\nI" in out
    assert "## QA Report & Tests\nQ" in out


def test_format_sectioned_partial_steps():
    out = _format_crew_message(CREW_SPECS["planning"], {"steps": {"analysis": "A"}})
    assert "## Requirements Analysis\nA" in out
    assert "## Execution Plan" not in out  # missing step omitted


def test_format_sectioned_no_steps_falls_back_to_result():
    out = _format_crew_message(CREW_SPECS["planning"], {"result": "flat"})
    assert out == "Planning Result:\n\nflat"


def test_make_crew_node_names_function():
    node = make_crew_node(CREW_SPECS["research"])
    assert node.__name__ == "node_research_crew"
    assert callable(node)


def test_make_crew_node_passes_through_empty_messages():
    node = make_crew_node(CREW_SPECS["research"])
    state = {"messages": []}
    assert node(state) is state  # no crew run, returned as-is


# ── make_route (W4.2) ─────────────────────────────────────────────────

def test_make_route_names_function():
    route = make_route(CREW_SPECS["analytics"])
    assert route.__name__ == "route_to_analytics_crew"


def test_route_matches_keyword_when_enabled():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        assert crew_nodes.route_to_research_crew(_msg("please research the latest news"))
        assert crew_nodes.route_to_analytics_crew(_msg("analyze the sales data"))
        assert crew_nodes.route_to_code_generation_crew(_msg("write a function to sort"))
        assert crew_nodes.route_to_planning_crew(_msg("create a roadmap for the project"))


def test_route_matches_pattern_when_no_keyword():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        # "what" hits the research question pattern (no plain keyword in this phrasing).
        assert crew_nodes.route_to_research_crew(_msg("what color is the sky"))


def test_route_rejects_greeting():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        assert not crew_nodes.route_to_code_generation_crew(_msg("hello there"))
        assert not crew_nodes.route_to_planning_crew(_msg("hello there"))


def test_route_disabled_returns_false():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=False):
        assert not crew_nodes.route_to_research_crew(_msg("research the latest news"))


def test_route_empty_messages_returns_false():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        assert not crew_nodes.route_to_research_crew({"messages": []})


# ── W1.7: word-boundary precision (no substring false positives) ──────

def test_route_no_substring_false_positives():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        # "wo" inside "password" no longer triggers research; "plan" inside "explanation"
        # no longer triggers planning; "fix" inside "suffix" no longer triggers code.
        assert not crew_nodes.route_to_research_crew(_msg("I forgot my password"))
        assert not crew_nodes.route_to_planning_crew(_msg("give me an explanation"))
        assert not crew_nodes.route_to_code_generation_crew(_msg("explain the suffix notation"))


def test_route_whole_word_keywords_still_match():
    with patch.object(crew_nodes, "_multi_agent_crews_enabled", return_value=True):
        assert crew_nodes.route_to_code_generation_crew(_msg("fix this bug in the code"))
        assert crew_nodes.route_to_research_crew(_msg("wo ist der bahnhof"))  # \bwo\b as a word
