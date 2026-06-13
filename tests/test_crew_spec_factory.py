"""Unit tests for the CrewSpec registry + node factory (W4.1)."""

from universal_agentic_framework.orchestration.crew_nodes import (
    CREW_SPECS,
    CrewSpec,
    make_crew_node,
    _format_crew_message,
)


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
