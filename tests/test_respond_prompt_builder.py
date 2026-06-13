"""Unit tests for the pure prompt-section builders extracted from node_generate_response (W3)."""

from universal_agentic_framework.orchestration.respond.prompt_builder import (
    build_tool_results_block,
    select_synthesis_instruction,
    build_memory_context_block,
    build_crew_findings_block,
)


def test_build_tool_results_block_renders_sections():
    block = build_tool_results_block(
        {"calculator_tool": "42"},
        {"calculator_tool": {"output_text": "42 exactly"}},
        {"calculator_tool": "semantic match"},
    )
    assert "=== TOOL RESULTS ===" in block
    assert "Tool: calculator_tool (via semantic match)" in block
    assert "42 exactly" in block  # prefers envelope output_text over the raw result
    assert "=== CONTEXT PRIORITY ===" in block


def test_build_tool_results_block_empty():
    assert build_tool_results_block({}, {}, {}) == ""


def test_select_synthesis_instruction_verbatim_for_relay_tools():
    text = select_synthesis_instruction({"ocr_tool"}, False, None, "en")
    assert "verbatim" in text.lower()


def test_select_synthesis_instruction_default_without_prompts_cfg():
    text = select_synthesis_instruction({"web_search_mcp"}, True, None, "en")
    assert "Synthesize" in text


def test_select_synthesis_instruction_uses_configured_prompt():
    class _Cfg:
        def get_prompt(self, lang, key, fallback_lang="en"):
            return f"CONFIGURED:{key}"

    assert select_synthesis_instruction({"web_search_mcp"}, True, _Cfg(), "en") == "CONFIGURED:synthesis_with_sources"
    assert select_synthesis_instruction({"web_search_mcp"}, False, _Cfg(), "en") == "CONFIGURED:synthesis"


def test_build_memory_context_block_priority_variants():
    mem = [{"text": "user likes Postgres"}]
    normal = build_memory_context_block(mem, web_tools_used=False)
    assert "=== PAST CONTEXT ===" in normal
    assert "user likes Postgres" in normal

    low = build_memory_context_block(mem, web_tools_used=True)
    assert "LOW PRIORITY" in low

    assert build_memory_context_block([], web_tools_used=False) == ""


def test_build_crew_findings_block():
    out = build_crew_findings_block(
        {
            "research": {"success": True, "result": "found X"},
            "analytics": {"success": False, "result": "ignored"},
            "planning": {"success": True, "result": ""},  # empty result skipped
        }
    )
    assert "=== RESEARCH FINDINGS ===" in out
    assert "found X" in out
    assert "ignored" not in out  # unsuccessful crew omitted
    assert "PLANNING" not in out  # empty result omitted
    assert build_crew_findings_block({}) == ""
