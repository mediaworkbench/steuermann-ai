"""Pure system-prompt section builders extracted from node_generate_response (W3).

Each returns a ready-to-append prompt-section string (or "" when the section is absent),
with NO side effects — no state mutation, no metrics, no logging — so they are directly
unit-testable. The node stays the orchestrator that accumulates these into the system prompt
in order and keeps the interwoven source/allowed-URL accumulation and `state` mutations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

# Tools whose output must be relayed verbatim — a synthesis instruction makes the model
# second-guess an already-correct result (OCR text, decoded barcodes, structured JSON).
_VERBATIM_RELAY_TOOLS = {
    "ocr_tool", "read_barcodes_tool", "image_metadata_tool",
    "analyze_document_tool", "analyze_chart_tool",
}

# Per-tool result text is capped before injection so one huge tool dump can't crowd out the
# rest of the prompt.
_TOOL_RESULT_RENDER_CAP = 8000


def build_tool_results_block(
    tool_results: Dict[str, str],
    tool_execution_results: Dict[str, Dict[str, Any]],
    routing_metadata: Dict[str, str],
) -> str:
    """Render the TOOL RESULTS + CONTEXT PRIORITY sections (empty string if no tool ran)."""
    if not tool_results:
        return ""
    lines = []
    for tool_name, result in tool_results.items():
        reason = routing_metadata.get(tool_name, "semantic match")
        header = f"Tool: {tool_name} (via {reason})\nResult:"
        envelope = tool_execution_results.get(tool_name, {})
        # Use full output_text (not the 300-char summary) so models have enough content.
        full_result = envelope.get("output_text") or result
        rendered = str(full_result)[:_TOOL_RESULT_RENDER_CAP]
        lines.append(f"{header}\n{rendered}")
    tool_context = "\n\n".join(lines)
    return (
        f"\n\n=== TOOL RESULTS ===\n{tool_context}\n=== ENDE TOOL RESULTS ===\n"
        "\n\n=== CONTEXT PRIORITY ===\n"
        "Use current-turn TOOL RESULTS as the primary source of truth. "
        "Use PAST CONTEXT only as secondary background. "
        "If there is any conflict, follow TOOL RESULTS. "
        "Do not mention model training data, knowledge-cutoff dates, or stale prior knowledge when TOOL RESULTS provide current information.\n"
        "=== END CONTEXT PRIORITY ===\n"
    )


def select_synthesis_instruction(
    used_tool_names: Set[str],
    has_citable_sources: bool,
    prompts_cfg: Optional[Any],
    lang: str,
) -> str:
    """Choose the synthesis instruction: verbatim relay, configured prompt, or built-in default."""
    if used_tool_names & _VERBATIM_RELAY_TOOLS:
        return (
            "The tool has returned its output. Present the result directly to the user:\n"
            "- For OCR / text extraction: display the extracted text verbatim as your answer. "
            "Do NOT paraphrase, summarize, or question whether the content is correct — it is.\n"
            "- For structured JSON (documents, charts, barcodes, metadata): present the "
            "information clearly and readably.\n"
            "Never say the result is missing or incorrect."
        )
    prompt_key = "synthesis_with_sources" if has_citable_sources else "synthesis"
    synthesis_text = (
        prompts_cfg.get_prompt(lang, prompt_key, fallback_lang="en") if prompts_cfg else None
    )
    return synthesis_text or (
        "Synthesize a coherent, well-structured answer from the tool results and knowledge base above. "
        "Do NOT list raw result items. Write a fluent summary that directly answers the user's question."
    )


def build_memory_context_block(loaded_memory: Any, web_tools_used: bool) -> str:
    """Render the PAST CONTEXT section from loaded memories (low-priority when web tools ran)."""
    if not loaded_memory:
        return ""
    memory_context = "\n\n".join(
        f"[Memory]\n{mem.text if hasattr(mem, 'text') else mem.get('text', '')}"
        for mem in loaded_memory[:5]
    )
    if web_tools_used:
        return (
            "\n\n=== PAST CONTEXT (LOW PRIORITY) ===\n"
            "Use this only as background. If it conflicts with current-turn TOOL RESULTS, "
            "always trust current-turn TOOL RESULTS.\n\n"
            f"{memory_context}\n"
            "=== END PAST CONTEXT (LOW PRIORITY) ===\n"
        )
    return f"\n\n=== PAST CONTEXT ===\n{memory_context}\n=== END PAST CONTEXT ===\n"


def build_crew_findings_block(crew_results: Dict[str, Any]) -> str:
    """Render successful crew results as === <CREW> FINDINGS === sections (empty if none)."""
    if not crew_results:
        return ""
    parts = []
    for crew_name, crew_result in crew_results.items():
        if isinstance(crew_result, dict) and crew_result.get("success") and crew_result.get("result"):
            section = crew_name.upper().replace("_", " ")
            parts.append(
                f"\n\n=== {section} FINDINGS ===\n"
                f"{crew_result['result']}\n"
                f"=== END {section} FINDINGS ===\n"
            )
    return "".join(parts)
