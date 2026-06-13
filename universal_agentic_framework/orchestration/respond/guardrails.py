"""Post-generation guardrail retries + final fallback, extracted from node_generate_response.

Each helper takes the current ``response_text`` plus the already-resolved ``model`` (the
same instance the node uses, so the test suite's monkeypatched model flows straight through)
and returns the possibly-updated text. Order and short-circuiting mirror the original node:
an empty response is retried for synthesis, then non-empty responses are checked for
attachment / web-extract contradictions, then any still-empty response gets a tool-based
fallback.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.orchestration.respond.text_cleanup import strip_control_tokens

logger = get_logger(__name__)


def _message_text(out: Any) -> str:
    if hasattr(out, "content"):
        return out.content
    if isinstance(out, str):
        return out
    return str(out)


def retry_synthesis_if_empty(
    response_text: str,
    *,
    model: Any,
    lang: str,
    user_msg: str,
    knowledge_context: List[Dict[str, Any]],
    context_text: str,
    tool_results: Dict[str, str],
    collected_sources: List[Dict[str, Any]],
) -> str:
    """When the model emitted only control tokens (empty after sanitization), retry once with a
    focused data-only synthesis prompt (no tool catalog). Returns "" if the retry also fails."""
    if response_text:
        return response_text

    logger.info("LLM produced empty response after sanitization, retrying with synthesis prompt")

    has_citable_sources = bool(collected_sources)
    _synth_default = {
        "en": (
            "Synthesize a coherent, well-structured answer from the tool results and knowledge base above. "
            "Do NOT list raw result items. Write a fluent summary that directly answers the user's question. "
            "Treat current-turn tool results as current facts. Do not mention training cutoffs, outdated knowledge limits, or inability to browse."
            + (
                " Cite sources using numbered references like [1], [2] etc. matching the SOURCES list below."
                if has_citable_sources
                else ""
            )
        ),
        "de": (
            "Fasse die obigen Tool-Ergebnisse und die Wissensdatenbank zu einer zusammenhaengenden, "
            "gut strukturierten Antwort zusammen. Liste KEINE rohen Ergebnis-Eintraege auf. "
            "Schreibe eine fluessige Zusammenfassung, die die Frage des Benutzers direkt beantwortet. "
            "Behandle aktuelle Tool-Ergebnisse als aktuelle Fakten. Erwaehne keine Wissensgrenzen, Trainingsdaten-Grenzen oder fehlende Browsing-Faehigkeit."
            + (
                " Zitiere Quellen mit nummerierten Referenzen wie [1], [2] usw. passend zur SOURCES-Liste unten."
                if has_citable_sources
                else ""
            )
        ),
    }
    synth_instr = _synth_default.get(lang, _synth_default["en"])

    retry_parts = [
        "You are a helpful AI assistant. Do NOT emit tool calls or control tokens. "
        "Return ONLY plain natural-language text. "
        "Use current-turn tool results as the source of truth and do not mention training cutoffs or browsing limitations.\n"
    ]
    if knowledge_context:
        retry_parts.append(f"=== KNOWLEDGE BASE ===\n{context_text}\n=== END KNOWLEDGE BASE ===\n")
    if tool_results:
        for _tn, _tr in tool_results.items():
            retry_parts.append(f"=== TOOL: {_tn} ===\n{_tr}\n=== END TOOL ===\n")
    if collected_sources:
        src_lines = []
        for src in collected_sources:
            _idx = src.get("index", 0)
            if src.get("url"):
                src_lines.append(f"[{_idx}] {src['label']} - {src['url']}")
            else:
                src_lines.append(f"[{_idx}] {src['label']} (knowledge base)")
        retry_parts.append("=== SOURCES ===\n" + "\n".join(src_lines) + "\n=== END SOURCES ===\n")
    retry_parts.append(f"=== TASK ===\n{synth_instr}\n=== END TASK ===\n")

    retry_messages = [
        SystemMessage(content="\n".join(retry_parts)),
        HumanMessage(content=user_msg),
    ]
    try:
        retry_out = model.invoke(retry_messages)
        response_text = strip_control_tokens(_message_text(retry_out))
        logger.info("Synthesis retry succeeded", response_length=len(response_text))
    except Exception as retry_err:
        logger.error("Synthesis retry failed", error=str(retry_err))
        response_text = ""
    return response_text


_ATTACHMENT_REFUSAL_RE = re.compile(
    r"("
    r"can(?:not|'t)\s+(?:view|access|see).{0,60}(?:attach|workspace\s+document|document)|"
    r"don['’]t\s+see.{0,60}(?:attach|workspace\s+document|document)|"
    r"(?:attach\w*|workspace\s+document\w*|document\w*).{0,60}"
    r"(?:not\s+)?(?:available|accessible|provided|found|readable|unavailable|inaccessible)|"
    r"not\s+accessible.{0,60}(?:content\s+extraction|extract)"
    r")",
    flags=re.IGNORECASE,
)


def retry_on_attachment_refusal(
    response_text: str,
    *,
    model: Any,
    state: Dict[str, Any],
    system_prompt: str,
    all_msgs: List[Dict[str, Any]],
) -> str:
    """If attachments were injected but the model claims it can't access them, retry once with
    an explicit instruction. No-op when the response is empty or has no refusal phrasing."""
    attachment_context = state.get("attachment_context") or []
    if not (response_text and attachment_context):
        return response_text
    if not _ATTACHMENT_REFUSAL_RE.search(response_text):
        return response_text

    from universal_agentic_framework.monitoring.metrics import (
        ATTACHMENT_REFUSAL_RETRIES_TOTAL,
        ATTACHMENT_REFUSAL_RETRIES_SUCCESS_TOTAL,
    )

    profile_name = state.get("profile_name", "unknown")
    logger.warning(
        "Model claimed attachments were unavailable despite injected attachment context; retrying once",
        attachment_count=len(attachment_context),
    )
    ATTACHMENT_REFUSAL_RETRIES_TOTAL.labels(profile_name=profile_name).inc()

    correction_prompt = (
        system_prompt
        + "\n\n=== ATTACHMENT HANDLING ===\n"
        + "The USER ATTACHMENTS section is already provided in this prompt and is readable context. "
        + "Do not state that attachments are unavailable. Use that content directly in your answer.\n"
        + "=== END ATTACHMENT HANDLING ===\n"
    )
    correction_messages = [SystemMessage(content=correction_prompt)]
    for msg in all_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            correction_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            correction_messages.append(AIMessage(content=content))
    try:
        correction_out = model.invoke(correction_messages)
        response_text = strip_control_tokens(_message_text(correction_out))
        logger.info("Attachment correction retry succeeded", response_length=len(response_text))
        ATTACHMENT_REFUSAL_RETRIES_SUCCESS_TOTAL.labels(profile_name=profile_name).inc()
    except Exception as correction_err:
        logger.error("Attachment correction retry failed", error=str(correction_err))
    return response_text


_ACCESS_REFUSAL_RE = re.compile(
    r"(unable\s+to\s+(?:access|retrieve|visit)|"
    r"cannot\s+(?:access|retrieve|visit|reach)|"
    r"can't\s+(?:access|retrieve|visit|reach)|"
    r"connection\s+error|not\s+directly\s+retrievable|"
    r"verbindungsproblem|nicht\s+abrufbar|nicht\s+zugaenglich|"
    r"kann\s+(?:nicht|keine)\s+(?:zugreifen|abrufen|erreichen))",
    flags=re.IGNORECASE,
)


def retry_on_web_extract_contradiction(
    response_text: str,
    *,
    model: Any,
    tool_results: Dict[str, str],
    user_msg: str,
) -> str:
    """If webpage extraction succeeded but the model claims it can't access the site, retry once
    using the extracted content directly. No-op when empty / no successful extraction."""
    extract_result = str(tool_results.get("extract_webpage_mcp", "")).strip()
    if not (response_text and extract_result and not extract_result.lower().startswith("error:")):
        return response_text
    if not _ACCESS_REFUSAL_RE.search(response_text):
        return response_text

    logger.warning(
        "Model contradicted successful extract_webpage_mcp result; retrying once",
        extract_length=len(extract_result),
    )
    correction_prompt = (
        "You are given successfully extracted webpage content. "
        "Answer the user's question ONLY from this extracted content. "
        "Do not claim connection or access errors.\n\n"
        "=== EXTRACTED WEBPAGE CONTENT ===\n"
        f"{extract_result[:12000]}\n"
        "=== END EXTRACTED WEBPAGE CONTENT ==="
    )
    correction_messages = [
        SystemMessage(content=correction_prompt),
        HumanMessage(content=user_msg),
    ]
    try:
        correction_out = model.invoke(correction_messages)
        response_text = strip_control_tokens(_message_text(correction_out))
        logger.info("Web extract contradiction retry succeeded", response_length=len(response_text))
    except Exception as correction_err:
        logger.error("Web extract contradiction retry failed", error=str(correction_err))
    return response_text


def format_tool_based_fallback(
    response_text: str,
    *,
    tool_results: Dict[str, str],
    lang: str,
) -> str:
    """When the response is still empty after retries, format raw tool output (or a polite
    rephrase request) so the user never gets a blank reply."""
    if response_text:
        return response_text

    tool_based_text = None

    # Prefer readable formatting of web search results over the raw dict string.
    web_search_raw = str(tool_results.get("web_search_mcp", "")).strip()
    if web_search_raw and not web_search_raw.lower().startswith("tool execution failed"):
        try:
            parsed = ast.literal_eval(web_search_raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                items = parsed.get("results", [])[:5]
                lines = []
                for idx, item in enumerate(items, 1):
                    title = str(item.get("title", "Untitled"))
                    url = str(item.get("url", ""))
                    snippet = str(item.get("snippet", "")).replace("\n", " ").strip()
                    if len(snippet) > 180:
                        snippet = snippet[:177] + "..."
                    lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")
                intro = (
                    "Here are the most relevant web results I found:"
                    if lang != "de"
                    else "Hier sind die relevantesten Web-Ergebnisse, die ich gefunden habe:"
                )
                tool_based_text = intro + "\n\n" + "\n\n".join(lines)
        except Exception:
            tool_based_text = None

    if tool_based_text is None:
        # Prioritize utility tools over web search in fallback.
        for tool_name in ["calculator_tool", "datetime_tool", "extract_webpage_mcp", "web_search_mcp"]:
            candidate = str(tool_results.get(tool_name, "")).strip()
            if candidate and not candidate.lower().startswith("tool execution failed"):
                tool_based_text = candidate
                break

    if tool_based_text:
        prefix = (
            "Hier ist das Ergebnis aus den ausgefuehrten Tools:\n\n"
            if lang == "de"
            else "Here is the result from the executed tools:\n\n"
        )
        return f"{prefix}{tool_based_text[:2500]}"

    return (
        "Entschuldigung, ich habe intern Werkzeuge ausgefuehrt, aber keine lesbare Antwort erhalten. "
        "Bitte formuliere deine Frage kurz neu."
        if lang == "de"
        else "Sorry, I executed tools internally but did not receive a readable answer. "
        "Please rephrase your question briefly."
    )
