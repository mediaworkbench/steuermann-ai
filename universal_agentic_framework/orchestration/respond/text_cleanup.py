"""Pure text-cleanup helpers for model output sanitization.

Extracted from node_generate_response, where control-token stripping was duplicated across
the main path and three retry paths, and URL filtering was an inline closure.
"""

from __future__ import annotations

import re
from typing import Set, Tuple

_CONTROL_BLOCK_RE = re.compile(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", re.DOTALL)
_CONTROL_TAG_RE = re.compile(r"<\|[^|>]+\|>")
_TOOL_RESULT_PLACEHOLDER_RE = re.compile(r"^\s*\[Tool Result\]\s*$", re.MULTILINE)
_URL_RE = re.compile(r"https?://[^\s)]+")
_URL_TRAILING_PUNCT = ".,;:!?"


def strip_control_tokens(text: str) -> str:
    """Remove leaked tool-call/control tokens and stray placeholders, then strip whitespace.

    Handles ``<|tool_call_start|>…<|tool_call_end|>`` blocks, bare ``<|…|>`` control tags, and
    the ``[Tool Result]`` placeholder some models emit when confused by tool headers. Safe to
    apply to any model output (a no-op when none are present).
    """
    if not text:
        return text
    text = _CONTROL_BLOCK_RE.sub("", text)
    text = _CONTROL_TAG_RE.sub("", text)
    text = _TOOL_RESULT_PLACEHOLDER_RE.sub("", text)
    return text.strip()


def filter_untrusted_urls(text: str, allowed_urls: Set[str]) -> Tuple[str, int]:
    """Replace URLs not present in ``allowed_urls`` with "source omitted".

    Comparison is on trailing-punctuation-normalized forms, so "…/page." matches an allowed
    "…/page". Returns ``(filtered_text, removed_count)``.
    """
    urls = _URL_RE.findall(text)
    if not urls:
        return text, 0
    allowed_norm = {a.rstrip(_URL_TRAILING_PUNCT) for a in allowed_urls}
    removed = 0
    for url in urls:
        if url.rstrip(_URL_TRAILING_PUNCT) not in allowed_norm:
            text = text.replace(url, "source omitted")
            removed += 1
    return text, removed
