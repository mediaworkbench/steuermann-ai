"""Token budgeting helpers and exceptions."""
from __future__ import annotations

import math
import re
from typing import Any, Dict


class TokenBudgetExceeded(Exception):
    """Raised when a node exceeds its configured token budget."""


def estimate_tokens(text: str) -> int:
    """Very simple token estimator.

    Approximate tokens as characters/4 rounded up. This is a pragmatic
    placeholder; models differ by tokenizer. Replace with model-specific
    counting if/when needed.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / 4.0))


_DIGIT_RE = re.compile(r"\d")
_LETTER_RE = re.compile(r"[^\W\d_]")
_WHITESPACE_RE = re.compile(r"\s")
_SPACE_RUN_RE = re.compile(r"[^\S\n]+")


def estimate_content_tokens(text: str) -> int:
    """Content-aware token estimate for budgeting injected documents.

    chars/4 holds for prose but badly undercounts data: SentencePiece-style tokenizers
    (Gemma, Llama) emit one token per digit and roughly one per punctuation mark, so a
    numeric JSON export runs at ~1.75 chars/token and a CSV at ~1.2 — chars/4 estimated
    a 13.5k-token JSON file as 5.9k. Weights (calibrated against Gemma prompt counts):
    digit 1, punctuation 0.9, letter 1/4.5, newline 1, run of other whitespace 0.2. This
    lands at 0.96–1.33x the real count across JSON, CSV, YAML, code and EN/DE prose, and
    over-counts on tokenizers that group digits — the safe direction for a budget.
    """
    if not text:
        return 0
    digits = len(_DIGIT_RE.findall(text))
    letters = len(_LETTER_RE.findall(text))
    whitespace = len(_WHITESPACE_RE.findall(text))
    punctuation = len(text) - digits - letters - whitespace
    newlines = text.count("\n")
    space_runs = len(_SPACE_RUN_RE.findall(text))
    return int(math.ceil(digits + punctuation * 0.9 + letters / 4.5 + newlines + space_runs * 0.2))


def count_tokens_for_model(model_name: str, text: str) -> int:
    """Model-aware token count using litellm.token_counter with char/4 fallback.

    Uses tiktoken for OpenAI-format models (including LM Studio). Falls back
    to estimate_tokens() for unknown models or when litellm is unavailable.
    """
    if not text:
        return 0
    try:
        from litellm import token_counter
        return token_counter(model=model_name, messages=[{"role": "user", "content": text}])
    except Exception:
        return estimate_tokens(text)


def get_conversation_budget(tokens_config: Any) -> int:
    """Return the global conversation budget.

    Backward-compatible behavior: if no explicit conversation budget exists,
    fall back to ``default_budget``.
    """
    explicit = getattr(tokens_config, "conversation_budget", None)
    if explicit is not None:
        return int(explicit)
    return int(getattr(tokens_config, "default_budget", 10000))


def get_per_turn_budget(tokens_config: Any, conversation_budget: int) -> int:
    """Return per-turn budget derived from ratio or explicit override."""
    explicit = getattr(tokens_config, "per_turn_budget", None)
    if explicit is not None:
        return max(1, min(int(explicit), int(conversation_budget)))

    ratio = float(getattr(tokens_config, "per_turn_budget_ratio", 1.0))
    computed = int(conversation_budget * ratio)
    return max(1, min(computed, int(conversation_budget)))


def get_node_budget(tokens_config: Any, node_name: str, per_turn_budget: int) -> int:
    """Return node-specific budget with per-turn fallback."""
    per_node = getattr(tokens_config, "per_node_budgets", {}) or {}
    return int(per_node.get(node_name, per_turn_budget))


def per_node_hard_limit_enabled(tokens_config: Any) -> bool:
    """Whether per-node budget exceedance should raise hard errors."""
    return bool(getattr(tokens_config, "enforce_per_node_hard_limit", True))


def get_response_reserve_tokens(tokens_config: Any, per_turn_budget: int) -> int:
    """Reserve response budget fraction for downstream nodes in the same turn."""
    ratio = float(getattr(tokens_config, "response_reserve_ratio", 0.0))
    return max(0, int(per_turn_budget * ratio))


def get_budget_context(state: Dict[str, Any], tokens_config: Any) -> Dict[str, int]:
    """Compute remaining global and per-turn budget for current state."""
    total_used = int(state.get("tokens_used") or 0)
    turn_used = int(state.get("turn_tokens_used") or 0)

    conversation_budget = get_conversation_budget(tokens_config)
    global_remaining = max(0, conversation_budget - total_used)
    per_turn_budget = get_per_turn_budget(tokens_config, conversation_budget)
    effective_turn_budget = min(per_turn_budget, global_remaining)
    turn_remaining = max(0, effective_turn_budget - turn_used)

    return {
        "conversation_budget": conversation_budget,
        "global_remaining": global_remaining,
        "per_turn_budget": effective_turn_budget,
        "turn_used": turn_used,
        "turn_remaining": turn_remaining,
    }


def require_tokens(required_tokens: int, remaining_tokens: int, label: str) -> None:
    """Raise when ``required_tokens`` exceeds ``remaining_tokens``."""
    if required_tokens > remaining_tokens:
        raise TokenBudgetExceeded(
            f"{label} exceeds remaining budget: {required_tokens}/{remaining_tokens}"
        )
