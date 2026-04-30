"""Graph utilities with echo and LLM-backed response node.

Provides a simple echo handler and an LLM-driven responder wired through
the factory with basic token budgeting enforcement.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .config import load_core_config
from .llm.factory import LLMFactory
from .llm.budget import estimate_tokens, TokenBudgetExceeded
from .memory.nodes import load_memory_node, update_memory_node
from .memory.factory import build_memory_backend


def run_graph(message: str) -> Dict[str, Any]:
    """Return a trivial echo response.

    Kept for backward compatibility with existing tests.
    """
    return {"response": f"Echo: {message}"}


def run_llm_graph(
    message: str,
    *,
    language: Optional[str] = None,
    prefer_local: bool = True,
    model_override: Optional[object] = None,
) -> Dict[str, Any]:
    """Run a minimal LLM-backed response node with token budgeting.

    - Loads core configuration (LLM providers and token budgets).
    - Selects model via LLMFactory based on `language` (falls back to fork language).
    - Enforces token budgeting using a simple estimator.

    Parameters:
        message: Input text from user.
        language: Optional language code (e.g., 'en', 'de'). If not provided, uses fork language.
        prefer_local: Try local (primary) provider first.
        model_override: Inject a fake/override model for testing; must implement `.invoke(prompt)` and
                        return an object with a `content` attribute or a string.

    Returns:
        A dict containing the response, tokens_used, and budget information.
    """
    config = load_core_config()
    fork_lang = config.fork.language
    lang = language or fork_lang

    # Token budgets
    default_budget = config.tokens.default_budget
    response_budget = config.tokens.per_node_budgets.get("response_node", default_budget)

    # Pre-check: input tokens should not exceed budget
    input_tokens = estimate_tokens(message)
    if input_tokens > response_budget:
        raise TokenBudgetExceeded(f"Input exceeds budget: {input_tokens}/{response_budget}")

    # Build model
    if model_override is not None:
        model = model_override
    else:
        factory = LLMFactory(config)
        model = factory.get_router_model(language=lang)

    # Invoke model
    result = getattr(model, "invoke", None)
    if callable(result):
        out = result(message)
    else:
        # Fallback: if model is callable itself
        out = model(message) if callable(model) else str(message)

    # Normalize output text
    if hasattr(out, "content"):
        response_text = out.content
    else:
        response_text = str(out)

    output_tokens = estimate_tokens(response_text)
    tokens_used = input_tokens + output_tokens

    if tokens_used > response_budget:
        raise TokenBudgetExceeded(
            f"Budget exhausted after generation: {tokens_used}/{response_budget}"
        )

    return {
        "response": response_text,
        "tokens_used": tokens_used,
        "token_budget": response_budget,
        "language": lang,
    }


def run_graph_with_memory(
    message: str,
    *,
    user_id: str,
    language: Optional[str] = None,
    prefer_local: bool = True,
    backend_override: Optional[object] = None,
    model_override: Optional[object] = None,
) -> Dict[str, Any]:
    """Run a minimal pipeline: load memory → generate response → summarize/update memory.

    Enforces budgets on input/response and on memory update text.
    Returns response and lightweight diagnostics.
    """
    config = load_core_config()
    lang = language or config.fork.language

    # Build memory backend
    backend = backend_override if backend_override is not None else build_memory_backend(config)

    # Prepare state and load memory
    state: Dict[str, Any] = {
        "user_id": user_id,
        "messages": [{"role": "user", "content": message}],
    }
    state = load_memory_node(state, backend=backend)

    # Generate response with LLM budgeting
    factory = LLMFactory(config)
    model = model_override if model_override is not None else factory.get_router_model(language=lang)

    input_tokens = estimate_tokens(message)
    resp_budget = config.tokens.per_node_budgets.get("response_node", config.tokens.default_budget)
    if input_tokens > resp_budget:
        raise TokenBudgetExceeded(f"Input exceeds budget: {input_tokens}/{resp_budget}")

    out = getattr(model, "invoke", None)
    out = out(message) if callable(out) else (model(message) if callable(model) else str(message))
    response_text = out.content if hasattr(out, "content") else str(out)

    output_tokens = estimate_tokens(response_text)
    tokens_used = input_tokens + output_tokens
    if tokens_used > resp_budget:
        raise TokenBudgetExceeded(f"Budget exhausted after generation: {tokens_used}/{resp_budget}")

    # Summarize and update memory (simple heuristic summary to keep deterministic tests)
    summary_text = f"Summary: {message} -> {response_text}"
    update_budget = config.tokens.per_node_budgets.get("update_memory", config.tokens.default_budget)
    if estimate_tokens(summary_text) > update_budget:
        raise TokenBudgetExceeded(
            f"Update memory exceeds budget: {estimate_tokens(summary_text)}/{update_budget}"
        )
    state = update_memory_node(state, text=summary_text, metadata={"type": "summary"}, backend=backend)

    return {
        "response": response_text,
        "tokens_used": tokens_used,
        "token_budget": resp_budget,
        "language": lang,
        "loaded_memory_count": len(state.get("loaded_memory", [])),
    }
