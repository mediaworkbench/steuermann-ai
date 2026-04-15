"""Regression tests for _run_async_node_sync in performance_nodes.

Verifies that:
1. No coroutine is ever left un-awaited (no RuntimeWarning).
2. When no event loop is running the async node runs to completion.
3. When an event loop IS running the wrapper still executes via worker thread.
4. Exceptions inside the async node are swallowed and state is returned.
"""

import asyncio
import warnings

import pytest
from universal_agentic_framework.orchestration.performance_nodes import (
    _run_async_node_sync,
)


# ── Helpers ───────────────────────────────────────────────────────────

async def _echo_node(state: dict) -> dict:
    """Minimal async node that adds a marker to state."""
    return {**state, "ran": True}


async def _raising_node(state: dict) -> dict:
    raise RuntimeError("deliberate failure")


# ── No running loop ───────────────────────────────────────────────────

def test_runs_when_no_loop():
    state = {"messages": []}
    result = _run_async_node_sync(_echo_node, "skip", "err", state)
    assert result.get("ran") is True


def test_no_coroutine_warning_no_loop():
    """asyncio should emit no RuntimeWarning when there is no loop."""
    state = {"messages": []}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _run_async_node_sync(_echo_node, "skip", "err", state)
    assert result.get("ran") is True


# ── Running loop (sync wrapper called from inside async context) ───────

def test_runs_when_loop_running():
    """Inside a running loop the wrapper executes the node via worker thread."""
    state = {"messages": [], "ran": False}

    async def _caller():
        return _run_async_node_sync(_echo_node, "skip", "err", state)

    result = asyncio.run(_caller())
    assert result.get("ran") is True


def test_no_coroutine_warning_with_loop():
    """When loop is running, execution should still avoid RuntimeWarning leaks."""
    state = {"messages": [], "ran": False}

    async def _caller():
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return _run_async_node_sync(_echo_node, "skip", "err", state)

    result = asyncio.run(_caller())
    assert result.get("ran") is True


# ── Exception handling ────────────────────────────────────────────────

def test_exception_returns_state_unchanged():
    """Exceptions inside the async node must be swallowed; original state returned."""
    state = {"messages": [], "value": 42}
    result = _run_async_node_sync(_raising_node, "skip", "err", state)
    assert result == state


def test_exception_no_coroutine_warning():
    """Swallowed exceptions must not produce un-awaited coroutine warnings."""
    state = {"messages": []}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _run_async_node_sync(_raising_node, "skip", "err", state)
    assert result == state
