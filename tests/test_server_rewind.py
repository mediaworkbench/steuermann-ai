"""History rewind applied at the LangGraph edge for Regenerate / Edit-and-resend."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def _rewind():
    from universal_agentic_framework.server import _rewind_history

    return _rewind_history


HISTORY = [
    {"role": "system", "content": "[SUMMARY] earlier", "type": "summary"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]


def test_rewind_one_turn_drops_last_question_and_answer():
    assert [m["content"] for m in _rewind()(HISTORY, 1)] == ["[SUMMARY] earlier", "q1", "a1"]


def test_rewind_past_available_turns_keeps_digest():
    assert [m["content"] for m in _rewind()(HISTORY, 5)] == ["[SUMMARY] earlier"]


@pytest.mark.parametrize("turns", [None, 0, -1, "x"])
def test_no_rewind_leaves_history(turns):
    assert _rewind()(HISTORY, turns) == HISTORY


def _final():
    from universal_agentic_framework.server import _final_assistant_content

    return _final_assistant_content


def test_final_content_is_last_assistant_message():
    out = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": " a "}]}
    assert _final()(out) == "a"


def test_final_content_strips_inline_reasoning_blocks():
    out = {"messages": [{"role": "assistant", "content": "<think>plan</think>Answer<thinking>x"}]}
    assert _final()(out) == "Answer"


def test_final_content_none_without_assistant_message():
    assert _final()({"messages": [{"role": "user", "content": "q"}]}) is None
    assert _final()({}) is None


@pytest.mark.asyncio
async def test_background_tasks_are_strongly_referenced_until_done():
    import asyncio

    from universal_agentic_framework import server

    gate = asyncio.Event()

    async def _work():
        await gate.wait()

    task = server._spawn_background(_work())
    assert task in server._BACKGROUND_TASKS
    gate.set()
    await task
    await asyncio.sleep(0)
    assert task not in server._BACKGROUND_TASKS
