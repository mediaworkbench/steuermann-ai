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
