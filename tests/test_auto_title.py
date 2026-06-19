"""Tests for two-stage conversation auto-titling.

Covers the pure title sanitizer, the stage-selection logic of
``maybe_generate_conversation_title`` (with a fake store + stubbed auxiliary
LLM), and the SQL wrapper contract of ``ConversationStore.set_auto_title``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from backend.routers import chat as chat_module
from backend.routers.chat import (
    TITLE_UPGRADE_TURNS,
    _sanitize_title,
    maybe_generate_conversation_title,
)


# ── _sanitize_title ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("   ", ""),
        ("Tax filing deadlines", "Tax filing deadlines"),
        ('"Quoted Title"', "Quoted Title"),
        ("`code title`", "code title"),
        ("**Bold Title**", "Bold Title"),
        ("*partial* emphasis", "partial emphasis"),
        ("Title: Cooking pasta", "Cooking pasta"),
        ("title - Cooking pasta", "Cooking pasta"),
        ("Trailing punctuation!", "Trailing punctuation"),
        ("Line one\nLine two extra", "Line one"),
        ("  multiple   inner   spaces  ", "multiple inner spaces"),
        ("<think>reasoning\nover lines</think>Real Title", "Real Title"),
        ("<think>truncated reasoning only", ""),
    ],
)
def test_sanitize_title_cases(raw: str, expected: str) -> None:
    assert _sanitize_title(raw) == expected


def test_sanitize_title_clamps_word_count() -> None:
    result = _sanitize_title("one two three four five six seven eight nine ten")
    assert len(result.split(" ")) <= 8


def test_sanitize_title_clamps_char_length() -> None:
    result = _sanitize_title("x" * 200)
    assert len(result) <= 60


# ── maybe_generate_conversation_title (stage selection) ───────────────


class _FakeStore:
    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._conv = {
            "id": "c1",
            "user_id": "u1",
            "title": "old title",
            "metadata": metadata or {},
        }
        self._messages = messages or []
        self.set_calls: List[tuple] = []

    def get_conversation(self, conversation_id: str, user_id: str):
        if conversation_id == "c1" and user_id == "u1":
            return dict(self._conv)
        return None

    def get_messages(self, conversation_id: str, limit: int = 500, offset: int = 0):
        return list(self._messages)[offset : offset + limit]

    def set_auto_title(self, conversation_id: str, user_id: str, title: str, *, stage: int):
        self.set_calls.append((title, stage))
        return {"title": title}


def _turns(n: int) -> List[Dict[str, Any]]:
    """n user/assistant pairs."""
    msgs: List[Dict[str, Any]] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:  # noqa: D401 - stub
        return None

    def json(self) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeAsyncClient:
    """Records the last request payload; returns a configurable title."""

    next_title = "Generated Title"
    last_payload: Optional[Dict[str, Any]] = None
    called = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, **kwargs: Any) -> _FakeResp:
        _FakeAsyncClient.called = True
        _FakeAsyncClient.last_payload = kwargs.get("json")
        return _FakeResp(_FakeAsyncClient.next_title)


class _FakeProvider:
    api_base = "http://aux/v1"
    api_key = "test-key"


class _FakeLLM:
    def get_role_provider(self, role: str) -> _FakeProvider:
        return _FakeProvider()

    def get_role_model_name(self, role: str, language: str) -> str:
        return "openai/test-model"


class _FakeConfig:
    llm = _FakeLLM()


@pytest.fixture()
def aux_stub(monkeypatch: pytest.MonkeyPatch):
    _FakeAsyncClient.called = False
    _FakeAsyncClient.last_payload = None
    _FakeAsyncClient.next_title = "Generated Title"
    monkeypatch.setattr(chat_module, "load_core_config", lambda: _FakeConfig())
    monkeypatch.setattr(chat_module.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_no_turns_no_title(aux_stub) -> None:
    store = _FakeStore(metadata={}, messages=[])
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


@pytest.mark.asyncio
async def test_stage1_first_exchange(aux_stub) -> None:
    store = _FakeStore(metadata={}, messages=_turns(1))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == [("Generated Title", 1)]
    # Stage 1 uses the first-exchange prompt.
    prompt = aux_stub.last_payload["messages"][0]["content"]
    assert "captures" in prompt


@pytest.mark.asyncio
async def test_stage2_upgrade_at_threshold(aux_stub) -> None:
    store = _FakeStore(metadata={"title_source": "auto", "title_stage": 1}, messages=_turns(TITLE_UPGRADE_TURNS))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == [("Generated Title", 2)]
    # Stage 2 uses the summarize-from-context prompt.
    prompt = aux_stub.last_payload["messages"][0]["content"]
    assert "Summarize" in prompt


@pytest.mark.asyncio
async def test_below_threshold_no_upgrade(aux_stub) -> None:
    store = _FakeStore(metadata={"title_source": "auto", "title_stage": 1}, messages=_turns(2))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


@pytest.mark.asyncio
async def test_user_locked_no_call(aux_stub) -> None:
    store = _FakeStore(metadata={"title_source": "user"}, messages=_turns(TITLE_UPGRADE_TURNS))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


@pytest.mark.asyncio
async def test_stage2_already_done_no_call(aux_stub) -> None:
    store = _FakeStore(metadata={"title_source": "auto", "title_stage": 2}, messages=_turns(5))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


@pytest.mark.asyncio
async def test_legacy_long_conversation_left_untouched(aux_stub) -> None:
    """No provenance + already past the upgrade window → legacy title is preserved."""
    store = _FakeStore(metadata={}, messages=_turns(TITLE_UPGRADE_TURNS + 2))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


@pytest.mark.asyncio
async def test_title_is_sanitized_before_write(aux_stub) -> None:
    aux_stub.next_title = '"Hello World."'
    store = _FakeStore(metadata={}, messages=_turns(1))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == [("Hello World", 1)]


@pytest.mark.asyncio
async def test_empty_title_not_written(aux_stub) -> None:
    aux_stub.next_title = "   "
    store = _FakeStore(metadata={}, messages=_turns(1))
    await maybe_generate_conversation_title("c1", "u1", store)
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_missing_conversation_no_call(aux_stub) -> None:
    store = _FakeStore(metadata={}, messages=_turns(1))
    await maybe_generate_conversation_title("does-not-exist", "u1", store)
    assert store.set_calls == []
    assert aux_stub.called is False


# ── ConversationStore.set_auto_title (SQL wrapper contract) ────────────


class _FetchCursor:
    def __init__(self, row: Optional[Dict[str, Any]]) -> None:
        self.row = row
        self.executed: List[tuple] = []

    def __enter__(self) -> "_FetchCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, statement: str, params: tuple) -> None:
        self.executed.append((statement, params))

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self.row


class _FetchConn:
    def __init__(self, cursor: _FetchCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def __enter__(self) -> "_FetchConn":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def cursor(self, **_kwargs: object) -> _FetchCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


class _FetchPool:
    def __init__(self, row: Optional[Dict[str, Any]]) -> None:
        self._conn = _FetchConn(_FetchCursor(row))

    def connection(self) -> _FetchConn:
        return self._conn


def _store(row: Optional[Dict[str, Any]]):
    from backend.db import ConversationStore

    store = ConversationStore.__new__(ConversationStore)
    pool = _FetchPool(row)
    store._db_pool = pool  # type: ignore[attr-defined]
    return store, pool


def test_set_auto_title_sql_has_guards_and_params() -> None:
    row = {"id": "c1", "user_id": "u1", "title": "New Title", "metadata": {}}
    store, pool = _store(row)

    result = store.set_auto_title("c1", "u1", "New Title", stage=2)

    assert result is not None and result["title"] == "New Title"
    assert pool._conn.committed is True
    statement, params = pool._conn._cursor.executed[0]
    # Both atomic guards present.
    assert "title_source" in statement and "<> 'user'" in statement
    assert "title_stage" in statement and "<" in statement
    # Provenance payload + stage param.
    assert '"title_source": "auto"' in params[1]
    assert '"title_stage": 2' in params[1]
    assert params[-1] == 2  # stage guard comparison value


def test_set_auto_title_noop_returns_none() -> None:
    # fetchone -> None models a guard miss (locked title or non-advancing stage).
    store, _pool = _store(None)
    assert store.set_auto_title("c1", "u1", "Whatever", stage=1) is None
