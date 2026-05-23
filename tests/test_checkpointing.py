import sys
import types
from types import SimpleNamespace

import pytest

from universal_agentic_framework.orchestration.checkpointing import build_checkpointer


def _config(postgres_dsn=None):
    return SimpleNamespace(
        checkpointing=SimpleNamespace(postgres_dsn=postgres_dsn)
    )


def _mock_deps(monkeypatch):
    """Inject fake psycopg_pool and langgraph.checkpoint.postgres.aio into sys.modules.

    Returns (FakeAsyncPostgresSaver, FakePool) so callers can assert isinstance.
    """
    class _FakePool:
        def __init__(self, conninfo: str, open: bool = True):
            self.conninfo = conninfo

    fake_psycopg_pool = types.ModuleType("psycopg_pool")
    fake_psycopg_pool.AsyncConnectionPool = _FakePool
    monkeypatch.setitem(sys.modules, "psycopg_pool", fake_psycopg_pool)

    class _FakeAsyncPostgresSaver:
        def __init__(self, pool):
            self.conn = pool

        async def setup(self):
            pass

    fake_postgres_aio = types.ModuleType("langgraph.checkpoint.postgres.aio")
    fake_postgres_aio.AsyncPostgresSaver = _FakeAsyncPostgresSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", fake_postgres_aio)

    return _FakeAsyncPostgresSaver, _FakePool


def test_build_checkpointer_raises_when_no_dsn(monkeypatch):
    _mock_deps(monkeypatch)
    config = _config(postgres_dsn=None)
    with pytest.raises(ValueError, match="CHECKPOINTER_POSTGRES_DSN"):
        build_checkpointer(config=config, env={})


def test_build_checkpointer_uses_env_dsn(monkeypatch):
    FakeAsyncPostgresSaver, FakePool = _mock_deps(monkeypatch)
    config = _config(postgres_dsn=None)
    env = {"CHECKPOINTER_POSTGRES_DSN": "postgresql://user:pw@host/db"}

    checkpointer = build_checkpointer(config=config, env=env)

    assert isinstance(checkpointer, FakeAsyncPostgresSaver)
    assert isinstance(checkpointer.conn, FakePool)
    assert checkpointer.conn.conninfo == "postgresql://user:pw@host/db"


def test_build_checkpointer_uses_config_dsn(monkeypatch):
    FakeAsyncPostgresSaver, FakePool = _mock_deps(monkeypatch)
    config = _config(postgres_dsn="postgresql://user:pw@config-host/db")

    checkpointer = build_checkpointer(config=config, env={})

    assert isinstance(checkpointer, FakeAsyncPostgresSaver)
    assert checkpointer.conn.conninfo == "postgresql://user:pw@config-host/db"


def test_build_checkpointer_env_takes_precedence_over_config(monkeypatch):
    FakeAsyncPostgresSaver, _ = _mock_deps(monkeypatch)
    config = _config(postgres_dsn="postgresql://config/db")
    env = {"CHECKPOINTER_POSTGRES_DSN": "postgresql://env/db"}

    checkpointer = build_checkpointer(config=config, env=env)

    assert checkpointer.conn.conninfo == "postgresql://env/db"


@pytest.mark.integration
def test_postgres_checkpointer_persists_across_turns():
    """Two sequential graph invocations with the same thread_id accumulate messages."""
    pytest.importorskip("psycopg_pool")
    import os
    from universal_agentic_framework.orchestration.graph_builder import build_graph

    dsn = os.environ.get(
        "CHECKPOINTER_POSTGRES_DSN",
        "postgresql://framework:framework@localhost:5432/framework",
    )
    config = _config(postgres_dsn=dsn)
    checkpointer = build_checkpointer(config=config, env={})

    graph = build_graph()
    thread_id = "test-multi-turn-integration"
    cfg = {"configurable": {"thread_id": thread_id}}

    # First turn
    ct1 = checkpointer.get_tuple(cfg)
    existing = ct1.checkpoint.get("channel_values", {}).get("messages", []) if ct1 else []
    state1 = {
        "messages": existing + [{"role": "user", "content": "What is 2+2?"}],
        "user_id": "test-user",
        "language": "en",
    }
    graph.invoke(state1, config=cfg)

    # Verify checkpoint was written
    ct2 = checkpointer.get_tuple(cfg)
    assert ct2 is not None
    messages_after_turn1 = ct2.checkpoint.get("channel_values", {}).get("messages", [])
    assert any(m["role"] == "user" for m in messages_after_turn1)

    # Second turn — load from checkpoint
    state2 = {
        "messages": messages_after_turn1 + [{"role": "user", "content": "Now what is 3+3?"}],
        "user_id": "test-user",
        "language": "en",
    }
    graph.invoke(state2, config=cfg)

    ct3 = checkpointer.get_tuple(cfg)
    assert ct3 is not None
    messages_after_turn2 = ct3.checkpoint.get("channel_values", {}).get("messages", [])
    user_contents = [m["content"] for m in messages_after_turn2 if m["role"] == "user"]
    assert any("2+2" in c for c in user_contents)
    assert any("3+3" in c for c in user_contents)
