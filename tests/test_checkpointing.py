import os
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
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_STACK_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set RUN_LIVE_STACK_TESTS=1 to run tests requiring a live PostgreSQL instance.",
)
async def test_postgres_checkpointer_persists_across_turns():
    """Two sequential graph invocations with the same thread_id accumulate messages.

    AsyncPostgresSaver requires a running event loop for both construction and
    all checkpoint operations, so this test is async throughout.
    """
    pytest.importorskip("psycopg_pool")
    from universal_agentic_framework.orchestration.checkpointing import setup_checkpointer
    from universal_agentic_framework.orchestration.graph_builder import build_graph

    dsn = os.environ.get(
        "CHECKPOINTER_POSTGRES_DSN",
        "postgresql://framework:framework@localhost:5432/framework",
    )
    config = _config(postgres_dsn=dsn)
    checkpointer = build_checkpointer(config=config, env={})
    # Open the connection pool (timeout=10 s so the test fails fast when Postgres
    # is not reachable rather than blocking for the pool's default 30 s retry window).
    await checkpointer.conn.open(wait=True, timeout=10.0)
    await checkpointer.setup()

    graph = build_graph()
    thread_id = "test-multi-turn-integration"
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        # First turn
        ct1 = await checkpointer.aget_tuple(cfg)
        existing = ct1.checkpoint.get("channel_values", {}).get("messages", []) if ct1 else []
        state1 = {
            "messages": existing + [{"role": "user", "content": "What is 2+2?"}],
            "user_id": "test-user",
            "language": "en",
        }
        await graph.ainvoke(state1, config=cfg)

        # Verify checkpoint was written
        ct2 = await checkpointer.aget_tuple(cfg)
        assert ct2 is not None
        messages_after_turn1 = ct2.checkpoint.get("channel_values", {}).get("messages", [])
        assert any(m["role"] == "user" for m in messages_after_turn1)

        # Second turn — load from checkpoint
        state2 = {
            "messages": messages_after_turn1 + [{"role": "user", "content": "Now what is 3+3?"}],
            "user_id": "test-user",
            "language": "en",
        }
        await graph.ainvoke(state2, config=cfg)

        ct3 = await checkpointer.aget_tuple(cfg)
        assert ct3 is not None
        messages_after_turn2 = ct3.checkpoint.get("channel_values", {}).get("messages", [])
        user_contents = [m["content"] for m in messages_after_turn2 if m["role"] == "user"]
        assert any("2+2" in c for c in user_contents)
        assert any("3+3" in c for c in user_contents)
    finally:
        # Always close the pool so the event loop is not left with open connections.
        await checkpointer.conn.close()


class _FakeCursor:
    def __init__(self, recorder):
        self._recorder = recorder
        self.rowcount = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, *args):
        self._recorder.append(sql)
        # Simulate one poisoned row removed on the repair DELETE.
        self.rowcount = 1 if "= '4'" in sql else 0


class _FakeConn:
    def __init__(self, recorder):
        self._recorder = recorder

    def cursor(self):
        return _FakeCursor(self._recorder)

    async def commit(self):
        self._recorder.append("COMMIT")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePoolConn:
    def __init__(self, recorder):
        self._recorder = recorder

    def connection(self):
        return _FakeConn(self._recorder)


@pytest.mark.asyncio
async def test_prune_runs_uuid4_repair_before_keep_latest():
    """The repair DELETE (version nibble == '4') runs before the keep-latest pass."""
    from universal_agentic_framework.orchestration.checkpointing import _prune_async

    recorder: list[str] = []
    checkpointer = SimpleNamespace(conn=_FakePoolConn(recorder))

    await _prune_async(checkpointer)

    sql_statements = [s for s in recorder if s != "COMMIT"]
    # First statement is the poisoned-checkpoint repair, identified by the version
    # nibble predicate; the keep-latest max() pass must come after it.
    assert "substring(c.checkpoint_id from 15 for 1) = '4'" in sql_statements[0]
    assert "= '6'" in sql_statements[0]  # guard: only delete when a real v6 exists
    keep_latest_idx = next(
        i for i, s in enumerate(sql_statements) if "max(checkpoint_id)" in s
    )
    assert keep_latest_idx > 0  # after the repair
