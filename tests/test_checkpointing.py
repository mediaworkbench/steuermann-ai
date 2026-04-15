from types import SimpleNamespace
import sys
import types

from universal_agentic_framework.orchestration.checkpointing import build_checkpointer


def _config(enabled=False, backend="sqlite", sqlite_path="./data/checkpoints/test.sqlite", postgres_dsn=None):
    return SimpleNamespace(
        checkpointing=SimpleNamespace(
            enabled=enabled,
            backend=backend,
            sqlite_path=sqlite_path,
            postgres_dsn=postgres_dsn,
        )
    )


def test_build_checkpointer_disabled_returns_none():
    config = _config(enabled=False)
    assert build_checkpointer(config=config, env={}) is None


def test_build_checkpointer_sqlite_env_override(monkeypatch):
    fake_module = types.ModuleType("langgraph.checkpoint.sqlite")

    class _FakeSqliteSaver:
        @staticmethod
        def from_conn_string(path: str):
            return {"backend": "sqlite", "path": path}

    fake_module.SqliteSaver = _FakeSqliteSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", fake_module)

    config = _config(enabled=False)
    env = {
        "CHECKPOINTER_ENABLED": "true",
        "CHECKPOINTER_BACKEND": "sqlite",
        "CHECKPOINTER_DB_PATH": "./data/checkpoints/env.sqlite",
    }

    checkpointer = build_checkpointer(config=config, env=env)
    assert checkpointer == {"backend": "sqlite", "path": "data/checkpoints/env.sqlite"}


def test_build_checkpointer_invalid_backend_returns_none():
    config = _config(enabled=True, backend="sqlite")
    env = {
        "CHECKPOINTER_ENABLED": "true",
        "CHECKPOINTER_BACKEND": "unknown",
    }
    assert build_checkpointer(config=config, env=env) is None
