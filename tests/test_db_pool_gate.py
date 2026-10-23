"""DatabasePool waits for a free connection instead of failing at maxconn."""
from __future__ import annotations

import threading
import time

import pytest
from psycopg2 import pool as pg_pool

from backend.db import DatabasePool


class _FakeThreadedPool:
    def __init__(self) -> None:
        self.out = 0

    def getconn(self):
        self.out += 1
        return object()

    def putconn(self, conn) -> None:
        self.out -= 1


def _pool(maxconn: int, timeout: float) -> DatabasePool:
    db = DatabasePool.__new__(DatabasePool)
    db._pool = _FakeThreadedPool()
    db._slots = threading.BoundedSemaphore(maxconn)
    db._acquire_timeout = timeout
    return db


@pytest.fixture(autouse=True)
def _no_json_registration(monkeypatch):
    monkeypatch.setattr("backend.db.extras.register_default_json", lambda *a, **k: None)
    monkeypatch.setattr("backend.db.extras.register_default_jsonb", lambda *a, **k: None)


def test_checkout_waits_for_a_released_connection():
    db = _pool(maxconn=1, timeout=2.0)
    released = threading.Event()

    def _hold():
        with db.connection():
            time.sleep(0.2)
        released.set()

    t = threading.Thread(target=_hold)
    t.start()
    time.sleep(0.05)
    with db.connection():  # blocks until the holder releases, rather than raising
        assert released.is_set()
    t.join()
    assert db._pool.out == 0


def test_checkout_times_out_with_pool_error():
    db = _pool(maxconn=1, timeout=0.1)
    with db.connection():
        with pytest.raises(pg_pool.PoolError):
            with db.connection():
                pass
    # The slot is released again after the timeout path.
    with db.connection():
        pass
