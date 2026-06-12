from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping, Optional

from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


def build_checkpointer(config: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Build an async Postgres checkpointer from config/env.

    Returns an AsyncPostgresSaver backed by an AsyncConnectionPool.
    The pool is created with open=False — call setup_checkpointer() from
    the ASGI startup event before the first request is served.

    Raises ValueError if no DSN is configured.
    """
    env_map = os.environ if env is None else env

    checkpointing_cfg = getattr(config, "checkpointing", None)
    postgres_dsn = (
        (env_map.get("CHECKPOINTER_POSTGRES_DSN") or "").strip()
        or (getattr(checkpointing_cfg, "postgres_dsn", None) if checkpointing_cfg else None)
    )

    if not postgres_dsn:
        raise ValueError(
            "Checkpointing requires a Postgres DSN. "
            "Set CHECKPOINTER_POSTGRES_DSN or checkpointing.postgres_dsn in core.yaml."
        )

    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(conninfo=postgres_dsn, open=False)
    checkpointer = AsyncPostgresSaver(pool)
    checkpointer._postgres_dsn = postgres_dsn  # kept for setup fallback
    logger.info("Checkpointing enabled", backend="postgres")
    return checkpointer


async def setup_checkpointer(checkpointer: Any) -> None:
    """Open the connection pool and create checkpoint tables.

    Must be called once from the ASGI startup event before the graph handles
    any request. Safe to call even if tables already exist.

    langgraph-checkpoint-postgres 3.x migrations 6-8 use CREATE INDEX
    CONCURRENTLY, which Postgres forbids inside a transaction block.
    When that error occurs we re-run all pending migrations via a direct
    autocommit connection so each statement commits individually.
    """
    await checkpointer.conn.open(wait=True)
    try:
        await checkpointer.setup()
    except Exception as exc:
        if "ActiveSqlTransaction" not in type(exc).__name__ and "CONCURRENTLY" not in str(exc):
            raise
        logger.warning(
            "setup() hit ActiveSqlTransaction (CREATE INDEX CONCURRENTLY); "
            "retrying via autocommit connection",
            error=str(exc),
        )
        await _setup_via_autocommit(checkpointer)
    logger.info("Checkpointer pool open and tables ready")


async def _setup_via_autocommit(checkpointer: Any) -> None:
    """Re-run pending checkpoint migrations using a direct autocommit connection.

    With autocommit=True every statement is its own implicit transaction, so
    CREATE INDEX CONCURRENTLY succeeds.  All migrations use IF NOT EXISTS, so
    re-running already-applied ones is safe.
    """
    import psycopg
    import psycopg.rows
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    dsn = getattr(checkpointer, "_postgres_dsn", None)
    if not dsn:
        raise RuntimeError("Cannot run autocommit setup: _postgres_dsn not set on checkpointer")

    migrations = AsyncPostgresSaver.MIGRATIONS

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # Always run migration 0 first so the checkpoint_migrations tracking
            # table exists before we query it. The loop then starts at 1 (or
            # wherever the DB left off) so migration 0 is never applied twice.
            await cur.execute(migrations[0])
            await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await cur.fetchone()
            version = -1 if row is None else row["v"]
            for v in range(max(version + 1, 1), len(migrations)):
                await cur.execute(migrations[v])
                await cur.execute(
                    "INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,)
                )


async def _prune_async(checkpointer: Any) -> None:
    """Run all pruning DELETEs via the async connection pool.

    Order matters: the poisoned-checkpoint repair runs *before* the keep-latest
    pass so the latter selects the genuine newest checkpoint, not a stale one.
    """
    async with checkpointer.conn.connection() as conn:
        async with conn.cursor() as cur:
            # ── Repair: remove poisoned UUIDv4 checkpoints ─────────────────────
            # A historical /compact bug wrote checkpoints with random uuid4 ids.
            # LangGraph ids are time-ordered uuid6; PostgresSaver selects the
            # latest via max(checkpoint_id) (lexical). A uuid4 (version nibble at
            # canonical position 15 == '4') sorts above all future uuid6 ids most
            # of the time, so it shadows every later turn — new messages stop
            # persisting and the keep-latest pass below would *preserve* the bad
            # checkpoint and delete the real ones. Delete uuid4 checkpoints only
            # for threads that still have a real uuid6 checkpoint, so a thread is
            # never left with zero checkpoints.
            await cur.execute(
                """
                DELETE FROM checkpoints c
                WHERE substring(c.checkpoint_id from 15 for 1) = '4'
                  AND EXISTS (
                      SELECT 1 FROM checkpoints v
                      WHERE v.thread_id = c.thread_id
                        AND v.checkpoint_ns = c.checkpoint_ns
                        AND substring(v.checkpoint_id from 15 for 1) = '6'
                  )
                """
            )
            if cur.rowcount:
                logger.warning(
                    "Removed poisoned uuid4 checkpoints", count=cur.rowcount
                )

            await cur.execute(
                """
                DELETE FROM checkpoints
                WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                    SELECT thread_id, checkpoint_ns, max(checkpoint_id)
                    FROM checkpoints
                    GROUP BY thread_id, checkpoint_ns
                )
                """
            )
            await cur.execute(
                """
                DELETE FROM checkpoint_blobs
                WHERE (thread_id, checkpoint_ns) NOT IN (
                    SELECT thread_id, checkpoint_ns FROM checkpoints
                )
                """
            )
            await cur.execute(
                """
                DELETE FROM checkpoint_writes
                WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                    SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
                )
                """
            )
        await conn.commit()
    logger.info("Checkpoint pruning complete")


async def prune_checkpoints(checkpointer: Any) -> None:
    """Keep only the latest checkpoint per (thread_id, checkpoint_ns).

    Cleans all three AsyncPostgresSaver tables: checkpoints, checkpoint_blobs,
    checkpoint_writes. Safe to call concurrently.
    """
    try:
        await _prune_async(checkpointer)
    except Exception as exc:
        logger.warning("Checkpoint pruning failed", error=str(exc))
