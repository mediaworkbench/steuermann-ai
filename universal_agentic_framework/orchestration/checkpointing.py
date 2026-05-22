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
    logger.info("Checkpointing enabled", backend="postgres")
    return checkpointer


async def setup_checkpointer(checkpointer: Any) -> None:
    """Open the connection pool and create checkpoint tables.

    Must be called once from the ASGI startup event before the graph handles
    any request. Safe to call even if tables already exist.
    """
    await checkpointer.conn.open(wait=True)
    await checkpointer.setup()
    logger.info("Checkpointer pool open and tables ready")


async def _prune_async(checkpointer: Any) -> None:
    """Run all three pruning DELETEs via the async connection pool."""
    async with checkpointer.conn.connection() as conn:
        async with conn.cursor() as cur:
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
