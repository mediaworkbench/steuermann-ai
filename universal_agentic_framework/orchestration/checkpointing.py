from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_checkpointer(config: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Build a LangGraph checkpointer from config/env.

    Returns None when checkpointing is disabled or unavailable.
    """
    env_map = env or os.environ

    checkpointing_cfg = getattr(config, "checkpointing", None)
    cfg_enabled = bool(getattr(checkpointing_cfg, "enabled", False)) if checkpointing_cfg else False

    enabled = _as_bool(env_map.get("CHECKPOINTER_ENABLED"), cfg_enabled)
    if not enabled:
        logger.info("Checkpointing disabled")
        return None

    backend = (
        (env_map.get("CHECKPOINTER_BACKEND") or "").strip().lower()
        or (getattr(checkpointing_cfg, "backend", "sqlite") if checkpointing_cfg else "sqlite")
    )

    if backend == "sqlite":
        sqlite_path = (
            (env_map.get("CHECKPOINTER_DB_PATH") or "").strip()
            or (getattr(checkpointing_cfg, "sqlite_path", "./data/checkpoints/langgraph_checkpoints.sqlite") if checkpointing_cfg else "./data/checkpoints/langgraph_checkpoints.sqlite")
        )
        sqlite_file = Path(sqlite_path)
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(str(sqlite_file))
            logger.info("Checkpointing enabled", backend="sqlite", sqlite_path=str(sqlite_file))
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Failed to initialize sqlite checkpointer, falling back to in-memory",
                error=str(exc),
            )
            try:
                from langgraph.checkpoint.memory import InMemorySaver

                return InMemorySaver()
            except Exception as fallback_exc:
                logger.error(
                    "Failed to initialize in-memory fallback checkpointer",
                    error=str(fallback_exc),
                )
                return None

    if backend == "postgres":
        postgres_dsn = (
            (env_map.get("CHECKPOINTER_POSTGRES_DSN") or "").strip()
            or (getattr(checkpointing_cfg, "postgres_dsn", None) if checkpointing_cfg else None)
        )
        if not postgres_dsn:
            logger.warning("Postgres checkpointer configured but no DSN provided")
            return None

        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            checkpointer = PostgresSaver.from_conn_string(postgres_dsn)
            # Ensure tables exist before first invoke.
            checkpointer.setup()
            logger.info("Checkpointing enabled", backend="postgres")
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Failed to initialize postgres checkpointer",
                error=str(exc),
            )
            return None

    logger.warning("Unsupported checkpointer backend", backend=backend)
    return None
