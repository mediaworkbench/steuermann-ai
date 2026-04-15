"""Startup secrets and configuration sanity checks.

On-premise Docker deployments should never ship with default shared secrets
or passwords. This module warns loudly when insecure defaults are detected so
operators catch misconfigurations before going live.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSECURE_DEFAULTS: dict[str, str] = {
    "POSTGRES_PASSWORD": "framework",
    "CHAT_ACCESS_TOKEN": "change-me",
}


def validate_secrets() -> None:
    """Warn in development and fail fast in production-like environments.

    Runs unconditionally at startup. Development keeps warning-only behavior,
    while production-like environments refuse to start with insecure values.
    """
    env = os.getenv("OTEL_ENVIRONMENT", "development").lower()
    is_prod_like = env in ("production", "prod", "staging", "docker")

    issues: list[str] = []

    for var, insecure_default in _INSECURE_DEFAULTS.items():
        value = os.getenv(var, insecure_default).strip()
        if not value or value == insecure_default:
            issues.append(var)

    if not issues:
        return

    message = (
        "Security notice: insecure/default values detected for: "
        f"{', '.join(issues)}. "
        + ("Refusing to start in production-like mode." if is_prod_like else "(acceptable in development)")
    )

    if is_prod_like:
        logger.error(message)
        raise RuntimeError(message)

    logger.debug(message)
