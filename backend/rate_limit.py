"""Shared rate-limiter instance for the FastAPI adapter service.

Rate limiting is enabled by default but can be disabled for development or
testing by setting RATE_LIMIT_ENABLED=false.

Rate limiting uses the configured single-user identity.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: F401 (re-exported)
from slowapi.errors import RateLimitExceeded  # noqa: F401 (re-exported)

from backend.single_user import get_single_user_id


def _get_user_or_ip(request: Request) -> str:
    """Key function: single-user ID."""
    return f"user:{get_single_user_id()}"


def _rate_limit_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")


limiter = Limiter(
    key_func=_get_user_or_ip,
    enabled=_rate_limit_enabled(),
)
