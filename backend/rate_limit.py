"""Shared rate-limiter instance for the FastAPI adapter service.

Rate limiting is enabled by default but can be disabled for development or
testing by setting RATE_LIMIT_ENABLED=false.

Each authenticated user gets their own bucket. With auth enabled the key comes from
the trusted ``x-authenticated-user-id`` header set by the proxy; with auth disabled it
falls back to the single env user (dev bypass).
"""
from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: F401 (re-exported)
from slowapi.errors import RateLimitExceeded  # noqa: F401 (re-exported)

from backend.single_user import get_single_user_id


def _auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _get_user_or_ip(request: Request) -> str:
    """Key function: per-authenticated-user bucket (dev-bypass → single env user)."""
    if _auth_enabled():
        uid = (request.headers.get("x-authenticated-user-id") or "").strip()
        if uid:
            return f"user:{uid}"
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"
    return f"user:{get_single_user_id()}"


def _rate_limit_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")


limiter = Limiter(
    key_func=_get_user_or_ip,
    enabled=_rate_limit_enabled(),
)
