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

from backend.single_user import auth_enabled, get_single_user_id


def _get_user_or_ip(request: Request) -> str:
    """Key function: per-authenticated-user bucket (dev-bypass → single env user)."""
    if auth_enabled():
        uid = (request.headers.get("x-authenticated-user-id") or "").strip()
        if uid:
            return f"user:{uid}"
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"
    return f"user:{get_single_user_id()}"


def _login_client_ip(request: Request) -> str:
    # Unauthenticated login traffic arrives from the Next.js server, so the socket peer
    # is the proxy container for every user; the proxy forwards the browser's IP instead.
    forwarded = (request.headers.get("x-login-client-ip") or "").strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def login_ip_key(request: Request) -> str:
    """Key function for login: per-client-IP bucket (not one bucket for the proxy)."""
    return f"login-ip:{_login_client_ip(request)}"


def login_username_key(request: Request) -> str:
    """Key function for login: per-username bucket, so rotating a spoofable forwarded
    IP can't lift the brute-force cap on one account."""
    username = (request.headers.get("x-login-username") or "").strip().lower()
    if username:
        return f"login-user:{username}"
    return login_ip_key(request)


def _rate_limit_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")


limiter = Limiter(
    key_func=_get_user_or_ip,
    enabled=_rate_limit_enabled(),
)
