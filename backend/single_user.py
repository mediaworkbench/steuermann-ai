from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, status


def get_single_user_id() -> str:
    return os.getenv("AUTH_USERNAME", "anonymous").strip() or "anonymous"


def _auth_enabled() -> bool:
    """Whether auth enforcement is active. Inlined (not imported from ``backend.auth``)
    because that module imports this one — importing back would be circular."""
    return os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def require_api_access(
    x_chat_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("CHAT_ACCESS_TOKEN", "").strip()
    if not expected:
        # Fail closed: with auth enabled, an unset token would otherwise leave the
        # (host-published) FastAPI port open to spoofed x-authenticated-* identity headers.
        if _auth_enabled():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server misconfigured: CHAT_ACCESS_TOKEN is required when AUTH_ENABLED=true",
            )
        return

    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    presented = (x_chat_token or "").strip() or bearer
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing chat access token",
        )
