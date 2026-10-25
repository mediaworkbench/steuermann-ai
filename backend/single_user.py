from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, status


def get_single_user_id() -> str:
    return os.getenv("AUTH_USERNAME", "anonymous").strip() or "anonymous"


def auth_enabled() -> bool:
    """Whether authentication enforcement is active (the security boundary).

    Single definition for the backend; lives here (not in ``backend.auth``) because
    ``backend.auth`` imports this module — the reverse import would be circular.
    """
    return os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def require_api_access(
    x_chat_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("CHAT_ACCESS_TOKEN", "").strip()
    if not expected:
        # Fail closed: with auth enabled, an unset token would otherwise leave the
        # (host-published) FastAPI port open to spoofed x-authenticated-* identity headers.
        if auth_enabled():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server misconfigured: CHAT_ACCESS_TOKEN is required when AUTH_ENABLED=true",
            )
        return

    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    presented = (x_chat_token or "").strip() or bearer
    if not hmac.compare_digest(presented.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing chat access token",
        )
