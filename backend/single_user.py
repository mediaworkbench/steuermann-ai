from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, status


def get_single_user_id() -> str:
    return os.getenv("AUTH_USERNAME", "anonymous").strip() or "anonymous"


def get_effective_user_id(requested_user_id: Optional[str] = None) -> str:
    return get_single_user_id()


def require_api_access(
    x_chat_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("CHAT_ACCESS_TOKEN", "").strip()
    if not expected:
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
