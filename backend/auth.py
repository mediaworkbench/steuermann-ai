"""Authenticated identity + role resolution for the FastAPI adapter.

When ``AUTH_ENABLED`` is true, the authenticated identity is derived exclusively from
the trusted headers set by the Next.js proxy. The proxy is the only legitimate caller —
it proves itself with ``CHAT_ACCESS_TOKEN`` (enforced by ``require_api_access``) and is
responsible for stripping any client-supplied identity headers before setting its own.

When ``AUTH_ENABLED`` is false, we fall back to the single configured env user with the
administrator role so local development keeps working without a login flow (dev bypass).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from backend.single_user import get_single_user_id

try:  # argon2-cffi is a hard runtime dependency; guard only to keep imports robust.
    from argon2 import PasswordHasher
except Exception:  # pragma: no cover
    PasswordHasher = None  # type: ignore[assignment]


# The three fixed roles. No custom roles are supported.
USER_ROLE = "user"
RESEARCHER_ROLE = "researcher"
ADMIN_ROLE = "administrator"
VALID_ROLES = (USER_ROLE, RESEARCHER_ROLE, ADMIN_ROLE)


@dataclass(frozen=True)
class CurrentUser:
    """The resolved identity for a request."""

    user_id: str
    username: str
    role: str


def auth_enabled() -> bool:
    """Whether authentication enforcement is active (the security boundary)."""
    return os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def resolve_current_user(
    x_authenticated_user_id: Optional[str] = Header(default=None),
    x_authenticated_username: Optional[str] = Header(default=None),
    x_authenticated_role: Optional[str] = Header(default=None),
) -> CurrentUser:
    """Resolve the authenticated user for the current request.

    Auth enabled: identity + role come exclusively from the trusted proxy headers; a
    missing id or an unknown role is rejected with 401.
    Auth disabled: fall back to the single env user with the administrator role.
    """
    if not auth_enabled():
        uid = get_single_user_id()
        return CurrentUser(user_id=uid, username=uid, role=ADMIN_ROLE)

    user_id = (x_authenticated_user_id or "").strip()
    username = (x_authenticated_username or "").strip()
    role = (x_authenticated_role or "").strip().lower()

    if not user_id or role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authenticated identity",
        )
    return CurrentUser(user_id=user_id, username=username or user_id, role=role)


def current_user_id(user: CurrentUser = Depends(resolve_current_user)) -> str:
    """Convenience dependency returning just the authenticated user id."""
    return user.user_id


def require_admin(user: CurrentUser = Depends(resolve_current_user)) -> CurrentUser:
    """Dependency that allows only administrators (server-side defense in depth)."""
    if user.role != ADMIN_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )
    return user


def require_researcher_or_admin(
    user: CurrentUser = Depends(resolve_current_user),
) -> CurrentUser:
    """Dependency that allows researchers and administrators."""
    if user.role not in (RESEARCHER_ROLE, ADMIN_ROLE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher or administrator role required",
        )
    return user


# --- Password hashing (argon2id) ---

_password_hasher = PasswordHasher() if PasswordHasher is not None else None


def hash_password(password: str) -> str:
    """Hash a plaintext password with argon2id."""
    if _password_hasher is None:  # pragma: no cover
        raise RuntimeError("argon2-cffi is not installed")
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against an argon2id hash. Never raises."""
    if _password_hasher is None or not password_hash:
        return False
    try:
        return _password_hasher.verify(password_hash, password)
    except Exception:
        return False
