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

from fastapi import Depends, Header, HTTPException, Request, status

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


def dev_bypass_role() -> str:
    """Role granted by the AUTH_ENABLED=false dev bypass.

    Read from NEXT_PUBLIC_AUTH_USER_ROLE so the backend matches the frontend's dev role
    (defaults to administrator). Keeps the single-user dev experience consistent.
    """
    raw = os.getenv("NEXT_PUBLIC_AUTH_USER_ROLE", ADMIN_ROLE).strip().lower()
    return raw if raw in VALID_ROLES else ADMIN_ROLE


def _is_auth_path(request: Optional[Request]) -> bool:
    """True for the /api/auth/* routes (login/change-password) that must stay reachable."""
    try:
        return bool(request) and request.url.path.startswith("/api/auth/")
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_current_user(
    request: Request = None,
    x_authenticated_user_id: Optional[str] = Header(default=None),
    x_authenticated_username: Optional[str] = Header(default=None),
    x_authenticated_role: Optional[str] = Header(default=None),
) -> CurrentUser:
    """Resolve the authenticated user for the current request.

    Auth enabled: identity comes from the trusted proxy headers, then is re-validated
    against the DB (when a user store is wired) so suspensions, deletions, and role
    changes take effect immediately rather than living only in the 7-day JWT. A user with
    a pending forced password change is blocked from everything except the /api/auth/*
    routes. Auth disabled: fall back to the single env user with the dev-bypass role.
    """
    if not auth_enabled():
        uid = get_single_user_id()
        return CurrentUser(user_id=uid, username=uid, role=dev_bypass_role())

    user_id = (x_authenticated_user_id or "").strip()
    username = (x_authenticated_username or "").strip()
    role = (x_authenticated_role or "").strip().lower()

    if not user_id or role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authenticated identity",
        )

    # Re-validate against the DB. Skipped when no user store is wired (some unit-test and
    # dev setups), in which case the trusted header is taken at face value.
    store = getattr(request.app.state, "user_store", None) if request is not None else None
    if store is not None and hasattr(store, "get_user_by_id"):
        record = store.get_user_by_id(user_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer exists"
            )
        if (record.get("status") or "active") != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Account is not active"
            )
        db_role = record.get("role_name")
        if db_role in VALID_ROLES:
            role = db_role  # the DB is authoritative for the current role
        if record.get("must_change_password") and not _is_auth_path(request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Password change required"
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
