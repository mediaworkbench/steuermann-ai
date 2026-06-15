"""Authentication endpoints: DB-backed login and forced password change.

Login verification happens here (argon2id) against the ``users`` table. The Next.js
login route calls ``POST /api/auth/login`` and, on success, mints the session JWT — the
JWT/session lifecycle stays in the frontend. ``POST /api/auth/change-password`` lets a
user (typically on first login) replace an auto-generated temporary password.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.auth import CurrentUser, hash_password, resolve_current_user, verify_password
from backend.db import UserStore
from backend.single_user import require_api_access

logger = logging.getLogger(__name__)

# Minimum length for a user-chosen password.
_MIN_PASSWORD_LENGTH = 8

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    dependencies=[Depends(require_api_access)],
)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    must_change_password: bool


class MeResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    must_change_password: bool


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=1024)
    new_password: str = Field(..., min_length=1, max_length=1024)


def _get_user_store(request: Request) -> UserStore:
    store = getattr(request.app.state, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")
    return store


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request) -> LoginResponse:
    """Verify credentials against the DB and return the resolved identity + role."""
    store = _get_user_store(request)
    record = store.get_user_by_username_with_hash(body.username.strip())

    # Uniform 401 for unknown user / bad password to avoid user enumeration.
    if not record or not verify_password(body.password, record.get("password_hash") or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if (record.get("status") or "active") != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active",
        )

    role = record.get("role_name")
    if not role:
        # A user with no role assigned cannot be authorized.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No role assigned to this account",
        )

    return LoginResponse(
        user_id=record["user_id"],
        username=record["username"],
        email=record.get("email") or "",
        role=role,
        must_change_password=bool(record.get("must_change_password", False)),
    )


@router.get("/me", response_model=MeResponse)
def me(request: Request, current_user: CurrentUser = Depends(resolve_current_user)) -> MeResponse:
    """Return the caller's authoritative identity (DB-fresh role/status via re-validation).

    ``resolve_current_user`` already rejects deleted (401) / suspended (403) accounts and
    resolves the current DB role, so the frontend can trust this over the JWT's stale
    claims. (The `/api/auth/*` exemption means a forced-change user can still call this.)
    """
    store = getattr(request.app.state, "user_store", None)
    record = store.get_user_by_id(current_user.user_id) if store is not None else None
    return MeResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=(record or {}).get("email") or "",
        role=current_user.role,
        must_change_password=bool((record or {}).get("must_change_password", False)),
    )


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    request: Request,
    current_user: CurrentUser = Depends(resolve_current_user),
) -> dict:
    """Replace the authenticated user's password and clear the must-change flag."""
    store = _get_user_store(request)
    record = store.get_user_by_id_with_hash(current_user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not verify_password(body.current_password, record.get("password_hash") or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    new_password = body.new_password
    if len(new_password) < _MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"New password must be at least {_MIN_PASSWORD_LENGTH} characters",
        )
    if new_password == body.current_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must differ from the current password",
        )

    store.set_password_hash(
        user_id=current_user.user_id,
        password_hash=hash_password(new_password),
        must_change_password=False,
    )
    return {"ok": True}
