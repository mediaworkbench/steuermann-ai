"""Admin-only user & role management API.

All routes require an authenticated administrator (server-side ``require_admin`` gate,
in addition to the proxy/middleware UI gating). Only the three fixed roles are
provisionable — custom roles are not supported. New users (and password resets) get an
auto-generated temporary password returned **once** to the admin; the user must change it
on first login (``must_change_password``).
"""
from __future__ import annotations

import logging
import secrets
import uuid

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from backend.auth import VALID_ROLES, CurrentUser, hash_password, require_admin
from backend.db import UserStore
from backend.single_user import require_api_access

logger = logging.getLogger(__name__)

# Length (in bytes of entropy) for generated temporary passwords.
_TEMP_PASSWORD_BYTES = 12

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-users"],
    dependencies=[Depends(require_api_access), Depends(require_admin)],
)


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field(...)


class UpdateUserRequest(BaseModel):
    role: str | None = Field(default=None)
    status: str | None = Field(default=None)
    reset_password: bool = Field(default=False)


def _get_user_store(request: Request) -> UserStore:
    store = getattr(request.app.state, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")
    return store


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role!r}. Must be one of {', '.join(VALID_ROLES)}.",
        )


def _generate_temp_password() -> str:
    return secrets.token_urlsafe(_TEMP_PASSWORD_BYTES)


def _is_last_active_admin(store: UserStore, target: dict) -> bool:
    return (
        target.get("role_name") == "administrator"
        and target.get("status") == "active"
        and store.count_admins(active_only=True) <= 1
    )


@router.get("/roles")
def list_roles(request: Request) -> dict:
    """List the available (fixed) roles."""
    store = _get_user_store(request)
    return {"roles": store.get_all_roles()}


@router.get("/users")
def list_users(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List users (paginated)."""
    store = _get_user_store(request)
    users, total = store.get_all_users(limit=limit, offset=offset)
    return {"users": users, "total": total, "limit": limit, "offset": offset}


@router.post("/users", status_code=201)
def create_user(body: CreateUserRequest, request: Request) -> dict:
    """Provision a new user with an auto-generated temporary password."""
    store = _get_user_store(request)

    username = body.username.strip()
    email = body.email.strip()
    _validate_role(body.role)
    if "@" not in email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email address")

    if store.get_user_by_username_with_hash(username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    temporary_password = _generate_temp_password()
    try:
        user = store.create_user_with_password(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=hash_password(temporary_password),
            role_name=body.role,
            must_change_password=True,
        )
    except psycopg2.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this username or email already exists",
        )
    return {"user": user, "temporary_password": temporary_password}


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
) -> dict:
    """Update a user's role/status and/or reset their password."""
    store = _get_user_store(request)
    target = store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    is_self = user_id == current_user.user_id

    # ── Role change ───────────────────────────────────────────────
    if body.role is not None:
        _validate_role(body.role)
        demoting_admin = target.get("role_name") == "administrator" and body.role != "administrator"
        if demoting_admin and is_self:
            raise HTTPException(status_code=400, detail="You cannot demote yourself")
        if demoting_admin and _is_last_active_admin(store, target):
            raise HTTPException(status_code=400, detail="Cannot demote the last active administrator")

    # ── Status change ─────────────────────────────────────────────
    if body.status is not None:
        if body.status not in ("active", "inactive", "suspended"):
            raise HTTPException(status_code=400, detail=f"Invalid status: {body.status!r}")
        deactivating = body.status != "active"
        if deactivating and is_self:
            raise HTTPException(status_code=400, detail="You cannot deactivate yourself")
        if deactivating and _is_last_active_admin(store, target):
            raise HTTPException(status_code=400, detail="Cannot deactivate the last active administrator")

    # ── Apply ─────────────────────────────────────────────────────
    if body.role is not None:
        store.update_user_role(user_id, body.role)
    if body.status is not None:
        store.update_user_status(user_id, body.status)

    temporary_password = None
    if body.reset_password:
        temporary_password = _generate_temp_password()
        store.set_password_hash(
            user_id=user_id,
            password_hash=hash_password(temporary_password),
            must_change_password=True,
        )

    # Any of these (role/status/password) changes the target's authorization — invalidate
    # their existing sessions so the change takes effect immediately rather than on token expiry.
    if body.role is not None or body.status is not None or body.reset_password:
        store.bump_token_version(user_id)

    updated = store.get_user_by_id(user_id)
    result: dict = {"user": updated}
    if temporary_password is not None:
        result["temporary_password"] = temporary_password
    return result


@router.delete("/users/{user_id}", status_code=204, response_class=Response)
def delete_user(
    user_id: str,
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
) -> Response:
    """Delete a user (hard delete)."""
    store = _get_user_store(request)
    target = store.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user_id == current_user.user_id:
        raise HTTPException(status_code=400, detail="You cannot delete yourself")
    if _is_last_active_admin(store, target):
        raise HTTPException(status_code=400, detail="Cannot delete the last active administrator")

    store.delete_user(user_id)
    return Response(status_code=204)
