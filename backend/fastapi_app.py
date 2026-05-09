from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.attachments import ChatAttachmentManager, ChatWorkspaceManager, UserWorkspaceFileManager
from backend.routers.chat import router as chat_router
from backend.routers.metrics import router as metrics_router
from backend.routers.settings import router as settings_router
from backend.routers.analytics import router as analytics_router
from backend.routers.conversations import router as conversations_router
from backend.routers.memories import router as memories_router
from backend.routers.workspace import router as workspace_router
from backend.db import (
    SettingsStore,
    AnalyticsStore,
    ConversationStore,
    ConversationAttachmentStore,
    ConversationWorkspaceStore,
    WorkspaceDocumentStore,
    init_db_pool,
)
from backend.rate_limit import limiter, RateLimitExceeded, _rate_limit_exceeded_handler
from backend.secrets import validate_secrets
from backend.version import get_framework_version
from universal_agentic_framework.monitoring.metrics import track_workspace_cleanup_deleted

logger = logging.getLogger(__name__)


def _ensure_writable_directory(path: str, *, setting_name: str) -> None:
    """Create and verify write access for a runtime directory."""
    directory = Path(path)
    probe_file = directory / ".steuermann-write-check"

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
    except Exception as exc:
        raise RuntimeError(
            "Workspace storage path is not writable: "
            f"{directory} (from {setting_name}). "
            "On Linux with bind mounts, set APP_UID/APP_GID in .env to your host user "
            "(APP_UID=$(id -u), APP_GID=$(id -g)), ensure the mapped host directories exist, "
            "and run chown -R <uid>:<gid> on them before rebuilding containers."
        ) from exc


def _validate_workspace_storage_permissions(chat_workspace_config: dict[str, object]) -> None:
    attachments_root = str(chat_workspace_config["attachments_root"])
    _ensure_writable_directory(attachments_root, setting_name="CHAT_ATTACHMENTS_ROOT")

    if bool(chat_workspace_config["workspace_enabled"]):
        workspace_root = chat_workspace_config.get("workspace_root")
        if isinstance(workspace_root, str) and workspace_root.strip():
            _ensure_writable_directory(workspace_root, setting_name="CHAT_WORKSPACE_ROOT")


def _run_workspace_startup_cleanup(app: FastAPI) -> None:
    workspace_store = getattr(app.state, "conversation_workspace_store", None)
    workspace_manager = getattr(app.state, "chat_workspace_manager", None)
    if workspace_store is None or workspace_manager is None:
        return

    fork_name = os.getenv("PROFILE_ID", "starter")
    deleted_items_total = 0
    expired_workspaces = workspace_store.list_expired_workspaces()

    for workspace in expired_workspaces:
        conversation_id = workspace.get("conversation_id")
        user_id = workspace.get("user_id")
        root_path = workspace.get("root_path")
        if not conversation_id or not root_path:
            continue

        try:
            deleted_items = workspace_manager.delete_workspace_tree(root_path=root_path)
            deleted_items_total += deleted_items
            workspace_store.log_workspace_operation(
                conversation_id=conversation_id,
                user_id=user_id or "unknown",
                operation="cleanup_workspace",
                result="allowed",
                reason="startup_ttl_sweep",
                target_path=root_path,
            )
            workspace_store.delete_workspace_record(conversation_id)
        except Exception as exc:
            logger.warning("Workspace startup cleanup failed for %s: %s", conversation_id, exc)
            try:
                workspace_store.log_workspace_operation(
                    conversation_id=conversation_id,
                    user_id=user_id or "unknown",
                    operation="cleanup_workspace",
                    result="failed",
                    reason=str(exc),
                    target_path=root_path,
                )
            except Exception:
                logger.warning("Failed to record cleanup failure for %s", conversation_id)

    if deleted_items_total > 0:
        track_workspace_cleanup_deleted(fork_name, deleted_items_total)

    logger.info(
        "Workspace startup cleanup complete",
        extra={
            "expired_workspaces_found": len(expired_workspaces),
            "deleted_items_total": deleted_items_total,
        },
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default=%s", name, value, default)
        return default


def _build_chat_workspace_config() -> dict[str, object]:
    attachments_root = os.getenv("CHAT_ATTACHMENTS_ROOT", "/tmp/steuermann-ai/chat-workspaces")
    workspace_root = os.getenv("CHAT_WORKSPACE_ROOT", "").strip() or None
    return {
        "attachments_root": attachments_root,
        "attachments_retention_hours": _env_int("CHAT_ATTACHMENTS_RETENTION_HOURS", 168),
        "workspace_enabled": _env_bool("CHAT_WORKSPACE_ENABLED", False),
        "workspace_root": workspace_root,
        "workspace_root_effective": workspace_root or f"{attachments_root}/<conversation_id>/workspace",
        "workspace_retention_hours": _env_int("CHAT_WORKSPACE_RETENTION_HOURS", 24),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_secrets()
    chat_workspace_config = _build_chat_workspace_config()
    _validate_workspace_storage_permissions(chat_workspace_config)
    app.state.chat_workspace_config = chat_workspace_config

    logger.info(
        "Chat workspace runtime config",
        extra={
            "attachments_root": chat_workspace_config["attachments_root"],
            "attachments_retention_hours": chat_workspace_config["attachments_retention_hours"],
            "workspace_enabled": chat_workspace_config["workspace_enabled"],
            "workspace_root_effective": chat_workspace_config["workspace_root_effective"],
            "workspace_retention_hours": chat_workspace_config["workspace_retention_hours"],
        },
    )

    db_pool = init_db_pool()
    app.state.db_pool = db_pool
    app.state.settings_store = SettingsStore(db_pool)
    app.state.analytics_store = AnalyticsStore(db_pool)
    app.state.conversation_store = ConversationStore(db_pool)
    app.state.conversation_attachment_store = ConversationAttachmentStore(db_pool)
    app.state.conversation_workspace_store = ConversationWorkspaceStore(db_pool)
    app.state.workspace_document_store = WorkspaceDocumentStore(db_pool)

    attachment_manager = ChatAttachmentManager()
    app.state.chat_attachment_manager = attachment_manager
    app.state.chat_workspace_manager = ChatWorkspaceManager(attachment_manager=attachment_manager)
    app.state.user_workspace_file_manager = UserWorkspaceFileManager(attachment_manager=attachment_manager)

    _run_workspace_startup_cleanup(app)

    try:
        yield
    finally:
        db_pool.close()
def create_app() -> FastAPI:
    """Create the FastAPI application for the adapter service."""
    app = FastAPI(title="Steuermann API", version=get_framework_version(), lifespan=lifespan)

    # Wire rate-limiter state so slowapi decorators can find it.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS: start from the hardcoded default whitelist then append any
    # additional origins supplied via CORS_ORIGINS (comma-separated).
    _default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://nextjs:3000",
    ]
    _extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    _allowed_origins = list(dict.fromkeys(_default_origins + _extra))  # dedup, preserve order

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    app.include_router(chat_router)
    app.include_router(settings_router)
    app.include_router(metrics_router)
    app.include_router(analytics_router)
    app.include_router(conversations_router)
    app.include_router(memories_router)
    app.include_router(workspace_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Backward-compatible health endpoint used by existing container checks.
        return {"status": "ok"}

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok", "check": "liveness"}

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        db_pool = getattr(app.state, "db_pool", None)
        if db_pool is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "db_pool_unavailable"},
            )

        try:
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": f"db_unreachable: {exc}"},
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "check": "readiness"},
        )

    @app.get("/metrics")
    async def metrics_export() -> Response:
        # Export the process-local Prometheus registry for scraping.
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.fastapi_app:app", host="0.0.0.0", port=8001, reload=False)
