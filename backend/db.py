from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

import psycopg2
from psycopg2 import extras, pool


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str
    minconn: int
    maxconn: int


class DatabasePool:
    def __init__(self, config: DatabaseConfig) -> None:
        self._pool = pool.ThreadedConnectionPool(
            minconn=config.minconn,
            maxconn=config.maxconn,
            dsn=config.dsn,
        )
        # Keep schema compatible when DatabasePool is instantiated directly in tests/tools.
        try:
            _ensure_core_tables(self)
        except Exception:
            # Avoid making pool construction brittle; explicit init_db_pool still performs setup.
            pass

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        conn = self._pool.getconn()
        try:
            extras.register_default_json(conn, loads=json.loads)
            extras.register_default_jsonb(conn, loads=json.loads)
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()


def init_db_pool() -> DatabasePool:
    """Initialize database pool from environment variables."""
    # DB connection from environment (set by docker-compose)
    db_url = os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'framework')}:{os.getenv('POSTGRES_PASSWORD', 'framework')}@{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'framework')}",
    )
    
    # Pool size from environment with sensible default
    pool_size_str = os.getenv("DB_POOL_SIZE", "5")
    try:
        pool_size = max(1, int(pool_size_str))
    except (ValueError, TypeError):
        pool_size = 5
    
    db_config = DatabaseConfig(
        dsn=db_url,
        minconn=1,
        maxconn=pool_size,
    )
    db_pool = DatabasePool(db_config)
    _ensure_core_tables(db_pool)
    return db_pool


def _ensure_core_tables(db_pool: DatabasePool) -> None:
    """Initialize the core schema expected by tests and runtime code."""
    _ensure_settings_table(db_pool)
    _ensure_role_tool_permissions_table(db_pool)
    _ensure_llm_probe_table(db_pool)
    _ensure_admin_tables(db_pool)
    _ensure_analytics_tables(db_pool)
    _ensure_conversation_tables(db_pool)
    _ensure_co_occurrence_tables(db_pool)
    _ensure_global_settings_table(db_pool)
    _ensure_heartbeat_tables(db_pool)
    _ensure_procedural_overrides_table(db_pool)
    _ensure_memory_conflicts_table(db_pool)
    _ensure_memory_audit_log_table(db_pool)


def _ensure_settings_table(db_pool: DatabasePool) -> None:
    statement = """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            tool_toggles JSONB NOT NULL DEFAULT '{}'::jsonb,
            rag_config JSONB NOT NULL DEFAULT '{"collection":"","top_k":5}'::jsonb,
            analytics_preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
            preferred_model TEXT,
            preferred_models JSONB NOT NULL DEFAULT '{}'::jsonb,
            theme TEXT NOT NULL DEFAULT 'auto' CHECK (theme IN ('light', 'dark', 'auto')),
            display_sidebar_visible BOOLEAN NOT NULL DEFAULT TRUE,
            display_compact_mode BOOLEAN NOT NULL DEFAULT FALSE,
            language TEXT NOT NULL DEFAULT 'en',
            notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            notifications_sound BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            cur.execute(
                "ALTER TABLE IF EXISTS user_settings "
                "ADD COLUMN IF NOT EXISTS preferred_models JSONB NOT NULL DEFAULT '{}'::jsonb;"
            )
        conn.commit()


def _ensure_role_tool_permissions_table(db_pool: DatabasePool) -> None:
    """Admin-controlled allowlist of tool ids per role.

    A present row is authoritative for that role (an empty list blocks every
    tool). A *missing* row means the role is unconfigured and is treated as
    fail-closed (block all) by callers; `administrator` is never stored.
    """
    statement = """
        CREATE TABLE IF NOT EXISTS role_tool_permissions (
            role_name     TEXT PRIMARY KEY,
            allowed_tools JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
        conn.commit()


def _ensure_llm_probe_table(db_pool: DatabasePool) -> None:
    statement = """
        CREATE TABLE IF NOT EXISTS llm_capability_probes (
            profile_id TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            api_base TEXT,
            configured_tool_calling_mode TEXT NOT NULL CHECK (
                configured_tool_calling_mode IN ('native', 'structured', 'react')
            ),
            supports_bind_tools BOOLEAN,
            supports_tool_schema BOOLEAN,
            capability_mismatch BOOLEAN NOT NULL DEFAULT FALSE,
            status TEXT NOT NULL CHECK (status IN ('ok', 'warning', 'error')),
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            probed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (profile_id, provider_id, model_name)
        );
    """
    index_statement = """
        CREATE INDEX IF NOT EXISTS idx_llm_capability_probes_status
        ON llm_capability_probes(status);
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            cur.execute(index_statement)
        conn.commit()


def _ensure_admin_tables(db_pool: DatabasePool) -> None:
    """Create users and roles tables for admin management."""
    # Roles table
    roles_statement = """
        CREATE TABLE IF NOT EXISTS roles (
            role_id SERIAL PRIMARY KEY,
            role_name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    
    # Users table (for admin management)
    users_statement = """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role_id INTEGER REFERENCES roles(role_id),
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
            password_hash TEXT,
            must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
            token_version INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    # Create index on username and email for faster lookups
    index_statement = """
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """

    # The three fixed roles are always present. No custom roles are supported.
    roles_seed_statement = """
        INSERT INTO roles (role_name, description) VALUES
            ('user', 'Standard user'),
            ('researcher', 'Standard user plus access to the RAG explorer'),
            ('administrator', 'Full access including user management')
        ON CONFLICT (role_name) DO NOTHING;
    """

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(roles_statement)
            cur.execute(users_statement)
            cur.execute(index_statement)
            cur.execute(roles_seed_statement)
        conn.commit()

    # Idempotent column adds so existing databases gain the auth columns.
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS password_hash TEXT;"
            )
            cur.execute(
                "ALTER TABLE IF EXISTS users "
                "ADD COLUMN IF NOT EXISTS must_change_password BOOLEAN NOT NULL DEFAULT FALSE;"
            )
            cur.execute(
                "ALTER TABLE IF EXISTS users "
                "ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;"
            )
        conn.commit()


def _ensure_analytics_tables(db_pool: DatabasePool) -> None:
    """Create analytics tables for historical metrics and trends."""
    # Analytics events table for time-series data
    analytics_statement = """
        CREATE TABLE IF NOT EXISTS analytics_events (
            event_id SERIAL PRIMARY KEY,
            user_id TEXT,
            event_type TEXT NOT NULL,
            model_name TEXT,
            tokens_used INTEGER,
            request_duration_seconds FLOAT,
            status TEXT,
            profile_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    
    # Create indices for fast queries
    indices_statement = """
        CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_analytics_model ON analytics_events(model_name);
    """
    
    # Daily aggregated stats table
    daily_stats_statement = """
        CREATE TABLE IF NOT EXISTS daily_analytics_stats (
            stat_id SERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            total_requests INTEGER NOT NULL DEFAULT 0,
            total_tokens_used BIGINT NOT NULL DEFAULT 0,
            avg_token_usage FLOAT,
            avg_request_duration_seconds FLOAT,
            min_request_duration_seconds FLOAT,
            max_request_duration_seconds FLOAT,
            unique_users INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(analytics_statement)
            for sql in indices_statement.split(';'):
                if sql.strip():
                    cur.execute(sql)
            cur.execute(daily_stats_statement)
        conn.commit()


class SettingsStore:
    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_settings_table(self._db_pool)

    def get_user_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        statement = """
            SELECT user_id, tool_toggles, rag_config, analytics_preferences, preferred_model, preferred_models,
                   theme, display_sidebar_visible, display_compact_mode, language,
                   notifications_enabled, notifications_sound, updated_at
            FROM user_settings
            WHERE user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                row = cur.fetchone()
        if not row:
            return None
        return _normalize_settings_row(row)

    def upsert_user_settings(
        self,
        user_id: str,
        tool_toggles: Dict[str, bool],
        rag_config: Dict[str, Any],
        analytics_preferences: Optional[Dict[str, Any]] = None,
        preferred_model: Optional[str] = None,
        preferred_models: Optional[Dict[str, Optional[str]]] = None,
        theme: str = "auto",
        display_sidebar_visible: bool = True,
        display_compact_mode: bool = False,
        language: str = "en",
        notifications_enabled: bool = True,
        notifications_sound: bool = True,
    ) -> Dict[str, Any]:
        resolved_analytics_preferences = analytics_preferences or {}
        statement = """
            INSERT INTO user_settings (
                user_id, tool_toggles, rag_config, analytics_preferences, preferred_model, preferred_models,
                theme, display_sidebar_visible, display_compact_mode, language,
                notifications_enabled, notifications_sound, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                tool_toggles = EXCLUDED.tool_toggles,
                rag_config = EXCLUDED.rag_config,
                analytics_preferences = EXCLUDED.analytics_preferences,
                preferred_model = EXCLUDED.preferred_model,
                preferred_models = EXCLUDED.preferred_models,
                theme = EXCLUDED.theme,
                display_sidebar_visible = EXCLUDED.display_sidebar_visible,
                display_compact_mode = EXCLUDED.display_compact_mode,
                language = EXCLUDED.language,
                notifications_enabled = EXCLUDED.notifications_enabled,
                notifications_sound = EXCLUDED.notifications_sound,
                updated_at = NOW()
            RETURNING user_id, tool_toggles, rag_config, analytics_preferences, preferred_model, preferred_models,
                      theme, display_sidebar_visible, display_compact_mode, language,
                      notifications_enabled, notifications_sound, updated_at;
        """
        resolved_preferred_models = preferred_models or {}
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        user_id,
                        extras.Json(tool_toggles),
                        extras.Json(rag_config),
                        extras.Json(resolved_analytics_preferences),
                        preferred_model,
                        extras.Json(resolved_preferred_models),
                        theme,
                        display_sidebar_visible,
                        display_compact_mode,
                        language,
                        notifications_enabled,
                        notifications_sound,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_settings_row(row)


class RoleToolPermissionStore:
    """Per-role tool allowlists (admin-controlled).

    ``get_allowed_tools`` returns ``None`` when a role has no row — callers
    interpret that as fail-closed (block all). ``administrator`` is never stored
    (admins are always unrestricted).
    """

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_role_tool_permissions_table(self._db_pool)

    @staticmethod
    def _coerce_tool_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def get_allowed_tools(self, role_name: str) -> Optional[List[str]]:
        """Return the stored allowlist for a role, or ``None`` if no row exists."""
        statement = "SELECT allowed_tools FROM role_tool_permissions WHERE role_name = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (role_name,))
                row = cur.fetchone()
        if not row:
            return None
        return self._coerce_tool_list(row.get("allowed_tools"))

    def get_all_role_permissions(self) -> Dict[str, Optional[List[str]]]:
        """Return ``{role_name: allowed_tools}`` for every stored role."""
        statement = "SELECT role_name, allowed_tools FROM role_tool_permissions;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        return {row["role_name"]: self._coerce_tool_list(row.get("allowed_tools")) for row in rows}

    def set_allowed_tools(self, role_name: str, tools: List[str]) -> List[str]:
        """Upsert the explicit allowlist for a role (stored verbatim, even when full)."""
        normalized = [str(item) for item in (tools or [])]
        statement = """
            INSERT INTO role_tool_permissions (role_name, allowed_tools, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (role_name) DO UPDATE SET
                allowed_tools = EXCLUDED.allowed_tools,
                updated_at = NOW()
            RETURNING allowed_tools;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (role_name, extras.Json(normalized)))
                row = cur.fetchone()
            conn.commit()
        return self._coerce_tool_list(row.get("allowed_tools")) if row else normalized


class LLMCapabilityProbeStore:
    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_llm_probe_table(self._db_pool)

    def get_probe_result(self, profile_id: str, provider_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        statement = """
            SELECT profile_id, provider_id, model_name, api_base, configured_tool_calling_mode,
                   supports_bind_tools, supports_tool_schema, capability_mismatch, status,
                   error_message, metadata, probed_at
            FROM llm_capability_probes
            WHERE profile_id = %s AND provider_id = %s AND model_name = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (profile_id, provider_id, model_name))
                row = cur.fetchone()
        if not row:
            return None
        return _normalize_probe_row(row)

    def list_probe_results(self, profile_id: str, limit: int = 100) -> list[Dict[str, Any]]:
        statement = """
            SELECT profile_id, provider_id, model_name, api_base, configured_tool_calling_mode,
                   supports_bind_tools, supports_tool_schema, capability_mismatch, status,
                   error_message, metadata, probed_at
            FROM llm_capability_probes
            WHERE profile_id = %s
            ORDER BY probed_at DESC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (profile_id, limit))
                rows = cur.fetchall()
        return [_normalize_probe_row(row) for row in rows]

    def upsert_probe_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        statement = """
            INSERT INTO llm_capability_probes (
                profile_id, provider_id, model_name, api_base, configured_tool_calling_mode,
                supports_bind_tools, supports_tool_schema, capability_mismatch, status,
                error_message, metadata, probed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (profile_id, provider_id, model_name) DO UPDATE SET
                api_base = EXCLUDED.api_base,
                configured_tool_calling_mode = EXCLUDED.configured_tool_calling_mode,
                supports_bind_tools = EXCLUDED.supports_bind_tools,
                supports_tool_schema = EXCLUDED.supports_tool_schema,
                capability_mismatch = EXCLUDED.capability_mismatch,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                metadata = EXCLUDED.metadata,
                probed_at = NOW()
            RETURNING profile_id, provider_id, model_name, api_base, configured_tool_calling_mode,
                      supports_bind_tools, supports_tool_schema, capability_mismatch, status,
                      error_message, metadata, probed_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        result.get("profile_id"),
                        result.get("provider_id"),
                        result.get("model_name"),
                        result.get("api_base"),
                        result.get("configured_tool_calling_mode"),
                        result.get("supports_bind_tools"),
                        result.get("supports_tool_schema"),
                        bool(result.get("capability_mismatch", False)),
                        result.get("status"),
                        result.get("error_message"),
                        extras.Json(result.get("metadata") or {}),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_probe_row(row)


def _normalize_settings_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    tool_toggles = row.get("tool_toggles") or {}
    if isinstance(tool_toggles, str):
        tool_toggles = json.loads(tool_toggles)

    rag_config = row.get("rag_config") or {}
    if isinstance(rag_config, str):
        rag_config = json.loads(rag_config)

    analytics_preferences = row.get("analytics_preferences") or {}
    if isinstance(analytics_preferences, str):
        analytics_preferences = json.loads(analytics_preferences)

    preferred_models = row.get("preferred_models") or {}
    if isinstance(preferred_models, str):
        preferred_models = json.loads(preferred_models)

    updated_at = row.get("updated_at")
    if isinstance(updated_at, datetime):
        updated_at_value = updated_at.isoformat()
    elif updated_at is None:
        updated_at_value = None
    else:
        updated_at_value = str(updated_at)

    return {
        "user_id": row.get("user_id"),
        "tool_toggles": tool_toggles,
        "rag_config": rag_config,
        "analytics_preferences": analytics_preferences,
        "preferred_model": row.get("preferred_model"),
        "preferred_models": preferred_models,
        "theme": row.get("theme", "auto"),
        "display_sidebar_visible": row.get("display_sidebar_visible", True),
        "display_compact_mode": row.get("display_compact_mode", False),
        "language": row.get("language", "en"),
        "notifications_enabled": row.get("notifications_enabled", True),
        "notifications_sound": row.get("notifications_sound", True),
        "updated_at": updated_at_value,
    }


def _normalize_probe_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    probed_at = row.get("probed_at")
    if isinstance(probed_at, datetime):
        probed_at_value = probed_at.isoformat()
    elif probed_at is None:
        probed_at_value = None
    else:
        probed_at_value = str(probed_at)

    return {
        "profile_id": row.get("profile_id"),
        "provider_id": row.get("provider_id"),
        "model_name": row.get("model_name"),
        "api_base": row.get("api_base"),
        "configured_tool_calling_mode": row.get("configured_tool_calling_mode"),
        "supports_bind_tools": row.get("supports_bind_tools"),
        "supports_tool_schema": row.get("supports_tool_schema"),
        "capability_mismatch": bool(row.get("capability_mismatch", False)),
        "status": row.get("status"),
        "error_message": row.get("error_message"),
        "metadata": metadata,
        "probed_at": probed_at_value,
    }


class UserStore:
    """Manages admin user operations (users, roles, permissions)."""
    
    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def get_all_users(self, limit: int = 100, offset: int = 0) -> tuple[list[Dict[str, Any]], int]:
        """Get all users with pagination."""
        # Get total count
        count_statement = "SELECT COUNT(*) as count FROM users;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(count_statement)
                count_row = cur.fetchone()
                total = int(count_row["count"]) if count_row else 0
        
        # Get paginated results
        statement = """
            SELECT u.user_id, u.username, u.email, u.role_id, r.role_name, u.status,
                   u.must_change_password, u.created_at, u.updated_at
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.role_id
            ORDER BY u.created_at DESC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (limit, offset))
                rows = cur.fetchall()
        
        return [_normalize_user_row(row) for row in rows], total

    def get_active_user_ids(self) -> list[str]:
        """Return the user_id of every active user (for heartbeat per-user fan-out)."""
        statement = "SELECT user_id FROM users WHERE status = 'active' ORDER BY created_at;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        return [str(row["user_id"]) for row in rows if row.get("user_id")]

    def get_recently_active_user_ids(self, days: int) -> list[str]:
        """Active users with a conversation touched within the last ``days``.

        Used by the Dreaming Engine so dormant accounts aren't dreamed (cost
        control). Conversation ``updated_at`` is the activity signal.
        """
        statement = """
            SELECT u.user_id
            FROM users u
            WHERE u.status = 'active'
              AND EXISTS (
                  SELECT 1 FROM conversations c
                  WHERE c.user_id = u.user_id
                    AND c.updated_at >= NOW() - make_interval(days => %s)
              )
            ORDER BY u.created_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (max(0, int(days)),))
                rows = cur.fetchall()
        return [str(row["user_id"]) for row in rows if row.get("user_id")]

    def count_admins(self, active_only: bool = True) -> int:
        """Count users with the administrator role (active only by default)."""
        statement = """
            SELECT COUNT(*) AS count
            FROM users u JOIN roles r ON u.role_id = r.role_id
            WHERE r.role_name = 'administrator'
        """
        if active_only:
            statement += " AND u.status = 'active'"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement + ";")
                row = cur.fetchone()
        return int(row["count"]) if row else 0

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific user by ID."""
        statement = """
            SELECT u.user_id, u.username, u.email, u.role_id, r.role_name, u.status,
                   u.must_change_password, u.token_version, u.created_at, u.updated_at
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.role_id
            WHERE u.user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                row = cur.fetchone()
        
        return _normalize_user_row(row) if row else None

    def create_user(self, user_id: str, username: str, email: str, role_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new user."""
        statement = """
            INSERT INTO users (user_id, username, email, role_id, status)
            VALUES (%s, %s, %s, %s, 'active')
            RETURNING user_id, username, email, role_id, status, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, username, email, role_id))
                row = cur.fetchone()
            conn.commit()
        
        return _normalize_user_row(row)

    def update_user_status(self, user_id: str, status: str) -> Dict[str, Any]:
        """Update user status (active, inactive, suspended)."""
        if status not in ("active", "inactive", "suspended"):
            raise ValueError(f"Invalid status: {status}")
        
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "UPDATE users SET status = %s, updated_at = NOW() WHERE user_id = %s RETURNING user_id;",
                    (status, user_id),
                )
                updated = cur.fetchone()
                row = None
                if updated is not None:
                    cur.execute(
                        """
                        SELECT u.user_id, u.username, u.email, u.role_id, r.role_name,
                               u.status, u.must_change_password, u.created_at, u.updated_at
                        FROM users u LEFT JOIN roles r ON u.role_id = r.role_id
                        WHERE u.user_id = %s;
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
            conn.commit()

        return _normalize_user_row(row) if row else {}

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        statement = "DELETE FROM users WHERE user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (user_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        
        return deleted

    def get_all_roles(self) -> list[Dict[str, Any]]:
        """Get all available roles."""
        statement = "SELECT role_id, role_name, description, created_at FROM roles ORDER BY role_name;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        
        return [_normalize_role_row(row) for row in rows]

    def create_role(self, role_name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new role."""
        statement = """
            INSERT INTO roles (role_name, description)
            VALUES (%s, %s)
            RETURNING role_id, role_name, description, created_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (role_name, description))
                row = cur.fetchone()
            conn.commit()

        return _normalize_role_row(row)

    # ── Auth helpers ─────────────────────────────────────────────────────────

    def get_user_by_username_with_hash(self, username: str) -> Optional[Dict[str, Any]]:
        """Return user record including password_hash for login validation."""
        statement = """
            SELECT u.user_id, u.username, u.email, u.password_hash,
                   u.role_id, r.role_name, u.status, u.must_change_password, u.token_version
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.role_id
            WHERE u.username = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (username,))
                row = cur.fetchone()
        return dict(row) if row else None

    def get_user_by_id_with_hash(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return user record including password_hash, keyed by the trusted user id."""
        statement = """
            SELECT u.user_id, u.username, u.email, u.password_hash,
                   u.role_id, r.role_name, u.status, u.must_change_password, u.token_version
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.role_id
            WHERE u.user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def _resolve_role_id(self, cur, role_name: str) -> int:
        """Resolve an existing role_id. Raises if the role is not one of the fixed roles."""
        cur.execute("SELECT role_id FROM roles WHERE role_name = %s;", (role_name,))
        role_row = cur.fetchone()
        if not role_row:
            raise ValueError(f"Unknown role: {role_name!r}")
        return role_row["role_id"]

    def create_user_with_password(
        self,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
        role_name: str,
        must_change_password: bool = False,
    ) -> Dict[str, Any]:
        """Create a user with an argon2id password hash and an existing role.

        ``role_name`` must be one of the fixed seeded roles — unknown roles raise
        ``ValueError`` (custom roles are not supported).
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                role_id = self._resolve_role_id(cur, role_name)
                cur.execute(
                    """
                    INSERT INTO users (user_id, username, email, password_hash, role_id,
                                       status, must_change_password)
                    VALUES (%s, %s, %s, %s, %s, 'active', %s)
                    RETURNING user_id, username, email, role_id, status,
                              must_change_password, created_at, updated_at;
                    """,
                    (user_id, username, email, password_hash, role_id, must_change_password),
                )
                row = cur.fetchone()
                # Re-read with the role name joined for a complete normalized row.
                if row is not None:
                    cur.execute(
                        """
                        SELECT u.user_id, u.username, u.email, u.role_id, r.role_name,
                               u.status, u.must_change_password, u.created_at, u.updated_at
                        FROM users u LEFT JOIN roles r ON u.role_id = r.role_id
                        WHERE u.user_id = %s;
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
            conn.commit()
        return _normalize_user_row(row)

    def set_password_hash(
        self,
        user_id: str,
        password_hash: str,
        must_change_password: bool = False,
    ) -> bool:
        """Set a user's password hash and the must-change flag. Returns True if a row changed."""
        statement = """
            UPDATE users
            SET password_hash = %s, must_change_password = %s, updated_at = NOW()
            WHERE user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (password_hash, must_change_password, user_id))
                changed = cur.rowcount > 0
            conn.commit()
        return changed

    def bump_token_version(self, user_id: str) -> Optional[int]:
        """Invalidate every existing session for a user by incrementing token_version.

        Any JWT minted before the bump carries a stale ``tv`` claim and is rejected by
        ``resolve_current_user``. Returns the new version, or ``None`` if no user matched.
        """
        statement = """
            UPDATE users SET token_version = token_version + 1, updated_at = NOW()
            WHERE user_id = %s
            RETURNING token_version;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (user_id,))
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else None

    def update_user_role(self, user_id: str, role_name: str) -> Dict[str, Any]:
        """Assign an existing fixed role to a user. Unknown roles raise ValueError."""
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                role_id = self._resolve_role_id(cur, role_name)
                cur.execute(
                    """
                    UPDATE users SET role_id = %s, updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING user_id;
                    """,
                    (role_id, user_id),
                )
                updated = cur.fetchone()
                row = None
                if updated is not None:
                    cur.execute(
                        """
                        SELECT u.user_id, u.username, u.email, u.role_id, r.role_name,
                               u.status, u.must_change_password, u.created_at, u.updated_at
                        FROM users u LEFT JOIN roles r ON u.role_id = r.role_id
                        WHERE u.user_id = %s;
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
            conn.commit()
        return _normalize_user_row(row) if row else {}

    def seed_bootstrap_admin(
        self,
        username: str,
        email: str,
        password_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Idempotently seed the bootstrap administrator from env config.

        The bootstrap admin's ``user_id`` equals its ``username`` so the identity is
        identical in both the auth-enabled (DB login) and dev-bypass (env) paths.
        Does nothing if a user with this username already exists.
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT user_id FROM users WHERE username = %s;", (username,))
                if cur.fetchone():
                    return None
                role_id = self._resolve_role_id(cur, "administrator")
                cur.execute(
                    """
                    INSERT INTO users (user_id, username, email, password_hash, role_id,
                                       status, must_change_password)
                    VALUES (%s, %s, %s, %s, %s, 'active', FALSE)
                    ON CONFLICT DO NOTHING
                    RETURNING user_id;
                    """,
                    (username, username, email, password_hash, role_id),
                )
                inserted = cur.fetchone()
                row = None
                if inserted is not None:
                    cur.execute(
                        """
                        SELECT u.user_id, u.username, u.email, u.role_id, r.role_name,
                               u.status, u.must_change_password, u.created_at, u.updated_at
                        FROM users u LEFT JOIN roles r ON u.role_id = r.role_id
                        WHERE u.user_id = %s;
                        """,
                        (username,),
                    )
                    row = cur.fetchone()
            conn.commit()
        return _normalize_user_row(row) if row else None


def _normalize_user_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize user row from database."""
    if row is None:
        return {}
    
    created_at = row.get("created_at")
    updated_at = row.get("updated_at")
    
    return {
        "user_id": row.get("user_id"),
        "username": row.get("username"),
        "email": row.get("email"),
        "role_id": row.get("role_id"),
        "role_name": row.get("role_name"),
        "status": row.get("status"),
        "must_change_password": bool(row.get("must_change_password", False)),
        "token_version": int(row.get("token_version") or 0),
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at) if created_at else None,
        "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at) if updated_at else None,
    }


def _normalize_role_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize role row from database."""
    if row is None:
        return {}
    
    created_at = row.get("created_at")
    
    return {
        "role_id": row.get("role_id"),
        "role_name": row.get("role_name"),
        "description": row.get("description"),
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at) if created_at else None,
    }


def _ensure_conversation_tables(db_pool: DatabasePool) -> None:
    """Create conversations and messages tables."""
    conversations_statement = """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'New conversation',
            language TEXT NOT NULL DEFAULT 'en',
            profile_name TEXT,
            pinned BOOLEAN NOT NULL DEFAULT false,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    messages_statement = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            tokens_used INTEGER,
            model_name TEXT,
            response_time_ms INTEGER,
            tools_used JSONB,
            feedback TEXT CHECK (feedback IN ('up', 'down', NULL)),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    attachments_statement = """
        CREATE TABLE IF NOT EXISTS conversation_attachments (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            original_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            sha256 TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'deleted', 'expired')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
    """
    workspaces_statement = """
        CREATE TABLE IF NOT EXISTS conversation_workspaces (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            root_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'expired', 'deleted')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
    """
    workspace_operations_statement = """
        CREATE TABLE IF NOT EXISTS conversation_workspace_operations (
            id BIGSERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            attachment_id TEXT,
            operation TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('allowed', 'denied', 'failed')),
            reason TEXT,
            source_path TEXT,
            target_path TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    workspace_documents_statement = """
        CREATE TABLE IF NOT EXISTS workspace_documents (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            sha256 TEXT NOT NULL,
            content_text TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
            last_source TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    workspace_document_versions_statement = """
        CREATE TABLE IF NOT EXISTS workspace_document_versions (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES workspace_documents(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content_text TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            sha256 TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_doc_versions_doc_version
            ON workspace_document_versions(document_id, version);
        CREATE INDEX IF NOT EXISTS idx_ws_doc_versions_document_id
            ON workspace_document_versions(document_id);
    """
    chat_document_refs_statement = """
        CREATE TABLE IF NOT EXISTS chat_document_refs (
            id BIGSERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES workspace_documents(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            attached_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    indices_statement = """
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_conversations_pinned ON conversations(pinned) WHERE pinned = true;
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
        CREATE INDEX IF NOT EXISTS idx_attachments_conversation_id ON conversation_attachments(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_attachments_user_id ON conversation_attachments(user_id);
        CREATE INDEX IF NOT EXISTS idx_attachments_status ON conversation_attachments(status);
        CREATE INDEX IF NOT EXISTS idx_attachments_created_at ON conversation_attachments(created_at);
        CREATE INDEX IF NOT EXISTS idx_attachments_expires_at ON conversation_attachments(expires_at);
        CREATE INDEX IF NOT EXISTS idx_workspaces_user_id ON conversation_workspaces(user_id);
        CREATE INDEX IF NOT EXISTS idx_workspaces_status ON conversation_workspaces(status);
        CREATE INDEX IF NOT EXISTS idx_workspaces_expires_at ON conversation_workspaces(expires_at);
        CREATE INDEX IF NOT EXISTS idx_workspace_ops_conversation_id ON conversation_workspace_operations(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_workspace_ops_user_id ON conversation_workspace_operations(user_id);
        CREATE INDEX IF NOT EXISTS idx_workspace_ops_created_at ON conversation_workspace_operations(created_at);
        CREATE INDEX IF NOT EXISTS idx_workspace_ops_operation ON conversation_workspace_operations(operation);
        CREATE INDEX IF NOT EXISTS idx_workspace_documents_user_id ON workspace_documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_workspace_documents_updated_at ON workspace_documents(updated_at);
        CREATE INDEX IF NOT EXISTS idx_workspace_documents_filename ON workspace_documents(filename);
        CREATE INDEX IF NOT EXISTS idx_chat_document_refs_conversation_id ON chat_document_refs(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_chat_document_refs_message_id ON chat_document_refs(message_id);
        CREATE INDEX IF NOT EXISTS idx_chat_document_refs_document_id ON chat_document_refs(document_id);
        CREATE INDEX IF NOT EXISTS idx_chat_document_refs_user_id ON chat_document_refs(user_id);
    """
    # Additive column migrations for tables that may have been created by an earlier
    # version of the schema. ADD COLUMN IF NOT EXISTS is a no-op on new tables.
    schema_migrations = """
        ALTER TABLE workspace_documents
            ADD COLUMN IF NOT EXISTS last_source TEXT NOT NULL DEFAULT 'user';
        ALTER TABLE workspace_document_versions
            ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'user';
    """

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(conversations_statement)
            cur.execute(messages_statement)
            cur.execute(attachments_statement)
            cur.execute(workspaces_statement)
            cur.execute(workspace_operations_statement)
            cur.execute(workspace_documents_statement)
            for sql in workspace_document_versions_statement.split(';'):
                if sql.strip():
                    cur.execute(sql)
            cur.execute(chat_document_refs_statement)
            for sql in indices_statement.split(';'):
                if sql.strip():
                    cur.execute(sql)
            for sql in schema_migrations.split(';'):
                if sql.strip():
                    cur.execute(sql)
        conn.commit()

    # Optional trigram (pg_trgm) indexes for fast substring search on chat title +
    # message content. Isolated in its own connection/try so a missing extension or
    # insufficient privilege degrades to sequential ILIKE rather than failing startup.
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_title_trgm "
                    "ON conversations USING gin (title gin_trgm_ops);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_content_trgm "
                    "ON messages USING gin (content gin_trgm_ops);"
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — optional perf index, never fatal
        logger.warning(
            "pg_trgm search indexes unavailable (%s); chat search falls back to sequential ILIKE",
            exc,
        )


def _ensure_co_occurrence_tables(db_pool: DatabasePool) -> None:
    """Create durable co-occurrence edges table for memory linking."""
    statement = """
        CREATE TABLE IF NOT EXISTS co_occurrence_edges (
            user_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            related_memory_id TEXT NOT NULL,
            session_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
            decayed_strength DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, memory_id, related_memory_id)
        );
    """
    indices_statement = """
        CREATE INDEX IF NOT EXISTS idx_co_occurrence_user_memory
            ON co_occurrence_edges(user_id, memory_id);
        CREATE INDEX IF NOT EXISTS idx_co_occurrence_user_updated
            ON co_occurrence_edges(user_id, updated_at);
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            for sql in indices_statement.split(';'):
                if sql.strip():
                    cur.execute(sql)
        conn.commit()


class CoOccurrenceEdgeStore:
    """Durable read/write access for memory co-occurrence edges."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_co_occurrence_tables(self._db_pool)

    def record_co_occurrence_pair(
        self,
        *,
        user_id: str,
        memory_id: str,
        related_memory_id: str,
        session_id: str,
        event_time: Optional[datetime] = None,
    ) -> None:
        if memory_id == related_memory_id:
            return
        event_time = event_time or datetime.now(timezone.utc)
        evidence = {
            "total_count": 1,
            "last_session_id": session_id,
            "last_seen_at": event_time.isoformat(),
        }
        statement = """
            INSERT INTO co_occurrence_edges (
                user_id,
                memory_id,
                related_memory_id,
                session_evidence,
                decayed_strength,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (user_id, memory_id, related_memory_id)
            DO UPDATE SET
                session_evidence = jsonb_build_object(
                    'total_count', COALESCE((co_occurrence_edges.session_evidence->>'total_count')::INT, 0) + 1,
                    'last_session_id', EXCLUDED.session_evidence->>'last_session_id',
                    'last_seen_at', EXCLUDED.session_evidence->>'last_seen_at'
                ),
                decayed_strength = LEAST(1.0, GREATEST(co_occurrence_edges.decayed_strength, 0.0) + 0.1),
                updated_at = NOW();
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    statement,
                    (
                        user_id,
                        memory_id,
                        related_memory_id,
                        extras.Json(evidence),
                        0.1,
                    ),
                )
            conn.commit()

    def get_related_edges(
        self,
        *,
        user_id: str,
        memory_id: str,
        top_k: int,
        min_strength: float,
        updated_since: Optional[datetime] = None,
    ) -> list[Dict[str, Any]]:
        params: list[Any] = [user_id, memory_id]
        where = "WHERE user_id = %s AND memory_id = %s"
        if updated_since is not None:
            where += " AND updated_at >= %s"
            params.append(updated_since)

        statement = f"""
            SELECT related_memory_id, decayed_strength, session_evidence, updated_at
            FROM co_occurrence_edges
            {where}
            ORDER BY decayed_strength DESC, updated_at DESC
            LIMIT %s;
        """
        params.append(max(top_k * 2, top_k))

        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                rows = cur.fetchall()

        out: list[Dict[str, Any]] = []
        for row in rows:
            strength = float(row.get("decayed_strength") or 0.0)
            if strength < min_strength:
                continue
            evidence = row.get("session_evidence") or {}
            out.append(
                {
                    "memory_id": row.get("related_memory_id"),
                    "strength": strength,
                    "co_occurrence_count": int(evidence.get("total_count") or 0),
                }
            )
            if len(out) >= top_k:
                break
        return out

    def prune_old_edges(self, *, cutoff_time: datetime) -> int:
        statement = """
            DELETE FROM co_occurrence_edges
            WHERE updated_at < %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (cutoff_time,))
                pruned = cur.rowcount or 0
            conn.commit()
        return int(pruned)


def _ensure_global_settings_table(db_pool: DatabasePool) -> None:
    """Deployment-wide key/value settings (admin-controlled, not per-user)."""
    statement = """
        CREATE TABLE IF NOT EXISTS global_settings (
            key        TEXT PRIMARY KEY,
            value      JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
        conn.commit()


# global_settings key for the admin-controlled heartbeat beat rate (minutes).
# Single source of truth shared by the FastAPI admin endpoint (writer) and the
# LangGraph-embedded HeartbeatScheduler (reader).
HEARTBEAT_RATE_SETTING_KEY = "heartbeat_rate_minutes"


class GlobalSettingsStore:
    """Read/write access for deployment-wide settings.

    Values are stored as JSON so a single table can hold heterogeneous settings.
    ``get_setting`` returns ``None`` when no row exists; callers fall back to the
    config default in that case.
    """

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_global_settings_table(self._db_pool)

    def get_setting(self, key: str) -> Optional[Any]:
        """Return the stored JSON value for ``key``, or ``None`` if unset."""
        statement = "SELECT value FROM global_settings WHERE key = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (key,))
                row = cur.fetchone()
        if not row:
            return None
        value = row.get("value")
        if isinstance(value, str):
            value = json.loads(value)
        return value

    def set_setting(self, key: str, value: Any) -> Any:
        """Upsert the JSON value for ``key`` and return what was stored."""
        statement = """
            INSERT INTO global_settings (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
            RETURNING value;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (key, extras.Json(value)))
                row = cur.fetchone()
            conn.commit()
        stored = row.get("value") if row else value
        if isinstance(stored, str):
            stored = json.loads(stored)
        return stored


def _ensure_heartbeat_tables(db_pool: DatabasePool) -> None:
    """Run-history for the heartbeat scheduler (observability + idempotency).

    ``user_id`` is NULL for global-scope tasks and carries the user for per-user
    fan-out runs, so global and per-user beats share one log.
    """
    statement = """
        CREATE TABLE IF NOT EXISTS heartbeat_runs (
            id          BIGSERIAL PRIMARY KEY,
            task_name   TEXT NOT NULL,
            user_id     TEXT NULL,
            fired_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status      TEXT NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            detail      JSONB NOT NULL DEFAULT '{}'::jsonb
        );
    """
    # Idempotent column add so a table created before per-user fan-out picks up
    # user_id without a full reset (no-op on a freshly created table). Mirrors the
    # ADD COLUMN IF NOT EXISTS pattern used elsewhere in this module.
    alter_statement = "ALTER TABLE heartbeat_runs ADD COLUMN IF NOT EXISTS user_id TEXT NULL;"
    # (task_name, user_id, fired_at) backs per-(task,user) cooldown + filtered
    # reads; (fired_at) backs the cross-task inspector log query.
    index_statements = (
        """
        CREATE INDEX IF NOT EXISTS idx_heartbeat_runs_task_user_fired
            ON heartbeat_runs(task_name, user_id, fired_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_heartbeat_runs_fired
            ON heartbeat_runs(fired_at DESC);
        """,
    )
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            cur.execute(alter_statement)
            for index_statement in index_statements:
                cur.execute(index_statement)
        conn.commit()


class HeartbeatRunStore:
    """Durable record of each heartbeat tick (status/duration/detail)."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_heartbeat_tables(self._db_pool)

    def record_run(
        self,
        *,
        task_name: str,
        status: str,
        duration_ms: int = 0,
        detail: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        statement = """
            INSERT INTO heartbeat_runs (task_name, user_id, status, duration_ms, detail)
            VALUES (%s, %s, %s, %s, %s);
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    statement,
                    (task_name, user_id, status, int(duration_ms), extras.Json(detail or {})),
                )
            conn.commit()

    def recent_runs(
        self, task_name: str, limit: int = 5, *, user_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Return the most recent runs for a task, newest first.

        ``user_id`` is keyword-only (and after ``limit``) so existing positional
        callers — ``recent_runs(name, 1)`` — keep treating the second arg as the
        limit. When given, results are scoped to that user (per-user cooldown).
        """
        clauses = ["task_name = %s"]
        params: list[Any] = [task_name]
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        params.append(int(limit))
        statement = f"""
            SELECT task_name, user_id, fired_at, status, duration_ms, detail
            FROM heartbeat_runs
            WHERE {" AND ".join(clauses)}
            ORDER BY fired_at DESC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def recent_runs_all(
        self,
        limit: int = 50,
        *,
        task_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Inspector log: most recent runs across all tasks/users, newest first.

        Optional ``task_name`` / ``user_id`` filters narrow the view.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if task_name is not None:
            clauses.append("task_name = %s")
            params.append(task_name)
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        statement = f"""
            SELECT task_name, user_id, fired_at, status, duration_ms, detail
            FROM heartbeat_runs
            {where}
            ORDER BY fired_at DESC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def prune_runs(self, *, cutoff: datetime) -> int:
        """Delete run rows older than ``cutoff`` (keeps per-user volume bounded)."""
        statement = "DELETE FROM heartbeat_runs WHERE fired_at < %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (cutoff,))
                pruned = cur.rowcount or 0
            conn.commit()
        return int(pruned)


# ---------------------------------------------------------------------------
# Cognitive memory + Dreaming Engine contracts (plan-memory.md, Phase 1)
#
# Three per-user tables backing learned procedural rules, semantic-drift
# conflicts, and an undo-able audit log. All reads filter on user_id (strict
# multi-tenancy); aggregate_metrics() selects only COUNT/GROUP BY over non-PII
# columns so the admin dashboard never touches user text or ids.
# ---------------------------------------------------------------------------

# observing = engine is still accumulating evidence (not surfaced, not in prompt);
# proposed  = matured → surfaced to the user for sign-off; active = approved (in prompt).
_PROCEDURAL_STATUSES = ("observing", "proposed", "active", "rejected", "reverted")


def _ensure_procedural_overrides_table(db_pool: DatabasePool) -> None:
    """Learned behavioural rules merged onto the static YAML persona (Cycle D).

    Keyed ``(user_id, rule_key)``; ``tier`` 1/2/3 (format/style/locked-core);
    ``status`` proposed→active|rejected|reverted (engine only ever writes
    ``proposed`` — promotion to ``active`` requires explicit user approval).
    """
    statement = """
        CREATE TABLE IF NOT EXISTS procedural_overrides (
            user_id    TEXT NOT NULL,
            rule_key   TEXT NOT NULL,
            rule_text  TEXT NOT NULL,
            tier       SMALLINT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'proposed',
            confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            evidence   JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, rule_key)
        );
    """
    index_statement = """
        CREATE INDEX IF NOT EXISTS idx_procedural_overrides_user_status
            ON procedural_overrides(user_id, status);
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            cur.execute(index_statement)
        conn.commit()


class ProceduralOverrideStore:
    """Per-user learned procedural rules (proposed by the engine, approved by user)."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_procedural_overrides_table(self._db_pool)

    def upsert_proposal(
        self,
        *,
        user_id: str,
        rule_key: str,
        rule_text: str,
        tier: int,
        confidence: float = 0.0,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Insert/refresh a proposed rule. On conflict the text/tier/confidence/
        evidence are refreshed but ``status`` is left untouched — a rule the user
        already approved or rejected is never silently flipped back to proposed
        (status transitions go through ``set_status`` only)."""
        statement = """
            INSERT INTO procedural_overrides (
                user_id, rule_key, rule_text, tier, status, confidence, evidence
            )
            VALUES (%s, %s, %s, %s, 'proposed', %s, %s)
            ON CONFLICT (user_id, rule_key) DO UPDATE SET
                rule_text  = EXCLUDED.rule_text,
                tier       = EXCLUDED.tier,
                confidence = EXCLUDED.confidence,
                evidence   = EXCLUDED.evidence,
                updated_at = NOW()
            RETURNING *;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        user_id,
                        rule_key,
                        rule_text,
                        int(tier),
                        float(confidence),
                        extras.Json(evidence or {}),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else {}

    def get(self, user_id: str, rule_key: str) -> Optional[Dict[str, Any]]:
        """Read a single rule (for the engine's evidence-merge before re-writing)."""
        statement = "SELECT * FROM procedural_overrides WHERE user_id = %s AND rule_key = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, rule_key))
                row = cur.fetchone()
        return dict(row) if row else None

    def upsert_observation(
        self,
        *,
        user_id: str,
        rule_key: str,
        rule_text: str,
        tier: int,
        confidence: float = 0.0,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Accumulate evidence for a candidate rule (Cycle D). New rows start
        ``observing`` (not surfaced); on conflict the text/tier/confidence/evidence
        are refreshed but ``status`` is left untouched — a rule already proposed /
        approved / rejected keeps its decision while its evidence stays current.
        Maturity → ``proposed`` is a separate, guarded transition."""
        statement = """
            INSERT INTO procedural_overrides (
                user_id, rule_key, rule_text, tier, status, confidence, evidence
            )
            VALUES (%s, %s, %s, %s, 'observing', %s, %s)
            ON CONFLICT (user_id, rule_key) DO UPDATE SET
                rule_text  = EXCLUDED.rule_text,
                tier       = EXCLUDED.tier,
                confidence = EXCLUDED.confidence,
                evidence   = EXCLUDED.evidence,
                updated_at = NOW()
            RETURNING *;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        user_id,
                        rule_key,
                        rule_text,
                        int(tier),
                        float(confidence),
                        extras.Json(evidence or {}),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else {}

    def promote_to_proposed(self, *, user_id: str, rule_key: str) -> Optional[Dict[str, Any]]:
        """Transition a matured candidate ``observing`` → ``proposed`` (surface it
        for sign-off). Guarded on the current status so an already-decided rule is
        never reopened; returns the row only when the transition actually happened."""
        statement = """
            UPDATE procedural_overrides
            SET status = 'proposed', updated_at = NOW()
            WHERE user_id = %s AND rule_key = %s AND status = 'observing'
            RETURNING *;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, rule_key))
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else None

    def list_for_user(self, user_id: str) -> list[Dict[str, Any]]:
        statement = """
            SELECT * FROM procedural_overrides
            WHERE user_id = %s
            ORDER BY tier ASC, updated_at DESC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_active(self, user_id: str) -> list[Dict[str, Any]]:
        """Active rules only — the set merged into the prompt (Phase 5)."""
        statement = """
            SELECT * FROM procedural_overrides
            WHERE user_id = %s AND status = 'active'
            ORDER BY tier ASC, updated_at DESC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def set_status(self, *, user_id: str, rule_key: str, status: str) -> Optional[Dict[str, Any]]:
        """Transition a rule's status (approve/reject/revert). user_id-scoped so a
        user can never mutate another user's rule. Returns the row or None."""
        if status not in _PROCEDURAL_STATUSES:
            raise ValueError(f"Invalid procedural status: {status!r}")
        statement = """
            UPDATE procedural_overrides
            SET status = %s, updated_at = NOW()
            WHERE user_id = %s AND rule_key = %s
            RETURNING *;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (status, user_id, rule_key))
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else None

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Admin-safe rollup — COUNT/GROUP BY over status+tier only (no user_id,
        no rule_text). Structurally PII-free."""
        statement = """
            SELECT status, tier, COUNT(*) AS n
            FROM procedural_overrides
            GROUP BY status, tier;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        by_status: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        total = 0
        for row in rows:
            n = int(row["n"])
            total += n
            by_status[row["status"]] = by_status.get(row["status"], 0) + n
            by_tier[str(row["tier"])] = by_tier.get(str(row["tier"]), 0) + n
        return {
            "total": total,
            "by_status": by_status,
            "by_tier": by_tier,
            "proposed_pending": by_status.get("proposed", 0),
        }


def _ensure_memory_conflicts_table(db_pool: DatabasePool) -> None:
    """Semantic-drift conflicts surfaced to the user as a templated MCQ (Cycle B).

    ``UNIQUE(user_id, semantic_memory_id, episodic_memory_id)`` makes
    ``open_conflict`` idempotent (a re-detected contradiction is a no-op, never a
    duplicate queue entry).
    """
    statement = """
        CREATE TABLE IF NOT EXISTS memory_conflicts (
            conflict_id        TEXT PRIMARY KEY,
            user_id            TEXT NOT NULL,
            semantic_memory_id TEXT NOT NULL,
            episodic_memory_id TEXT NOT NULL,
            semantic_text      TEXT NOT NULL DEFAULT '',
            contradiction_text TEXT NOT NULL DEFAULT '',
            prior_confidence   DOUBLE PRECISION,
            new_confidence     DOUBLE PRECISION,
            choices            JSONB NOT NULL DEFAULT '[]'::jsonb,
            status             TEXT NOT NULL DEFAULT 'open',
            resolution         JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, semantic_memory_id, episodic_memory_id)
        );
    """
    index_statement = """
        CREATE INDEX IF NOT EXISTS idx_memory_conflicts_user_status
            ON memory_conflicts(user_id, status);
    """
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            cur.execute(index_statement)
        conn.commit()


class MemoryConflictStore:
    """Per-user semantic-drift conflict queue (open → resolved)."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_memory_conflicts_table(self._db_pool)

    def open_conflict(
        self,
        *,
        user_id: str,
        semantic_memory_id: str,
        episodic_memory_id: str,
        semantic_text: str = "",
        contradiction_text: str = "",
        prior_confidence: Optional[float] = None,
        new_confidence: Optional[float] = None,
        choices: Optional[list[Any]] = None,
    ) -> str:
        """Open a conflict if one isn't already tracked for this (user, semantic,
        episodic) triple. Idempotent: a duplicate detection is a no-op and the
        existing ``conflict_id`` is returned."""
        conflict_id = uuid.uuid4().hex
        statement = """
            INSERT INTO memory_conflicts (
                conflict_id, user_id, semantic_memory_id, episodic_memory_id,
                semantic_text, contradiction_text, prior_confidence, new_confidence,
                choices, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open')
            ON CONFLICT (user_id, semantic_memory_id, episodic_memory_id) DO NOTHING
            RETURNING conflict_id;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        conflict_id,
                        user_id,
                        semantic_memory_id,
                        episodic_memory_id,
                        semantic_text,
                        contradiction_text,
                        prior_confidence,
                        new_confidence,
                        extras.Json(choices or []),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    # Conflict already exists — fetch its id (no duplicate created).
                    cur.execute(
                        """
                        SELECT conflict_id FROM memory_conflicts
                        WHERE user_id = %s AND semantic_memory_id = %s
                              AND episodic_memory_id = %s;
                        """,
                        (user_id, semantic_memory_id, episodic_memory_id),
                    )
                    row = cur.fetchone()
            conn.commit()
        return str(row["conflict_id"]) if row else conflict_id

    def list_open(self, user_id: str) -> list[Dict[str, Any]]:
        statement = """
            SELECT * FROM memory_conflicts
            WHERE user_id = %s AND status = 'open'
            ORDER BY created_at DESC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def resolve(
        self,
        *,
        conflict_id: str,
        user_id: str,
        resolution: Optional[Dict[str, Any]] = None,
        status: str = "resolved",
    ) -> Optional[Dict[str, Any]]:
        """Resolve a conflict (user_id-scoped so cross-user resolution is impossible)."""
        statement = """
            UPDATE memory_conflicts
            SET status = %s, resolution = %s, updated_at = NOW()
            WHERE conflict_id = %s AND user_id = %s
            RETURNING *;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (status, extras.Json(resolution or {}), conflict_id, user_id),
                )
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else None

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Admin-safe rollup — counts by status only (no text/ids)."""
        statement = "SELECT status, COUNT(*) AS n FROM memory_conflicts GROUP BY status;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        by_status: Dict[str, int] = {}
        total = 0
        for row in rows:
            n = int(row["n"])
            total += n
            by_status[row["status"]] = by_status.get(row["status"], 0) + n
        return {"total": total, "by_status": by_status, "open": by_status.get("open", 0)}


def _ensure_memory_audit_log_table(db_pool: DatabasePool) -> None:
    """Append-only audit of every engine mutation, with a 7-day undo window.

    ``before_state``/``after_state`` snapshot the record for reversal (concept §6
    previous_state + new_state); ``reversible_until`` is set to created_at + the
    undo window for undoable actions, NULL otherwise. Retention (prune) runs on a
    floor wider than the undo window so the undo feed is never pruned out.
    """
    statement = """
        CREATE TABLE IF NOT EXISTS memory_audit_log (
            audit_id        TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            cycle           TEXT NOT NULL,
            action          TEXT NOT NULL,
            target_kind     TEXT,
            target_id       TEXT,
            before_state    JSONB,
            after_state     JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reversible_until TIMESTAMPTZ NULL
        );
    """
    index_statements = (
        """
        CREATE INDEX IF NOT EXISTS idx_memory_audit_user_cycle_created
            ON memory_audit_log(user_id, cycle, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_audit_user_reversible
            ON memory_audit_log(user_id, reversible_until);
        """,
    )
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            for index_statement in index_statements:
                cur.execute(index_statement)
        conn.commit()


class MemoryAuditLogStore:
    """Per-user audit log: cycle scheduling (last_cycle_run) + 7-day undo feed."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
        _ensure_memory_audit_log_table(self._db_pool)

    def record(
        self,
        *,
        user_id: str,
        cycle: str,
        action: str,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        reversible_until: Optional[datetime] = None,
    ) -> str:
        """Append an audit row. ``before_state`` must be written before any
        destructive op (undo-safe ordering). Returns the new ``audit_id``."""
        audit_id = uuid.uuid4().hex
        statement = """
            INSERT INTO memory_audit_log (
                audit_id, user_id, cycle, action, target_kind, target_id,
                before_state, after_state, reversible_until
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    statement,
                    (
                        audit_id,
                        user_id,
                        cycle,
                        action,
                        target_kind,
                        target_id,
                        extras.Json(before_state) if before_state is not None else None,
                        extras.Json(after_state) if after_state is not None else None,
                        reversible_until,
                    ),
                )
            conn.commit()
        return audit_id

    def last_cycle_run(self, *, user_id: str, cycle: str) -> Optional[datetime]:
        """Latest audit timestamp for a (user, cycle) — drives cost-tiered
        sub-cycle scheduling (a cycle is due when now − last_cycle_run ≥ cadence)."""
        statement = """
            SELECT MAX(created_at) AS last_run
            FROM memory_audit_log
            WHERE user_id = %s AND cycle = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, cycle))
                row = cur.fetchone()
        return row.get("last_run") if row else None

    def list_reversible(self, *, user_id: str, now: Optional[datetime] = None) -> list[Dict[str, Any]]:
        """Rows still inside their undo window (reversible_until > now), newest first."""
        now = now or datetime.now(timezone.utc)
        statement = """
            SELECT * FROM memory_audit_log
            WHERE user_id = %s AND reversible_until IS NOT NULL AND reversible_until > %s
            ORDER BY created_at DESC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, now))
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def prune(self, *, cutoff: datetime) -> int:
        """Delete rows older than ``cutoff`` (retention floor, wider than the undo
        window so a still-reversible row is never removed). Returns rows deleted."""
        statement = "DELETE FROM memory_audit_log WHERE created_at < %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (cutoff,))
                pruned = cur.rowcount or 0
            conn.commit()
        return int(pruned)

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Admin-safe rollup — counts by cycle+action only (no user_id, no states,
        no target_id). Feeds deletion/promotion-rate cards."""
        statement = """
            SELECT cycle, action, COUNT(*) AS n
            FROM memory_audit_log
            GROUP BY cycle, action;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement)
                rows = cur.fetchall()
        by_cycle: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        total = 0
        for row in rows:
            n = int(row["n"])
            total += n
            by_cycle[row["cycle"]] = by_cycle.get(row["cycle"], 0) + n
            by_action[row["action"]] = by_action.get(row["action"], 0) + n
        return {"total": total, "by_cycle": by_cycle, "by_action": by_action}


def _escape_like(value: str) -> str:
    """Escape LIKE/ILIKE metacharacters so a search term is matched literally.

    Used with ``ILIKE ... ESCAPE '\\'`` so a query such as ``50%`` or ``foo_bar``
    is treated as a substring rather than a wildcard pattern.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class ConversationStore:
    """Manages conversation persistence (conversations + messages)."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    # ── Conversations CRUD ──────────────────────────────────────────────

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str = "New conversation",
        language: str = "en",
        profile_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO conversations (id, user_id, title, language, profile_name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, user_id, title, language, profile_name,
                      pinned, metadata, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, user_id, title, language, profile_name))
                row = cur.fetchone()
            conn.commit()
        return _normalize_conversation_row(row)

    def list_conversations(
        self,
        user_id: str,
        q: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Dict[str, Any]], int]:
        """List conversations for a user, ordered by pinned first then updated_at desc.

        When ``q`` is provided, the list is filtered to conversations whose title or
        any message content contains the (case-insensitive) substring, and each row
        carries a ``match_snippet`` — the most recent matching message excerpt (NULL
        when only the title matched).
        """
        where = "WHERE c.user_id = %s"
        where_params: list[Any] = [user_id]

        q_clean = (q or "").strip()
        has_q = bool(q_clean)
        like = f"%{_escape_like(q_clean)}%" if has_q else None
        if has_q:
            where += (
                " AND (c.title ILIKE %s ESCAPE '\\' "
                "OR EXISTS (SELECT 1 FROM messages m "
                "WHERE m.conversation_id = c.id AND m.content ILIKE %s ESCAPE '\\'))"
            )
            where_params += [like, like]

        count_stmt = f"SELECT COUNT(*) AS count FROM conversations c {where};"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(count_stmt, tuple(where_params))
                count_row = cur.fetchone()
                total = int(count_row["count"]) if count_row else 0

        # Snippet via LATERAL only when searching. Placeholders appear in SQL order:
        # the LATERAL (FROM clause) precedes the WHERE clause, so its param comes first.
        snippet_select = ", snip.content AS match_snippet" if has_q else ""
        snippet_join = (
            " LEFT JOIN LATERAL (SELECT content FROM messages m "
            "WHERE m.conversation_id = c.id AND m.content ILIKE %s ESCAPE '\\' "
            "ORDER BY m.created_at DESC LIMIT 1) snip ON true"
            if has_q
            else ""
        )
        list_params: list[Any] = ([like] if has_q else []) + where_params + [limit, offset]

        list_stmt = f"""
            SELECT c.id, c.user_id, c.title, c.language, c.profile_name,
                   c.pinned, c.metadata, c.created_at, c.updated_at,
                   (SELECT content FROM messages WHERE conversation_id = c.id
                    ORDER BY created_at DESC LIMIT 1) AS last_message,
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count{snippet_select}
            FROM conversations c{snippet_join}
            {where}
            ORDER BY c.pinned DESC, c.updated_at DESC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(list_stmt, tuple(list_params))
                rows = cur.fetchall()

        return [_normalize_conversation_row(r) for r in rows], total

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a single conversation (without messages), scoped to its owner.

        ``user_id`` is required: a conversation owned by a different user resolves to
        ``None`` (the router turns that into a 404), so ownership cannot be bypassed.
        """
        statement = """
            SELECT id, user_id, title, language, profile_name,
                   pinned, metadata, created_at, updated_at
            FROM conversations WHERE id = %s AND user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, user_id))
                row = cur.fetchone()
        return _normalize_conversation_row(row) if row else None

    def update_conversation(
        self, conversation_id: str, user_id: str, **fields: Any
    ) -> Optional[Dict[str, Any]]:
        """Update mutable conversation fields (title, pinned, language, metadata).

        Scoped to ``user_id``: a non-owner update matches no row and returns ``None``.
        """
        allowed = {"title", "pinned", "language", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_conversation(conversation_id, user_id)

        set_clauses = []
        params: list[Any] = []
        for key, value in updates.items():
            if key == "metadata":
                set_clauses.append(f"{key} = %s::jsonb")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = %s")
                params.append(value)
        set_clauses.append("updated_at = NOW()")
        params.append(conversation_id)
        params.append(user_id)

        statement = f"""
            UPDATE conversations SET {', '.join(set_clauses)}
            WHERE id = %s AND user_id = %s
            RETURNING id, user_id, title, language, profile_name,
                      pinned, metadata, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                row = cur.fetchone()
            conn.commit()
        return _normalize_conversation_row(row) if row else None

    def set_auto_title(
        self, conversation_id: str, user_id: str, title: str, *, stage: int
    ) -> Optional[Dict[str, Any]]:
        """Persist an auto-generated title with provenance, atomically guarded.

        Records ``title_source="auto"`` and ``title_stage=stage`` in ``metadata``
        (JSONB-merged, so other keys survive). The write is a no-op (returns
        ``None``) when the title has been locked by a manual rename
        (``title_source == "user"``) or when ``stage`` would not advance the
        current ``title_stage`` — the latter serializes concurrent turns so a
        late Stage-1 write can never overwrite a Stage-2 title.
        """
        provenance = json.dumps({"title_source": "auto", "title_stage": stage})
        statement = """
            UPDATE conversations
            SET title = %s,
                metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
              AND COALESCE(metadata->>'title_source', 'auto') <> 'user'
              AND COALESCE((metadata->>'title_stage')::int, 0) < %s
            RETURNING id, user_id, title, language, profile_name,
                      pinned, metadata, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (title, provenance, conversation_id, user_id, stage))
                row = cur.fetchone()
            conn.commit()
        return _normalize_conversation_row(row) if row else None

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Hard-delete a conversation and its messages (CASCADE), scoped to its owner."""
        statement = "DELETE FROM conversations WHERE id = %s AND user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id, user_id))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    # ── Messages ────────────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        model_name: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        tools_used: Optional[list[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO messages (
                conversation_id, role, content, tokens_used, model_name,
                response_time_ms, tools_used, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, conversation_id, role, content, tokens_used, model_name,
                      response_time_ms, tools_used, feedback, metadata, created_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        conversation_id,
                        role,
                        content,
                        tokens_used,
                        model_name,
                        response_time_ms,
                        extras.Json(tools_used) if tools_used else None,
                        extras.Json(metadata or {}),
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        # Touch conversation updated_at
        self._touch_conversation(conversation_id, conn=None)
        return _normalize_message_row(row)

    def get_messages(
        self, conversation_id: str, limit: int = 500, offset: int = 0
    ) -> list[Dict[str, Any]]:
        statement = """
            SELECT id, conversation_id, role, content, tokens_used, model_name,
                   response_time_ms, tools_used, feedback, metadata, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, limit, offset))
                rows = cur.fetchall()
        return [_normalize_message_row(r) for r in rows]

    def update_message_feedback(
        self, message_id: int, feedback: Optional[str], conversation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if feedback is not None and feedback not in ("up", "down"):
            raise ValueError(f"Invalid feedback: {feedback}")
        # When conversation_id is supplied, scope the update to it so a message can only
        # be rated through its own (ownership-verified) conversation.
        if conversation_id is not None:
            statement = """
                UPDATE messages SET feedback = %s
                WHERE id = %s AND conversation_id = %s
                RETURNING id, conversation_id, role, content, tokens_used, model_name,
                          response_time_ms, tools_used, feedback, metadata, created_at;
            """
            params: tuple = (feedback, message_id, conversation_id)
        else:
            statement = """
                UPDATE messages SET feedback = %s
                WHERE id = %s
                RETURNING id, conversation_id, role, content, tokens_used, model_name,
                          response_time_ms, tools_used, feedback, metadata, created_at;
            """
            params = (feedback, message_id)
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, params)
                row = cur.fetchone()
            conn.commit()
        return _normalize_message_row(row) if row else None

    def update_assistant_node_trace_by_turn(
        self, turn_id: str, node_trace: list[Dict[str, Any]]
    ) -> bool:
        """Merge the complete Inspector node trace onto a specific assistant message.

        The post-response graph nodes (compress/summarize/update_memory/cache_stats) run
        after [DONE] in a background drain, so the trace persisted at stream time only
        covers the answer path. The LangGraph drain calls this once those nodes finish to
        write the full trace back. Matching by ``turn_id`` (stored in the message metadata
        at insert time) — not by recency — keeps this correct when a follow-up/queued turn
        inserts a newer assistant row during the drain. No-op (returns False) when no row
        matches. Overwrites only the ``node_trace`` metadata key.
        """
        statement = """
            UPDATE messages
            SET metadata = COALESCE(metadata, '{}'::jsonb)
                || jsonb_build_object('node_trace', %s::jsonb)
            WHERE metadata->>'turn_id' = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (extras.Json(node_trace), turn_id))
                updated = cur.rowcount
            conn.commit()
        return bool(updated)

    # ── Export ──────────────────────────────────────────────────────────

    def export_conversation(
        self, conversation_id: str, user_id: str, fmt: str = "json"
    ) -> Optional[Dict[str, Any] | str]:
        conv = self.get_conversation(conversation_id, user_id)
        if not conv:
            return None
        msgs = self.get_messages(conversation_id, limit=10000)

        if fmt == "markdown":
            lines = [f"# {conv['title']}\n"]
            lines.append(f"**Created**: {conv['created_at']}  ")
            lines.append(f"**Language**: {conv['language']}\n")
            lines.append("---\n")
            for m in msgs:
                speaker = "**You**" if m["role"] == "user" else "**AI Agent**"
                lines.append(f"{speaker} ({m['created_at']}):\n")
                lines.append(f"{m['content']}\n")
                lines.append("---\n")
            return "\n".join(lines)

        # Default: JSON
        return {"conversation": conv, "messages": msgs}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _touch_conversation(self, conversation_id: str, conn: Any = None) -> None:
        statement = "UPDATE conversations SET updated_at = NOW() WHERE id = %s;"
        if conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id,))
        else:
            with self._db_pool.connection() as c:
                with c.cursor() as cur:
                    cur.execute(statement, (conversation_id,))
                c.commit()


class ConversationAttachmentStore:
    """Manage conversation-scoped links to canonical workspace documents."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def create_attachment(
        self,
        attachment_id: str,
        conversation_id: str,
        user_id: str,
        original_name: str,
        stored_path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        extracted_text: str,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        statement = """
            WITH inserted AS (
                INSERT INTO chat_document_refs (
                    conversation_id, message_id, document_id, user_id
                )
                VALUES (%s, NULL, %s, %s)
                RETURNING conversation_id, document_id, user_id, attached_at
            )
            SELECT d.id, inserted.conversation_id, inserted.user_id,
                   d.filename AS original_name, d.stored_path,
                   d.mime_type, d.size_bytes, d.sha256,
                   d.content_text AS extracted_text,
                   'active' AS status,
                   inserted.attached_at AS created_at,
                   NULL::TIMESTAMPTZ AS expires_at
            FROM inserted
            JOIN workspace_documents d ON d.id = inserted.document_id;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        conversation_id,
                        attachment_id,
                        user_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_attachment_row(row)

    def get_attachment(
        self,
        attachment_id: str,
        conversation_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        where_clauses = ["d.id = %s"]
        params: list[Any] = [attachment_id]
        if conversation_id is not None:
            where_clauses.append("r.conversation_id = %s")
            params.append(conversation_id)

        statement = f"""
            SELECT d.id, r.conversation_id, r.user_id,
                   d.filename AS original_name, d.stored_path,
                   d.mime_type, d.size_bytes, d.sha256,
                   d.content_text AS extracted_text,
                   'active' AS status,
                   r.attached_at AS created_at,
                   NULL::TIMESTAMPTZ AS expires_at
            FROM chat_document_refs r
            JOIN workspace_documents d ON d.id = r.document_id
            WHERE {' AND '.join(where_clauses)};
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                row = cur.fetchone()
        return _normalize_attachment_row(row) if row else None

    def list_attachments(
        self,
        conversation_id: str,
        include_inactive: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Dict[str, Any]]:
        where = "WHERE r.conversation_id = %s"
        params: list[Any] = [conversation_id]

        statement = f"""
            SELECT d.id, r.conversation_id, r.user_id,
                   d.filename AS original_name, d.stored_path,
                   d.mime_type, d.size_bytes, d.sha256,
                   d.content_text AS extracted_text,
                   'active' AS status,
                   r.attached_at AS created_at,
                   NULL::TIMESTAMPTZ AS expires_at
            FROM chat_document_refs r
            JOIN workspace_documents d ON d.id = r.document_id
            {where}
            ORDER BY r.attached_at ASC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (*params, limit, offset))
                rows = cur.fetchall()
        return [_normalize_attachment_row(row) for row in rows]

    def get_attachments_by_ids(
        self,
        conversation_id: str,
        attachment_ids: list[str],
        include_inactive: bool = False,
    ) -> list[Dict[str, Any]]:
        if not attachment_ids:
            return []

        where_clauses = ["r.conversation_id = %s", "d.id = ANY(%s)"]
        params: list[Any] = [conversation_id, attachment_ids]

        statement = f"""
            SELECT d.id, r.conversation_id, r.user_id,
                   d.filename AS original_name, d.stored_path,
                   d.mime_type, d.size_bytes, d.sha256,
                   d.content_text AS extracted_text,
                   'active' AS status,
                   r.attached_at AS created_at,
                   NULL::TIMESTAMPTZ AS expires_at
            FROM chat_document_refs r
            JOIN workspace_documents d ON d.id = r.document_id
            WHERE {' AND '.join(where_clauses)};
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                rows = cur.fetchall()

        rows_by_id = {row["id"]: _normalize_attachment_row(row) for row in rows}
        return [rows_by_id[attachment_id] for attachment_id in attachment_ids if attachment_id in rows_by_id]

    def mark_attachment_deleted(
        self,
        attachment_id: str,
        conversation_id: Optional[str] = None,
    ) -> bool:
        where_clauses = ["document_id = %s"]
        params: list[Any] = [attachment_id]
        if conversation_id is not None:
            where_clauses.append("conversation_id = %s")
            params.append(conversation_id)

        statement = f"""
            DELETE FROM chat_document_refs
            WHERE {' AND '.join(where_clauses)};
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, tuple(params))
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def conversation_belongs_to_user(self, conversation_id: str, user_id: str) -> bool:
        statement = "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id, user_id))
                row = cur.fetchone()
        return bool(row)


class ConversationWorkspaceStore:
    """Manage conversation-scoped workspace lifecycle metadata."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def upsert_workspace(
        self,
        conversation_id: str,
        user_id: str,
        root_path: str,
        expires_at: Optional[datetime] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO conversation_workspaces (
                conversation_id, user_id, root_path, status, expires_at, last_activity_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (conversation_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                root_path = EXCLUDED.root_path,
                status = EXCLUDED.status,
                expires_at = EXCLUDED.expires_at,
                last_activity_at = NOW(),
                updated_at = NOW()
            RETURNING conversation_id, user_id, root_path, status,
                      created_at, updated_at, last_activity_at, expires_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (conversation_id, user_id, root_path, status, expires_at),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_row(row)

    def get_workspace(
        self,
        conversation_id: str,
        include_inactive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        where = "conversation_id = %s"
        params: list[Any] = [conversation_id]
        if not include_inactive:
            where += " AND status = 'active'"

        statement = f"""
            SELECT conversation_id, user_id, root_path, status,
                   created_at, updated_at, last_activity_at, expires_at
            FROM conversation_workspaces
            WHERE {where};
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                row = cur.fetchone()
        return _normalize_workspace_row(row) if row else None

    def touch_workspace(
        self,
        conversation_id: str,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        statement = """
            UPDATE conversation_workspaces
            SET last_activity_at = NOW(),
                updated_at = NOW(),
                status = 'active',
                expires_at = COALESCE(%s, expires_at)
            WHERE conversation_id = %s
            RETURNING conversation_id, user_id, root_path, status,
                      created_at, updated_at, last_activity_at, expires_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (expires_at, conversation_id))
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_row(row) if row else None

    def mark_workspace_deleted(self, conversation_id: str) -> bool:
        statement = """
            UPDATE conversation_workspaces
            SET status = 'deleted', updated_at = NOW()
            WHERE conversation_id = %s AND status != 'deleted';
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id,))
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def list_expired_workspaces(
        self,
        reference_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Dict[str, Any]]:
        effective_reference = reference_time or datetime.now(timezone.utc)
        statement = """
            SELECT conversation_id, user_id, root_path, status,
                   created_at, updated_at, last_activity_at, expires_at
            FROM conversation_workspaces
            WHERE status != 'deleted'
              AND expires_at IS NOT NULL
              AND expires_at <= %s
            ORDER BY expires_at ASC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (effective_reference, limit))
                rows = cur.fetchall()
        return [_normalize_workspace_row(row) for row in rows]

    def delete_workspace_record(self, conversation_id: str) -> bool:
        statement = "DELETE FROM conversation_workspaces WHERE conversation_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def log_workspace_operation(
        self,
        conversation_id: str,
        user_id: str,
        operation: str,
        result: str,
        attachment_id: Optional[str] = None,
        reason: Optional[str] = None,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if result not in ("allowed", "denied", "failed"):
            raise ValueError(f"Invalid workspace operation result: {result}")

        statement = """
            INSERT INTO conversation_workspace_operations (
                conversation_id, user_id, attachment_id, operation, result,
                reason, source_path, target_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, conversation_id, user_id, attachment_id, operation, result,
                      reason, source_path, target_path, created_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        conversation_id,
                        user_id,
                        attachment_id,
                        operation,
                        result,
                        reason,
                        source_path,
                        target_path,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_operation_row(row)

    def list_workspace_operations(
        self,
        conversation_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Dict[str, Any]]:
        statement = """
            SELECT id, conversation_id, user_id, attachment_id, operation, result,
                   reason, source_path, target_path, created_at
            FROM conversation_workspace_operations
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, limit, offset))
                rows = cur.fetchall()
        return [_normalize_workspace_operation_row(row) for row in rows]


class WorkspaceVersionConflictError(Exception):
    """Raised when expected_version does not match the current document version.

    Carries the current (server-side) version so callers can surface it to the UI.
    """

    def __init__(self, current_version: int, expected_version: int) -> None:
        self.current_version = current_version
        self.expected_version = expected_version
        super().__init__(
            f"Version conflict: expected v{expected_version}, but document is at v{current_version}"
        )


class WorkspaceDocumentStore:
    """Manage persistent user-scoped workspace documents."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool

    def create_document(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        stored_path: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        content_text: str,
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO workspace_documents (
                id, user_id, filename, stored_path, mime_type,
                size_bytes, sha256, content_text, version, last_source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, 'user')
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version, last_source,
                      created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    statement,
                    (
                        document_id,
                        user_id,
                        filename,
                        stored_path,
                        mime_type,
                        size_bytes,
                        sha256,
                        content_text,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_document_row(row)

    def get_document(self, document_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        statement = """
            SELECT id, user_id, filename, stored_path, mime_type,
                   size_bytes, sha256, content_text, version, last_source,
                   created_at, updated_at
            FROM workspace_documents
            WHERE id = %s AND user_id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (document_id, user_id))
                row = cur.fetchone()
        return _normalize_workspace_document_row(row) if row else None

    def list_documents(self, user_id: str, limit: int = 200, offset: int = 0) -> list[Dict[str, Any]]:
        statement = """
            SELECT id, user_id, filename, stored_path, mime_type,
                   size_bytes, sha256, content_text, version, last_source,
                   created_at, updated_at
            FROM workspace_documents
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, limit, offset))
                rows = cur.fetchall()
        return [_normalize_workspace_document_row(row) for row in rows]

    def get_documents_by_ids(self, user_id: str, document_ids: list[str]) -> list[Dict[str, Any]]:
        if not document_ids:
            return []

        statement = """
            SELECT id, user_id, filename, stored_path, mime_type,
                   size_bytes, sha256, content_text, version, last_source,
                   created_at, updated_at
            FROM workspace_documents
            WHERE user_id = %s AND id = ANY(%s);
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, document_ids))
                rows = cur.fetchall()

        rows_by_id = {row["id"]: _normalize_workspace_document_row(row) for row in rows}
        return [rows_by_id[document_id] for document_id in document_ids if document_id in rows_by_id]

    def update_document_content(
        self,
        document_id: str,
        user_id: str,
        content_text: str,
        size_bytes: int,
        sha256: str,
        expected_version: Optional[int] = None,
        source: str = "user",
    ) -> Optional[Dict[str, Any]]:
        """Overwrite a document's content, snapshotting the prior state into versions.

        The snapshot records the PRE-update content; its `source` is the document's
        current `last_source` (the author of that prior content). The new content's
        author is recorded as `last_source = source`.

        With `expected_version` set, the current version is checked under a row lock
        (`FOR UPDATE`) before any write — a mismatch raises
        `WorkspaceVersionConflictError` and writes nothing (no spurious snapshot).
        Returns None only when the document does not exist.
        """
        import uuid as _uuid

        lock_statement = """
            SELECT version FROM workspace_documents
            WHERE id = %s AND user_id = %s
            FOR UPDATE;
        """
        snapshot_statement = """
            INSERT INTO workspace_document_versions
                (id, document_id, user_id, version, content_text, size_bytes, sha256, source, created_at)
            SELECT %s, id, user_id, version, content_text, size_bytes, sha256, last_source, NOW()
            FROM workspace_documents
            WHERE id = %s AND user_id = %s
            ON CONFLICT (document_id, version) DO NOTHING;
        """
        update_statement = """
            UPDATE workspace_documents
            SET content_text = %s,
                size_bytes = %s,
                sha256 = %s,
                version = version + 1,
                last_source = %s,
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version, last_source,
                      created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                # Lock the row and read the current version before any write so the
                # version check is race-free (check-then-write under FOR UPDATE).
                cur.execute(lock_statement, (document_id, user_id))
                locked = cur.fetchone()
                if not locked:
                    conn.rollback()
                    return None
                current_version = int(locked["version"])
                if expected_version is not None and current_version != expected_version:
                    conn.rollback()
                    raise WorkspaceVersionConflictError(current_version, expected_version)
                # Snapshot CURRENT state before overwriting
                cur.execute(snapshot_statement, (str(_uuid.uuid4()), document_id, user_id))
                cur.execute(
                    update_statement,
                    (content_text, size_bytes, sha256, source, document_id, user_id),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_document_row(row) if row else None

    def count_documents(self, user_id: str) -> int:
        """Total number of workspace documents for a user (for accurate list pagination)."""
        statement = "SELECT COUNT(*) AS n FROM workspace_documents WHERE user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id,))
                row = cur.fetchone()
        return int(row["n"]) if row else 0

    def save_document_version_snapshot(
        self,
        *,
        document_id: str,
        user_id: str,
        version: int,
        content_text: str,
        size_bytes: int,
        sha256: str,
        source: str = "user",
    ) -> None:
        import uuid as _uuid

        statement = """
            INSERT INTO workspace_document_versions
                (id, document_id, user_id, version, content_text, size_bytes, sha256, source, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (document_id, version) DO NOTHING;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (str(_uuid.uuid4()), document_id, user_id, version, content_text, size_bytes, sha256, source))
            conn.commit()

    def list_document_versions(self, *, document_id: str, user_id: str, limit: int = 50) -> list[Dict[str, Any]]:
        statement = """
            SELECT v.id, v.document_id, v.version, v.size_bytes, v.sha256, v.source, v.created_at
            FROM workspace_document_versions v
            JOIN workspace_documents d ON d.id = v.document_id
            WHERE v.document_id = %s AND d.user_id = %s
            ORDER BY v.version DESC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (document_id, user_id, limit))
                rows = cur.fetchall()
        return [_normalize_version_row(dict(row)) for row in rows]

    def get_document_version(self, *, document_id: str, user_id: str, version: int) -> Optional[Dict[str, Any]]:
        statement = """
            SELECT v.id, v.document_id, v.version, v.content_text, v.size_bytes, v.sha256, v.source, v.created_at
            FROM workspace_document_versions v
            JOIN workspace_documents d ON d.id = v.document_id
            WHERE v.document_id = %s AND d.user_id = %s AND v.version = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (document_id, user_id, version))
                row = cur.fetchone()
        return _normalize_version_row(dict(row)) if row else None

    def rename_document(self, document_id: str, user_id: str, filename: str) -> Optional[Dict[str, Any]]:
        statement = """
            UPDATE workspace_documents
            SET filename = %s,
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version, last_source,
                      created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (filename, document_id, user_id))
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_document_row(row) if row else None

    def delete_document(self, document_id: str, user_id: str) -> bool:
        statement = "DELETE FROM workspace_documents WHERE id = %s AND user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (document_id, user_id))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def delete_all_documents(self, user_id: str) -> int:
        """Delete all workspace documents for a user. Returns the number of rows deleted."""
        statement = "DELETE FROM workspace_documents WHERE user_id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (user_id,))
                count = cur.rowcount
            conn.commit()
        return count

    def attach_document_reference(
        self,
        conversation_id: str,
        message_id: Optional[int],
        document_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO chat_document_refs (
                conversation_id, message_id, document_id, user_id
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id, conversation_id, message_id, document_id, user_id, attached_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, message_id, document_id, user_id))
                row = cur.fetchone()
            conn.commit()
        return {
            "id": row.get("id"),
            "conversation_id": row.get("conversation_id"),
            "message_id": row.get("message_id"),
            "document_id": row.get("document_id"),
            "user_id": row.get("user_id"),
            "attached_at": row.get("attached_at").isoformat() if row.get("attached_at") else None,
        }


def _normalize_conversation_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}
    md = row.get("metadata") or {}
    if isinstance(md, str):
        md = json.loads(md)
    created = row.get("created_at")
    updated = row.get("updated_at")
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id"),
        "title": row.get("title"),
        "language": row.get("language"),
        "profile_name": row.get("profile_name"),
        "pinned": row.get("pinned", False),
        "metadata": md,
        "last_message": row.get("last_message"),
        "message_count": row.get("message_count"),
        "match_snippet": row.get("match_snippet"),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
        "updated_at": updated.isoformat() if isinstance(updated, datetime) else str(updated) if updated else None,
    }


def _normalize_message_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}
    tools = row.get("tools_used")
    if isinstance(tools, str):
        tools = json.loads(tools)
    md = row.get("metadata") or {}
    if isinstance(md, str):
        md = json.loads(md)
    created = row.get("created_at")
    return {
        "id": row.get("id"),
        "conversation_id": row.get("conversation_id"),
        "role": row.get("role"),
        "content": row.get("content"),
        "tokens_used": row.get("tokens_used"),
        "model_name": row.get("model_name"),
        "response_time_ms": row.get("response_time_ms"),
        "tools_used": tools,
        "feedback": row.get("feedback"),
        "metadata": md,
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
    }


def _normalize_attachment_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    created = row.get("created_at")
    expires = row.get("expires_at")

    return {
        "id": row.get("id"),
        "conversation_id": row.get("conversation_id"),
        "user_id": row.get("user_id"),
        "original_name": row.get("original_name"),
        "stored_path": row.get("stored_path"),
        "mime_type": row.get("mime_type"),
        "size_bytes": row.get("size_bytes"),
        "sha256": row.get("sha256"),
        "extracted_text": row.get("extracted_text"),
        "status": row.get("status", "active"),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
        "expires_at": expires.isoformat() if isinstance(expires, datetime) else str(expires) if expires else None,
    }


def _normalize_workspace_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    created = row.get("created_at")
    updated = row.get("updated_at")
    last_activity = row.get("last_activity_at")
    expires = row.get("expires_at")

    return {
        "conversation_id": row.get("conversation_id"),
        "user_id": row.get("user_id"),
        "root_path": row.get("root_path"),
        "status": row.get("status", "active"),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
        "updated_at": updated.isoformat() if isinstance(updated, datetime) else str(updated) if updated else None,
        "last_activity_at": (
            last_activity.isoformat()
            if isinstance(last_activity, datetime)
            else str(last_activity)
            if last_activity
            else None
        ),
        "expires_at": expires.isoformat() if isinstance(expires, datetime) else str(expires) if expires else None,
    }


def _normalize_workspace_operation_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    created = row.get("created_at")
    return {
        "id": row.get("id"),
        "conversation_id": row.get("conversation_id"),
        "user_id": row.get("user_id"),
        "attachment_id": row.get("attachment_id"),
        "operation": row.get("operation"),
        "result": row.get("result"),
        "reason": row.get("reason"),
        "source_path": row.get("source_path"),
        "target_path": row.get("target_path"),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
    }


def _normalize_version_row(row: Dict[str, Any]) -> Dict[str, Any]:
    created = row.get("created_at")
    row["created_at"] = created.isoformat() if isinstance(created, datetime) else str(created) if created else None
    return row


def _normalize_workspace_document_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}

    created = row.get("created_at")
    updated = row.get("updated_at")
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id"),
        "filename": row.get("filename"),
        "stored_path": row.get("stored_path"),
        "mime_type": row.get("mime_type"),
        "size_bytes": row.get("size_bytes"),
        "sha256": row.get("sha256"),
        "content_text": row.get("content_text"),
        "version": row.get("version", 1),
        "last_source": row.get("last_source", "user"),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None,
        "updated_at": updated.isoformat() if isinstance(updated, datetime) else str(updated) if updated else None,
    }


class AnalyticsStore:
    """Manages analytics and metrics data for historical trends."""
    
    def __init__(self, db_pool: DatabasePool) -> None:
        self._db_pool = db_pool
    
    def log_event(
        self,
        user_id: Optional[str],
        event_type: str,
        model_name: Optional[str] = None,
        tokens_used: Optional[int] = None,
        request_duration_seconds: Optional[float] = None,
        status: str = "success",
        profile_name: Optional[str] = None,
    ) -> bool:
        """Log an analytics event."""
        statement = """
            INSERT INTO analytics_events (
                user_id, event_type, model_name, tokens_used, 
                request_duration_seconds, status, profile_name
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        try:
            with self._db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        statement,
                        (
                            user_id,
                            event_type,
                            model_name,
                            tokens_used,
                            request_duration_seconds,
                            status,
                            profile_name,
                        ),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log analytics event: {e}")
            return False
    
    def get_usage_trends(self, days: int = 30) -> list[Dict[str, Any]]:
        """Get daily usage trends over the past N days."""
        statement = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as request_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM analytics_events
            WHERE created_at >= NOW() - INTERVAL '%s days'
            AND event_type = 'chat_request'
            GROUP BY DATE(created_at)
            ORDER BY date ASC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (days,))
                rows = cur.fetchall()
        
        return [
            {
                "date": row["date"].isoformat(),
                "requests": row["request_count"],
                "users": row["unique_users"],
            }
            for row in rows
        ]
    
    def get_token_consumption(self, days: int = 30) -> list[Dict[str, Any]]:
        """Get daily token consumption trends."""
        statement = """
            SELECT 
                DATE(created_at) as date,
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                AVG(COALESCE(tokens_used, 0)) as avg_tokens,
                COUNT(*) as request_count
            FROM analytics_events
            WHERE created_at >= NOW() - INTERVAL '%s days'
            AND event_type = 'chat_request'
            AND tokens_used IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY date ASC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (days,))
                rows = cur.fetchall()
        
        return [
            {
                "date": row["date"].isoformat(),
                "total_tokens": int(row["total_tokens"] or 0),
                "avg_tokens": float(row["avg_tokens"] or 0),
                "requests": row["request_count"],
            }
            for row in rows
        ]
    
    def get_latency_analysis(self, days: int = 30) -> list[Dict[str, Any]]:
        """Get daily latency analysis."""
        statement = """
            SELECT 
                DATE(created_at) as date,
                AVG(request_duration_seconds) as avg_latency,
                MIN(request_duration_seconds) as min_latency,
                MAX(request_duration_seconds) as max_latency,
                COUNT(*) as request_count
            FROM analytics_events
            WHERE created_at >= NOW() - INTERVAL '%s days'
            AND event_type = 'chat_request'
            AND request_duration_seconds IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY date ASC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (days,))
                rows = cur.fetchall()
        
        return [
            {
                "date": row["date"].isoformat(),
                "avg_latency_ms": float((row["avg_latency"] or 0) * 1000),
                "min_latency_ms": float((row["min_latency"] or 0) * 1000),
                "max_latency_ms": float((row["max_latency"] or 0) * 1000),
                "requests": row["request_count"],
            }
            for row in rows
        ]
    
    def get_cost_projection(self, cost_per_token: float = 0.00002, days: int = 30) -> Dict[str, Any]:
        """Calculate cost projection based on token usage."""
        statement = """
            SELECT 
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                COUNT(*) as request_count,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(COALESCE(tokens_used, 0)) as avg_tokens_per_request
            FROM analytics_events
            WHERE created_at >= NOW() - INTERVAL '%s days'
            AND event_type = 'chat_request';
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (days,))
                row = cur.fetchone()

        if row is None:
            return {
                "period_days": days,
                "total_tokens": 0,
                "request_count": 0,
                "unique_users": 0,
                "avg_tokens_per_request": 0.0,
                "cost_per_token": cost_per_token,
                "current_cost": 0.0,
                "daily_avg_cost": 0.0,
                "monthly_projection": 0.0,
            }
        
        total_tokens = int(row["total_tokens"] or 0)
        request_count = row["request_count"] or 0
        unique_users = row["unique_users"] or 0
        avg_tokens = float(row["avg_tokens_per_request"] or 0)
        
        current_cost = total_tokens * cost_per_token
        daily_avg_cost = (current_cost / days) if days > 0 else 0
        monthly_projection = daily_avg_cost * 30
        
        return {
            "period_days": days,
            "total_tokens": total_tokens,
            "request_count": request_count,
            "unique_users": unique_users,
            "avg_tokens_per_request": avg_tokens,
            "cost_per_token": cost_per_token,
            "current_cost": round(current_cost, 4),
            "daily_avg_cost": round(daily_avg_cost, 4),
            "monthly_projection": round(monthly_projection, 4),
        }

    def get_message_quality(self, days: int = 30) -> list[Dict[str, Any]]:
        """Get daily message quality (thumbs up/down) trends over the past N days.

        Queries the messages table directly for assistant messages that have
        received explicit thumbs feedback from the user.
        """
        statement = """
            SELECT
                DATE(m.created_at) AS date,
                COUNT(*) FILTER (WHERE m.feedback = 'up')   AS up_count,
                COUNT(*) FILTER (WHERE m.feedback = 'down') AS down_count,
                COUNT(*) FILTER (WHERE m.feedback IS NOT NULL) AS total_feedback,
                COUNT(*) AS total_assistant_messages
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.role = 'assistant'
              AND m.created_at >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(m.created_at)
            ORDER BY date ASC;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (days,))
                rows = cur.fetchall()
        return [
            {
                "date": row["date"].isoformat(),
                "up_count": int(row["up_count"] or 0),
                "down_count": int(row["down_count"] or 0),
                "total_feedback": int(row["total_feedback"] or 0),
                "total_assistant_messages": int(row["total_assistant_messages"] or 0),
                "net_score": int(row["up_count"] or 0) - int(row["down_count"] or 0),
            }
            for row in rows
        ]
