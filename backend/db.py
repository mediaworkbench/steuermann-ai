from __future__ import annotations

import json
import logging
import os
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
    _ensure_llm_probe_table(db_pool)
    _ensure_admin_tables(db_pool)
    _ensure_analytics_tables(db_pool)
    _ensure_conversation_tables(db_pool)
    _ensure_co_occurrence_tables(db_pool)


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
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    
    # Create index on username and email for faster lookups
    index_statement = """
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(roles_statement)
            cur.execute(users_statement)
            cur.execute(index_statement)
        conn.commit()

    # Legacy compatibility: keep password_hash column if older databases already have it.
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS password_hash TEXT;"
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
            fork_name TEXT,
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
            SELECT u.user_id, u.username, u.email, u.role_id, r.role_name, u.status, u.created_at, u.updated_at
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

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific user by ID."""
        statement = """
            SELECT u.user_id, u.username, u.email, u.role_id, r.role_name, u.status, u.created_at, u.updated_at
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
        
        statement = """
            UPDATE users SET status = %s, updated_at = NOW()
            WHERE user_id = %s
            RETURNING user_id, username, email, role_id, status, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (status, user_id))
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
                   u.role_id, r.role_name, u.status
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.role_id
            WHERE u.username = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (username,))
                row = cur.fetchone()
        return dict(row) if row else None

    def create_user_with_password(
        self,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
        role_name: str = "viewer",
    ) -> Dict[str, Any]:
        """Create a user with a bcrypt password hash and the given role."""
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                # Ensure role exists.
                cur.execute("SELECT role_id FROM roles WHERE role_name = %s;", (role_name,))
                role_row = cur.fetchone()
                if role_row:
                    role_id = role_row["role_id"]
                else:
                    cur.execute(
                        "INSERT INTO roles (role_name) VALUES (%s) "
                        "ON CONFLICT (role_name) DO NOTHING RETURNING role_id;",
                        (role_name,),
                    )
                    inserted = cur.fetchone()
                    if inserted:
                        role_id = inserted["role_id"]
                    else:
                        cur.execute(
                            "SELECT role_id FROM roles WHERE role_name = %s;",
                            (role_name,),
                        )
                        role_row = cur.fetchone()
                        if not role_row:
                            raise ValueError(f"Failed to resolve role_id for role '{role_name}'")
                        role_id = role_row["role_id"]

                cur.execute(
                    """
                    INSERT INTO users (user_id, username, email, password_hash, role_id, status)
                    VALUES (%s, %s, %s, %s, %s, 'active')
                    RETURNING user_id, username, email, role_id, status, created_at, updated_at;
                    """,
                    (user_id, username, email, password_hash, role_id),
                )
                row = cur.fetchone()
            conn.commit()
        return _normalize_user_row(row)


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
            fork_name TEXT,
            archived BOOLEAN NOT NULL DEFAULT false,
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
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
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
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(conversations_statement)
            cur.execute(messages_statement)
            cur.execute(attachments_statement)
            cur.execute(workspaces_statement)
            cur.execute(workspace_operations_statement)
            cur.execute(workspace_documents_statement)
            cur.execute(chat_document_refs_statement)
            for sql in indices_statement.split(';'):
                if sql.strip():
                    cur.execute(sql)
        conn.commit()


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
        fork_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        statement = """
            INSERT INTO conversations (id, user_id, title, language, fork_name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, user_id, title, language, fork_name,
                      archived, pinned, metadata, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id, user_id, title, language, fork_name))
                row = cur.fetchone()
            conn.commit()
        return _normalize_conversation_row(row)

    def list_conversations(
        self,
        user_id: str,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Dict[str, Any]], int]:
        """List conversations for a user, ordered by pinned first then updated_at desc."""
        where = "WHERE user_id = %s"
        params: list[Any] = [user_id]
        if not include_archived:
            where += " AND archived = false"

        count_stmt = f"SELECT COUNT(*) as count FROM conversations {where};"
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(count_stmt, tuple(params))
                count_row = cur.fetchone()
                total = int(count_row["count"]) if count_row else 0

        list_stmt = f"""
            SELECT c.id, c.user_id, c.title, c.language, c.fork_name,
                   c.archived, c.pinned, c.metadata, c.created_at, c.updated_at,
                   (SELECT content FROM messages WHERE conversation_id = c.id
                    ORDER BY created_at DESC LIMIT 1) AS last_message,
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count
            FROM conversations c
            {where}
            ORDER BY c.pinned DESC, c.updated_at DESC
            LIMIT %s OFFSET %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(list_stmt, (*params, limit, offset))
                rows = cur.fetchall()

        return [_normalize_conversation_row(r) for r in rows], total

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a single conversation (without messages)."""
        statement = """
            SELECT id, user_id, title, language, fork_name,
                   archived, pinned, metadata, created_at, updated_at
            FROM conversations WHERE id = %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (conversation_id,))
                row = cur.fetchone()
        return _normalize_conversation_row(row) if row else None

    def update_conversation(
        self, conversation_id: str, **fields: Any
    ) -> Optional[Dict[str, Any]]:
        """Update mutable conversation fields (title, archived, pinned, language, metadata)."""
        allowed = {"title", "archived", "pinned", "language", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_conversation(conversation_id)

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

        statement = f"""
            UPDATE conversations SET {', '.join(set_clauses)}
            WHERE id = %s
            RETURNING id, user_id, title, language, fork_name,
                      archived, pinned, metadata, created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                row = cur.fetchone()
            conn.commit()
        return _normalize_conversation_row(row) if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Hard-delete a conversation and its messages (CASCADE)."""
        statement = "DELETE FROM conversations WHERE id = %s;"
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(statement, (conversation_id,))
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
        self, message_id: int, feedback: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if feedback is not None and feedback not in ("up", "down"):
            raise ValueError(f"Invalid feedback: {feedback}")
        statement = """
            UPDATE messages SET feedback = %s
            WHERE id = %s
            RETURNING id, conversation_id, role, content, tokens_used, model_name,
                      response_time_ms, tools_used, feedback, metadata, created_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (feedback, message_id))
                row = cur.fetchone()
            conn.commit()
        return _normalize_message_row(row) if row else None

    def search_messages(
        self, user_id: str, query: str, limit: int = 50
    ) -> list[Dict[str, Any]]:
        """Full-text search across all messages for a user."""
        statement = """
            SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                   c.title AS conversation_title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = %s
              AND m.content ILIKE %s
            ORDER BY m.created_at DESC
            LIMIT %s;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, (user_id, f"%{query}%", limit))
                rows = cur.fetchall()
        return [
            {
                "message_id": r["id"],
                "conversation_id": r["conversation_id"],
                "conversation_title": r["conversation_title"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
            }
            for r in rows
        ]

    # ── Export ──────────────────────────────────────────────────────────

    def export_conversation(
        self, conversation_id: str, fmt: str = "json"
    ) -> Optional[Dict[str, Any] | str]:
        conv = self.get_conversation(conversation_id)
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
                size_bytes, sha256, content_text, version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version,
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
                   size_bytes, sha256, content_text, version,
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
                   size_bytes, sha256, content_text, version,
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
                   size_bytes, sha256, content_text, version,
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
    ) -> Optional[Dict[str, Any]]:
        if expected_version is None:
            where_version = ""
            params: list[Any] = [content_text, size_bytes, sha256, document_id, user_id]
        else:
            where_version = " AND version = %s"
            params = [content_text, size_bytes, sha256, document_id, user_id, expected_version]

        statement = f"""
            UPDATE workspace_documents
            SET content_text = %s,
                size_bytes = %s,
                sha256 = %s,
                version = version + 1,
                updated_at = NOW()
            WHERE id = %s AND user_id = %s{where_version}
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version,
                      created_at, updated_at;
        """
        with self._db_pool.connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(statement, tuple(params))
                row = cur.fetchone()
            conn.commit()
        return _normalize_workspace_document_row(row) if row else None

    def rename_document(self, document_id: str, user_id: str, filename: str) -> Optional[Dict[str, Any]]:
        statement = """
            UPDATE workspace_documents
            SET filename = %s,
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING id, user_id, filename, stored_path, mime_type,
                      size_bytes, sha256, content_text, version,
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
        "fork_name": row.get("fork_name"),
        "archived": row.get("archived", False),
        "pinned": row.get("pinned", False),
        "metadata": md,
        "last_message": row.get("last_message"),
        "message_count": row.get("message_count"),
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
        fork_name: Optional[str] = None,
    ) -> bool:
        """Log an analytics event."""
        statement = """
            INSERT INTO analytics_events (
                user_id, event_type, model_name, tokens_used, 
                request_duration_seconds, status, fork_name
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
                            fork_name,
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
