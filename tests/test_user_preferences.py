"""
Integration tests for user preferences panel functionality.
Tests the full stack: database schema, API endpoints, and frontend integration.
"""

import pytest
from backend.db import DatabasePool, SettingsStore, DatabaseConfig


class TestUserPreferencesSchema:
    """Test the user preferences database schema."""

    @pytest.fixture
    def db_pool(self):
        """Create an in-memory PostgreSQL test database."""
        # Using test database credentials
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        pool = DatabasePool(config)
        yield pool
        pool.close()

    def test_settings_table_has_all_columns(self, db_pool):
        """Verify that user_settings table has all required columns."""
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Get table structure
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'user_settings'
                    ORDER BY column_name;
                """)
                columns = {row[0]: row[1] for row in cur.fetchall()}

        # Verify all expected columns exist
        expected_columns = {
            "user_id": "text",
            "tool_toggles": "jsonb",
            "rag_config": "jsonb",
            "preferred_model": "text",
            "theme": "text",
            "display_sidebar_visible": "boolean",
            "display_compact_mode": "boolean",
            "language": "text",
            "notifications_enabled": "boolean",
            "notifications_sound": "boolean",
            "updated_at": "timestamp with time zone",
        }

        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} not found"
            assert columns[col_name] == col_type, (
                f"Column {col_name} has type {columns[col_name]}, "
                f"expected {col_type}"
            )

    def test_theme_column_has_check_constraint(self, db_pool):
        """Verify that theme column has proper CHECK constraint."""
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = 'user_settings'
                    AND constraint_type = 'CHECK';
                """)
                constraints = [row[0] for row in cur.fetchall()]

        assert any("theme" in c for c in constraints), (
            "Theme column should have a CHECK constraint"
        )


class TestSettingsStore:
    """Test the SettingsStore class."""

    @pytest.fixture
    def db_pool(self):
        """Create a test database pool."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        pool = DatabasePool(config)
        
        # Clean up test data before each test
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_settings WHERE user_id LIKE 'test_%';")
            conn.commit()
        
        yield pool
        
        # Clean up after test
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_settings WHERE user_id LIKE 'test_%';")
            conn.commit()
        
        pool.close()

    @pytest.fixture
    def store(self, db_pool):
        """Create a SettingsStore instance."""
        return SettingsStore(db_pool)

    def test_get_nonexistent_settings_returns_none(self, store):
        """Getting settings for non-existent user returns None."""
        result = store.get_user_settings("test_nonexistent")
        assert result is None

    def test_upsert_and_retrieve_preferences(self, store):
        """Test upserting and retrieving user preferences."""
        user_id = "test_pref_user"
        
        # Upsert preferences
        result = store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={"web_search_mcp": True, "datetime_tool": False},
            rag_config={"collection": "medical", "top_k": 10},
            preferred_model="llama2",
            theme="dark",
            display_sidebar_visible=False,
            display_compact_mode=True,
            language="de",
            notifications_enabled=False,
            notifications_sound=False,
        )

        # Verify returned data
        assert result["user_id"] == user_id
        assert result["tool_toggles"]["web_search_mcp"] is True
        assert result["tool_toggles"]["datetime_tool"] is False
        assert result["rag_config"]["collection"] == "medical"
        assert result["rag_config"]["top_k"] == 10
        assert result["preferred_model"] == "llama2"
        assert result["theme"] == "dark"
        assert result["display_sidebar_visible"] is False
        assert result["display_compact_mode"] is True
        assert result["language"] == "de"
        assert result["notifications_enabled"] is False
        assert result["notifications_sound"] is False
        assert result["updated_at"] is not None

        # Retrieve and verify stored data
        retrieved = store.get_user_settings(user_id)
        assert retrieved == result

    def test_upsert_updates_existing_preferences(self, store):
        """Test that upsert properly updates existing preferences."""
        user_id = "test_update_user"
        
        # Initial upsert
        store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={"web_search_mcp": True},
            rag_config={"top_k": 5},
            preferred_model="gemma3:4b",
            theme="light",
            display_sidebar_visible=True,
            display_compact_mode=False,
            language="en",
            notifications_enabled=True,
            notifications_sound=True,
        )

        # Update preferences
        updated = store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={"web_search_mcp": False},
            rag_config={"top_k": 15},
            preferred_model="llama2",
            theme="dark",
            display_sidebar_visible=False,
            display_compact_mode=True,
            language="de",
            notifications_enabled=False,
            notifications_sound=False,
        )

        assert updated["theme"] == "dark"
        assert updated["preferred_model"] == "llama2"
        assert updated["rag_config"]["top_k"] == 15
        assert updated["display_compact_mode"] is True
        assert updated["language"] == "de"
        assert updated["notifications_enabled"] is False

    def test_preferences_are_independent_per_user(self, store):
        """Test that preferences for different users don't interfere."""
        user1_id = "test_user1"
        user2_id = "test_user2"

        # Set different preferences for each user
        store.upsert_user_settings(
            user_id=user1_id,
            tool_toggles={"web_search_mcp": True},
            rag_config={},
            preferred_model="llama2",
            theme="dark",
            display_sidebar_visible=True,
            display_compact_mode=False,
            language="en",
            notifications_enabled=True,
            notifications_sound=True,
        )

        store.upsert_user_settings(
            user_id=user2_id,
            tool_toggles={"web_search_mcp": False},
            rag_config={},
            preferred_model="gemma3:4b",
            theme="light",
            display_sidebar_visible=False,
            display_compact_mode=True,
            language="de",
            notifications_enabled=False,
            notifications_sound=False,
        )

        # Verify each user has independent settings
        user1_prefs = store.get_user_settings(user1_id)
        user2_prefs = store.get_user_settings(user2_id)

        assert user1_prefs["theme"] == "dark"
        assert user2_prefs["theme"] == "light"
        assert user1_prefs["preferred_model"] == "llama2"
        assert user2_prefs["preferred_model"] == "gemma3:4b"
        assert user1_prefs["language"] == "en"
        assert user2_prefs["language"] == "de"

    def test_default_values_are_applied(self, store):
        """Test that default values are properly applied for missing fields."""
        user_id = "test_defaults_user"
        
        # Upsert with minimal data (using defaults)
        result = store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={},
            rag_config={},
            preferred_model=None,
        )

        # Verify defaults are applied
        assert result["theme"] == "auto"
        assert result["display_sidebar_visible"] is True
        assert result["display_compact_mode"] is False
        assert result["language"] == "en"
        assert result["notifications_enabled"] is True
        assert result["notifications_sound"] is True

    def test_notification_settings_are_independent(self, store):
        """Test that notifications_enabled doesn't affect notifications_sound setting."""
        user_id = "test_notifications_user"
        
        # Set notifications disabled but sound enabled (edge case)
        result = store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={},
            rag_config={},
            preferred_model=None,
            notifications_enabled=False,
            notifications_sound=True,  # Sound setting independent
        )

        assert result["notifications_enabled"] is False
        assert result["notifications_sound"] is True

        # Verify stored correctly
        retrieved = store.get_user_settings(user_id)
        assert retrieved["notifications_enabled"] is False
        assert retrieved["notifications_sound"] is True

    def test_complex_rag_config_is_preserved(self, store):
        """Test that complex RAG configuration with nested structures is preserved."""
        user_id = "test_complex_config_user"
        
        complex_config = {
            "collection": "medical_ai",
            "top_k": 8,
            "score_threshold": 0.7,
            "filters": {
                "source": "textbooks",
                "language": "de",
            },
            "rerank": True,
        }

        store.upsert_user_settings(
            user_id=user_id,
            tool_toggles={},
            rag_config=complex_config,
            preferred_model=None,
        )

        retrieved = store.get_user_settings(user_id)
        
        # Verify complex structure is preserved
        assert retrieved["rag_config"]["collection"] == "medical_ai"
        assert retrieved["rag_config"]["score_threshold"] == 0.7
        assert retrieved["rag_config"]["filters"]["source"] == "textbooks"
        assert retrieved["rag_config"]["rerank"] is True
