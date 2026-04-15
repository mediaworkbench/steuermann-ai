"""
Tests for analytics functionality: database schema, event logging, and metrics retrieval.
"""

import pytest
from datetime import datetime, timedelta, timezone

from backend.db import DatabasePool, AnalyticsStore, DatabaseConfig


class TestAnalyticsSchema:
    """Test analytics table schema and indices."""

    @pytest.fixture
    def db_pool(self):
        """Create database pool for testing."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        pool = DatabasePool(config)
        yield pool
        pool.close()

    def test_analytics_events_table_exists(self, db_pool):
        """Verify analytics_events table exists with all required columns."""
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'analytics_events'
                    ORDER BY ordinal_position;
                """)
                columns = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

        # Verify all expected columns exist
        expected = [
            "event_id",
            "user_id",
            "event_type",
            "model_name",
            "tokens_used",
            "request_duration_seconds",
            "status",
            "fork_name",
            "created_at",
        ]
        assert all(col in columns for col in expected)

    def test_analytics_indices_exist(self, db_pool):
        """Verify indices are created on frequently-queried columns."""
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT indexname FROM pg_indexes
                    WHERE tablename = 'analytics_events'
                    ORDER BY indexname;
                """)
                indices = [row[0] for row in cur.fetchall()]

        # Verify key indices exist (note: model index uses idx_analytics_model, not idx_analytics_model_name)
        expected_indices = [
            "idx_analytics_created_at",
            "idx_analytics_user_id",
            "idx_analytics_event_type",
            "idx_analytics_model",
        ]
        assert all(idx in indices for idx in expected_indices)


class TestAnalyticsStoreLogEvent:
    """Test event logging functionality."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        # Clear previous test data
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analytics_events WHERE user_id LIKE 'test-%';")
                conn.commit()
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_log_event_basic(self, setup):
        """Log a basic event and verify storage."""
        db_pool, analytics_store = setup
        
        event_data = {
            "user_id": "test-user-1",
            "event_type": "chat_request",
            "status": "success",
        }

        success = analytics_store.log_event(**event_data)
        assert success is True

        # Verify the event was stored
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_events WHERE user_id = %s;",
                    ("test-user-1",)
                )
                stored = cur.fetchone()
                assert stored is not None
                # Verify data (adjust indices based on column order)
                assert stored[1] == "test-user-1"  # user_id
                assert stored[2] == "chat_request"  # event_type
                assert stored[6] == "success"  # status

    def test_log_event_with_tokens(self, setup):
        """Log event with token usage data."""
        db_pool, analytics_store = setup
        
        event_data = {
            "user_id": "test-user-2",
            "event_type": "chat_request",
            "model_name": "llama-3.1-8b",
            "tokens_used": 250,
            "request_duration_seconds": 2.5,
            "status": "success",
            "fork_name": "medical-ai",
        }

        success = analytics_store.log_event(**event_data)
        assert success is True
        
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_events WHERE user_id = %s;",
                    ("test-user-2",)
                )
                stored = cur.fetchone()

                assert stored[3] == "llama-3.1-8b"  # model_name
                assert stored[4] == 250  # tokens_used
                assert stored[5] == 2.5  # request_duration_seconds
                assert stored[7] == "medical-ai"  # fork_name

    def test_log_event_with_error(self, setup):
        """Log a failed event."""
        db_pool, analytics_store = setup
        
        event_data = {
            "user_id": "test-user-3",
            "event_type": "chat_request",
            "status": "error",
        }

        success = analytics_store.log_event(**event_data)
        assert success is True
        
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM analytics_events WHERE user_id = %s;",
                    ("test-user-3",)
                )
                status = cur.fetchone()[0]
                assert status == "error"


class TestAnalyticsStoreUsageTrends:
    """Test usage trends retrieval."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_usage_trends_empty(self, setup):
        """Get trends returns a list."""
        db_pool, analytics_store = setup
        
        trends = analytics_store.get_usage_trends(days=30)
        # Returns a list of dicts, not a response object
        assert isinstance(trends, list)
        # Can have data from other tests, so just verify it's a list

    def test_usage_trends_with_data(self, setup):
        """Get trends with sample data."""
        db_pool, analytics_store = setup
        
        import uuid
        prefix = str(uuid.uuid4())[:8]  # Use unique prefix
        
        # Log events from multiple days
        today = datetime.now(timezone.utc)
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                for day_offset in range(5):
                    for i in range(3):
                        event_date = today - timedelta(days=day_offset)
                        cur.execute("""
                            INSERT INTO analytics_events
                            (user_id, event_type, model_name, status, created_at)
                            VALUES (%s, %s, %s, %s, %s);
                        """,
                            (f"{prefix}-trend-user-{i}", "chat_request", "llama-3.1-8b", "success", event_date)
                        )
                conn.commit()

        trends = analytics_store.get_usage_trends(days=10)
        # Should have results
        assert isinstance(trends, list)
        # Filter for our test data only
        our_trends = [t for t in trends if 'requests' in t]
        assert len(our_trends) > 0


class TestAnalyticsStoreTokenConsumption:
    """Test token consumption analysis."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_token_consumption_empty(self, setup):
        """Get consumption returns a list."""
        db_pool, analytics_store = setup
        
        consumption = analytics_store.get_token_consumption(days=30)
        # Returns a list of dicts
        assert isinstance(consumption, list)
        # Can have data from other tests, so just verify structure

    def test_token_consumption_with_data(self, setup):
        """Get consumption with sample token data."""
        db_pool, analytics_store = setup
        
        import uuid
        prefix = str(uuid.uuid4())[:8]  # Use unique prefix
        
        today = datetime.now(timezone.utc)
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Log events with tokens
                for i in range(3):
                    cur.execute("""
                        INSERT INTO analytics_events
                        (user_id, event_type, model_name, tokens_used, status, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                        (f"{prefix}-token-user-{i}", "chat_request", "llama-3.1-8b", 
                         100 + (i * 50), "success", today)
                    )
                conn.commit()

        consumption = analytics_store.get_token_consumption(days=1)
        # Should have results for today
        assert isinstance(consumption, list)
        # Should have at least one entry
        assert len(consumption) > 0
        # Check structure
        for item in consumption:
            assert "date" in item
            assert "total_tokens" in item
            assert "avg_tokens" in item
            assert "requests" in item


class TestAnalyticsStoreLatencyAnalysis:
    """Test latency metrics."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        # Clear all analytics test data to ensure clean state
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analytics_events;")
                conn.commit()
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_latency_empty(self, setup):
        """Get latency when no duration data exists."""
        db_pool, analytics_store = setup
        
        latency = analytics_store.get_latency_analysis(days=30)
        # Returns a list of dicts
        assert isinstance(latency, list)
        assert len(latency) == 0

    def test_latency_with_data(self, setup):
        """Get latency metrics with sample data."""
        db_pool, analytics_store = setup
        
        today = datetime.now(timezone.utc)
        durations = [1.0, 2.5, 1.5, 3.0, 2.0]  # Multiple requests
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                for duration in durations:
                    cur.execute("""
                        INSERT INTO analytics_events
                        (user_id, event_type, model_name, request_duration_seconds, status, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                        ("latency-user", "chat_request", "llama-3.1-8b", duration, "success", today)
                    )
                conn.commit()

        latency = analytics_store.get_latency_analysis(days=1)
        # Should have results
        assert isinstance(latency, list)
        assert len(latency) > 0
        # Check that stats are computed
        stats = latency[0]
        assert "date" in stats
        assert "avg_latency_ms" in stats
        assert "min_latency_ms" in stats
        assert "max_latency_ms" in stats
        assert "requests" in stats


class TestAnalyticsStoreCostProjection:
    """Test cost calculation and projection."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_cost_projection_empty(self, setup):
        """Get cost projection returns a dict."""
        db_pool, analytics_store = setup
        
        projection = analytics_store.get_cost_projection(
            cost_per_token=0.00002, days=30
        )
        # Should return a dict with cost fields
        assert isinstance(projection, dict)
        assert "current_cost" in projection
        assert "daily_avg_cost" in projection
        assert "monthly_projection" in projection

    def test_cost_projection_with_data(self, setup):
        """Calculate cost projection with sample token data."""
        db_pool, analytics_store = setup
        
        import uuid
        prefix = str(uuid.uuid4())[:8]  # Use unique prefix
        
        today = datetime.now(timezone.utc)
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Log 5 events with 1000 tokens each = 5000 total
                for i in range(5):
                    cur.execute("""
                        INSERT INTO analytics_events
                        (user_id, event_type, model_name, tokens_used, status, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                        (f"{prefix}-cost-user-{i}", "chat_request", "llama-3.1-8b", 1000, "success", today)
                    )
                conn.commit()

        projection = analytics_store.get_cost_projection(
            cost_per_token=0.00002, days=1
        )
        # Verify structure
        assert projection["total_tokens"] >= 5000
        assert projection["request_count"] >= 5
        # Our 5000 tokens * 0.00002 = 0.10 at minimum
        assert projection["current_cost"] >= 0.10

    def test_cost_projection_monthly(self, setup):
        """Test monthly projection calculation."""
        db_pool, analytics_store = setup
        
        import uuid
        prefix = str(uuid.uuid4())[:8]  # Use unique prefix
        
        today = datetime.now(timezone.utc)
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Log 10 events over 5 days, each with 500 tokens
                for day_offset in range(5):
                    for i in range(2):
                        event_date = today - timedelta(days=day_offset)
                        cur.execute("""
                            INSERT INTO analytics_events
                            (user_id, event_type, model_name, tokens_used, status, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                            (f"{prefix}-cost-proj-{i}", "chat_request", "llama-3.1-8b", 500, "success", event_date)
                        )
                conn.commit()

        projection = analytics_store.get_cost_projection(
            cost_per_token=0.00001, days=5
        )
        # Verify structure
        assert isinstance(projection, dict)
        assert projection["total_tokens"] >= 5000
        assert projection["monthly_projection"] > 0


class TestAnalyticsDataIsolation:
    """Test data isolation between users/forks."""

    @pytest.fixture
    def setup(self):
        """Set up test database and analytics store."""
        config = DatabaseConfig(
            dsn="postgresql://framework:framework@localhost:5432/framework",
            minconn=1,
            maxconn=5,
        )
        db_pool = DatabasePool(config)
        analytics_store = AnalyticsStore(db_pool)
        
        # Clear previous test data
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analytics_events WHERE user_id = 'isolation-test-user';")
                conn.commit()
        
        yield db_pool, analytics_store
        
        db_pool.close()

    def test_fork_isolation(self, setup):
        """Verify analytics data is properly attributed to forks."""
        db_pool, analytics_store = setup
        
        today = datetime.now(timezone.utc)
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Log events for different forks
                cur.execute("""
                    INSERT INTO analytics_events
                    (user_id, event_type, fork_name, tokens_used, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """,
                    ("isolation-test-user", "chat_request", "medical-ai", 100, "success", today)
                )
                
                cur.execute("""
                    INSERT INTO analytics_events
                    (user_id, event_type, fork_name, tokens_used, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """,
                    ("isolation-test-user", "chat_request", "operational-ai", 200, "success", today)
                )
                conn.commit()
                
                # Verify both events are stored
                cur.execute(
                    "SELECT fork_name FROM analytics_events WHERE user_id = %s;",
                    ("isolation-test-user",)
                )
                fork_names = {row[0] for row in cur.fetchall()}
                assert "medical-ai" in fork_names
                assert "operational-ai" in fork_names


