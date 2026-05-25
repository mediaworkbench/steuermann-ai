"""Tests for curated reprobe triggers on settings/model changes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

from backend.llm_capability_probe import LLMCapabilityProbeResult


class _FakeSettingsStore:
    """In-memory settings store — prevents writing to the shared dev DB during tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get_user_settings(self, user_id: str) -> dict | None:
        return dict(self._store[user_id]) if user_id in self._store else None

    def upsert_user_settings(self, **kwargs) -> dict:
        record = {
            "user_id": kwargs["user_id"],
            "tool_toggles": kwargs["tool_toggles"],
            "rag_config": kwargs["rag_config"],
            "analytics_preferences": kwargs["analytics_preferences"],
            "preferred_model": kwargs["preferred_model"],
            "preferred_models": kwargs["preferred_models"],
            "theme": kwargs["theme"],
            "language": kwargs["language"],
            "updated_at": None,
        }
        self._store[kwargs["user_id"]] = record
        return dict(record)


class _FakeProbeStore:
    """In-memory probe store — prevents writing fake probe results to the shared dev DB."""

    def __init__(self) -> None:
        self._results: list = []

    def get_all_results(self) -> list:
        return list(self._results)

    def upsert_results(self, results: list) -> None:
        self._results.extend(results)

    def get_results_for_model(self, model_name: str) -> list:
        return [r for r in self._results if getattr(r, "model_name", None) == model_name]


@pytest.fixture
def settings_store():
    return _FakeSettingsStore()


@pytest.fixture
def probe_store():
    return _FakeProbeStore()


class FakeLLMCapabilityProbeRunner:
    """Mock probe runner for testing."""

    def __init__(self, profile_id=None, core_config=None):
        self.profile_id = profile_id or "test_profile"
        self.core_config = core_config
        self.reprobe_calls = []

    def run(self):
        """Return fake probe results (full reprobe)."""
        self.reprobe_calls.append(("run", None))
        return [
            LLMCapabilityProbeResult(
                profile_id=self.profile_id,
                provider_id="test_provider",
                model_name="test-model",
                configured_tool_calling_mode="native",
                supports_bind_tools=True,
                supports_tool_schema=True,
                capability_mismatch=False,
                status="ok",
            )
        ]

    def reprobe_for_model(self, model_name: str):
        """Return fake probe result for the given model (curated reprobe)."""
        self.reprobe_calls.append(("reprobe_for_model", model_name))
        return [
            LLMCapabilityProbeResult(
                profile_id=self.profile_id,
                provider_id="test_provider",
                model_name=model_name,
                configured_tool_calling_mode="native",
                supports_bind_tools=True,
                supports_tool_schema=True,
                capability_mismatch=False,
                status="ok",
            )
        ]

    def reprobe_model(self, provider_id: str, model_name: str):
        """Return fake reprobe result for specific provider+model."""
        self.reprobe_calls.append(("reprobe_model", provider_id, model_name))
        return LLMCapabilityProbeResult(
            profile_id=self.profile_id,
            provider_id=provider_id,
            model_name=model_name,
            configured_tool_calling_mode="native",
            supports_bind_tools=True,
            supports_tool_schema=True,
            capability_mismatch=False,
            status="ok",
        )


@pytest.fixture
def client_with_mocks(probe_store, settings_store):
    """Create FastAPI test client with mocked probe runner."""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Wire test stores - minimal setup
    app.state.probe_store = probe_store
    app.state.settings_store = settings_store
    app.state.llm_capability_probe_store = probe_store
    
    # Mock require_api_access dependency
    def mock_require_api_access():
        return None
    
    # Import and register the router, but override dependencies
    from backend.routers.settings import router
    from backend.single_user import require_api_access
    
    app.dependency_overrides[require_api_access] = mock_require_api_access
    app.include_router(router)
    
    # Mock the probe runner
    fake_runner = FakeLLMCapabilityProbeRunner(profile_id="starter")
    
    with patch("backend.routers.settings.LLMCapabilityProbeRunner", return_value=fake_runner):
        with patch("backend.routers.settings.get_active_profile_id", return_value="starter"):
            with patch("backend.routers.settings._validate_chat_preference", new=AsyncMock(side_effect=lambda model: (model, None))):
                yield TestClient(app), fake_runner


class TestReprobeTriggersOnSettingsChange:
    """Test reprobe triggers when user settings are updated."""
    
    def test_reprobe_triggered_on_model_change(self, client_with_mocks, probe_store):
        """Reprobe should run when preferred_model changes."""
        client, fake_runner = client_with_mocks
        
        # Set initial model
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "old-model",
                "theme": "auto",
                "language": "en",
            },
        )
        assert response.status_code == 200
        
        # Clear reprobe calls from initial setup
        fake_runner.reprobe_calls.clear()
        
        # Change model - should trigger reprobe
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "new-model",
                "theme": "auto",
                "language": "en",
            },
        )
        assert response.status_code == 200
        
        # Verify curated reprobe was called for the specific new model
        assert len(fake_runner.reprobe_calls) > 0
        assert fake_runner.reprobe_calls[0][0] == "reprobe_for_model"
        assert fake_runner.reprobe_calls[0][1] == "new-model"
    
    def test_no_reprobe_when_model_unchanged(self, client_with_mocks, probe_store):
        """No reprobe should run if preferred_model doesn't change."""
        client, fake_runner = client_with_mocks
        
        # Set initial model
        client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "same-model",
                "theme": "auto",
                "language": "en",
            },
        )
        
        fake_runner.reprobe_calls.clear()
        
        # Update settings without changing model
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {"web_search": True},  # Changed, but model didn't
                "rag_config": {"collection": "test", "top_k": 10},
                "analytics_preferences": {},
                "preferred_model": "same-model",  # Same model
                "theme": "dark",  # Changed theme
                "language": "de",  # Changed language
            },
        )
        assert response.status_code == 200
        
        # Verify NO reprobe was called
        assert len(fake_runner.reprobe_calls) == 0
    
    def test_reprobe_from_none_to_model(self, client_with_mocks, probe_store):
        """Reprobe should trigger when user selects a model for the first time."""
        client, fake_runner = client_with_mocks
        
        # Initially no preferred model
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": None,
                "theme": "auto",
                "language": "en",
            },
        )
        assert response.status_code == 200
        
        fake_runner.reprobe_calls.clear()
        
        # Now select a model - should trigger reprobe
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "first-model",
                "theme": "auto",
                "language": "en",
            },
        )
        assert response.status_code == 200
        
        # Verify reprobe was called
        assert len(fake_runner.reprobe_calls) > 0
    
    def test_reprobe_failure_doesnt_block_settings_update(self, client_with_mocks):
        """Settings update should succeed even if reprobe fails."""
        client, fake_runner = client_with_mocks
        
        # Make curated reprobe fail
        fake_runner.reprobe_for_model = MagicMock(side_effect=Exception("Probe failed"))
        
        # Update settings with model change - should still succeed
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "failed-probe-model",
                "theme": "auto",
                "language": "en",
            },
        )
        
        # Settings update should still succeed despite probe failure
        assert response.status_code == 200
        assert response.json()["preferred_model"] == "failed-probe-model"
    
    def test_reprobe_none_to_none_no_trigger(self, client_with_mocks):
        """No reprobe if both old and new model are None."""
        client, fake_runner = client_with_mocks
        
        # Start with no model
        client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": None,
                "theme": "auto",
                "language": "en",
            },
        )
        
        fake_runner.reprobe_calls.clear()
        
        # Update other settings, still no model
        response = client.post(
            "/api/settings/user/test_user",
            json={
                "tool_toggles": {"web_search": True},
                "rag_config": {"collection": "test", "top_k": 10},
                "analytics_preferences": {},
                "preferred_model": None,  # Still None
                "theme": "dark",
                "language": "de",
            },
        )
        
        assert response.status_code == 200
        # No reprobe should have been triggered
        assert len(fake_runner.reprobe_calls) == 0


class TestReprobeWithMultipleUsers:
    """Test reprobe behavior with multiple concurrent user settings updates."""
    
    def test_reprobe_triggered_independently_per_user(self, client_with_mocks):
        """Reprobe for each user is independent."""
        client, fake_runner = client_with_mocks
        
        # User 1 selects a model - should trigger reprobe
        response1 = client.post(
            "/api/settings/user/user_1",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "user1-model",
                "theme": "auto",
                "language": "en",
            },
        )
        assert response1.status_code == 200
        call_count_after_user1 = len(fake_runner.reprobe_calls)
        
        # User 2 selects a model - should trigger reprobe again
        response2 = client.post(
            "/api/settings/user/user_2",
            json={
                "tool_toggles": {},
                "rag_config": {"collection": "", "top_k": 5},
                "analytics_preferences": {},
                "preferred_model": "user2-model",
                "theme": "auto",
                "language": "en",
            },
        )
        assert response2.status_code == 200
        
        # Verify reprobe was called for both users
        assert len(fake_runner.reprobe_calls) > call_count_after_user1
