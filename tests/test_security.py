"""Security tests for the current single-user adapter and secret validation."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestSingleUserIdentity:
    def test_single_user_id_defaults_to_anonymous(self):
        with patch.dict(os.environ, {}, clear=True):
            from backend.single_user import get_single_user_id
            assert get_single_user_id() == "anonymous"

    def test_single_user_id_uses_env_value(self):
        with patch.dict(os.environ, {"AUTH_USERNAME": "patrick"}, clear=True):
            from backend.single_user import get_single_user_id
            assert get_single_user_id() == "patrick"


class TestResolveCurrentUser:
    def test_dev_bypass_returns_env_admin_when_auth_disabled(self):
        with patch.dict(os.environ, {"AUTH_USERNAME": "patrick"}, clear=True):
            from backend.auth import resolve_current_user
            user = resolve_current_user()
            assert user.user_id == "patrick"
            assert user.username == "patrick"
            assert user.role == "administrator"

    def test_identity_from_trusted_headers_when_auth_enabled(self):
        with patch.dict(os.environ, {"AUTH_ENABLED": "true"}, clear=True):
            from backend.auth import resolve_current_user
            user = resolve_current_user(
                x_authenticated_user_id="u-123",
                x_authenticated_username="alice",
                x_authenticated_role="researcher",
            )
            assert user.user_id == "u-123"
            assert user.username == "alice"
            assert user.role == "researcher"

    def test_rejects_missing_identity_when_auth_enabled(self):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"AUTH_ENABLED": "true"}, clear=True):
            from backend.auth import resolve_current_user
            with pytest.raises(HTTPException) as exc_info:
                resolve_current_user(
                    x_authenticated_user_id=None,
                    x_authenticated_username=None,
                    x_authenticated_role=None,
                )
        assert exc_info.value.status_code == 401

    def test_rejects_invalid_role_when_auth_enabled(self):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"AUTH_ENABLED": "true"}, clear=True):
            from backend.auth import resolve_current_user
            with pytest.raises(HTTPException) as exc_info:
                resolve_current_user(
                    x_authenticated_user_id="u-1",
                    x_authenticated_username="bob",
                    x_authenticated_role="superuser",
                )
        assert exc_info.value.status_code == 401

    def test_role_gates_allow_and_deny(self):
        from fastapi import HTTPException
        from backend.auth import (
            CurrentUser,
            require_admin,
            require_researcher_or_admin,
        )

        admin = CurrentUser(user_id="a", username="a", role="administrator")
        researcher = CurrentUser(user_id="r", username="r", role="researcher")
        basic = CurrentUser(user_id="u", username="u", role="user")

        assert require_admin(admin) is admin
        assert require_researcher_or_admin(researcher) is researcher
        assert require_researcher_or_admin(admin) is admin

        for bad in (researcher, basic):
            with pytest.raises(HTTPException) as exc_info:
                require_admin(bad)
            assert exc_info.value.status_code == 403

        with pytest.raises(HTTPException) as exc_info:
            require_researcher_or_admin(basic)
        assert exc_info.value.status_code == 403


class TestApiAccessGuard:
    def test_access_passes_when_backend_token_is_disabled(self):
        with patch.dict(os.environ, {"CHAT_ACCESS_TOKEN": ""}, clear=True):
            from backend.single_user import require_api_access
            require_api_access(x_chat_token=None, authorization=None)

    def test_access_passes_with_matching_header_token(self):
        with patch.dict(os.environ, {"CHAT_ACCESS_TOKEN": "secret-token"}, clear=True):
            from backend.single_user import require_api_access
            require_api_access(x_chat_token="secret-token", authorization=None)

    def test_access_passes_with_matching_bearer_token(self):
        with patch.dict(os.environ, {"CHAT_ACCESS_TOKEN": "secret-token"}, clear=True):
            from backend.single_user import require_api_access
            require_api_access(x_chat_token=None, authorization="Bearer secret-token")

    def test_access_rejects_missing_token(self):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"CHAT_ACCESS_TOKEN": "secret-token"}, clear=True):
            from backend.single_user import require_api_access
            with pytest.raises(HTTPException) as exc_info:
                require_api_access(x_chat_token=None, authorization=None)
        assert exc_info.value.status_code == 401

    def test_access_rejects_wrong_token(self):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"CHAT_ACCESS_TOKEN": "secret-token"}, clear=True):
            from backend.single_user import require_api_access
            with pytest.raises(HTTPException) as exc_info:
                require_api_access(x_chat_token="wrong-token", authorization=None)
        assert exc_info.value.status_code == 401


# ── Secrets validation ─────────────────────────────────────────────────────

class TestValidateSecrets:
    def test_rejects_insecure_defaults_in_prod(self, caplog):
        import logging
        from backend.secrets import validate_secrets

        with patch.dict(os.environ, {
            "OTEL_ENVIRONMENT": "production",
            "POSTGRES_PASSWORD": "framework",
            "CHAT_ACCESS_TOKEN": "change-me",
        }):
            with caplog.at_level(logging.WARNING, logger="backend.secrets"):
                with pytest.raises(RuntimeError):
                    validate_secrets()
        assert any("CHAT_ACCESS_TOKEN" in r.message for r in caplog.records)

    def test_no_warning_with_strong_secrets(self, caplog):
        import logging
        from backend.secrets import validate_secrets

        with patch.dict(os.environ, {
            "OTEL_ENVIRONMENT": "production",
            "POSTGRES_PASSWORD": "strong-db-password-123",
            "CHAT_ACCESS_TOKEN": "strong-shared-secret-123",
        }):
            with caplog.at_level(logging.WARNING, logger="backend.secrets"):
                validate_secrets()
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_messages

    def test_allows_insecure_defaults_in_development(self, caplog):
        import logging
        from backend.secrets import validate_secrets

        with patch.dict(os.environ, {
            "OTEL_ENVIRONMENT": "development",
            "POSTGRES_PASSWORD": "framework",
            "CHAT_ACCESS_TOKEN": "change-me",
        }):
            with caplog.at_level(logging.DEBUG, logger="backend.secrets"):
                validate_secrets()
        assert any("acceptable in development" in r.message for r in caplog.records)


# ── Input validation (ChatRequest) ─────────────────────────────────────────

class TestChatRequestValidation:
    def test_valid_request(self):
        from backend.routers.chat import ChatRequest

        req = ChatRequest(message="Hello world", user_id="alice", language="en")
        assert req.message == "Hello world"
        assert req.user_id == "alice"

    def test_message_whitespace_stripped(self):
        from backend.routers.chat import ChatRequest

        req = ChatRequest(message="  trimmed  ", user_id="u1")
        assert req.message == "trimmed"

    def test_message_empty_raises(self):
        from pydantic import ValidationError
        from backend.routers.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="   ", user_id="u1")  # strips to empty → min_length=1

    def test_user_id_invalid_chars_raises(self):
        from pydantic import ValidationError
        from backend.routers.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="hi", user_id="user<script>")

    def test_message_too_long_raises(self):
        from pydantic import ValidationError
        from backend.routers.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 32001, user_id="u1")

    def test_invalid_language_raises(self):
        from pydantic import ValidationError
        from backend.routers.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="hi", user_id="u1", language="<script>")

    def test_language_normalised_to_lowercase(self):
        from backend.routers.chat import ChatRequest

        req = ChatRequest(message="hi", user_id="u1", language="EN")
        assert req.language == "en"
