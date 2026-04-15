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

    def test_effective_user_id_ignores_requested_user(self):
        with patch.dict(os.environ, {"AUTH_USERNAME": "patrick"}, clear=True):
            from backend.single_user import get_effective_user_id
            assert get_effective_user_id("someone-else") == "patrick"


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
