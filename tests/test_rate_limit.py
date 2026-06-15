"""Tests for the per-user rate-limit key function."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from backend.rate_limit import _get_user_or_ip


def _request(headers: dict[str, str] | None = None, host: str | None = "10.0.0.1"):
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(headers=headers or {}, client=client)


def test_dev_bypass_keys_on_env_user():
    with patch.dict(os.environ, {"AUTH_USERNAME": "patrick"}, clear=True):
        assert _get_user_or_ip(_request()) == "user:patrick"


def test_auth_enabled_keys_on_authenticated_header():
    with patch.dict(os.environ, {"AUTH_ENABLED": "true", "AUTH_USERNAME": "patrick"}, clear=True):
        key = _get_user_or_ip(_request({"x-authenticated-user-id": "u-123"}))
    # Must bucket by the authenticated user, NOT the env single user.
    assert key == "user:u-123"


def test_auth_enabled_distinct_users_get_distinct_buckets():
    with patch.dict(os.environ, {"AUTH_ENABLED": "true"}, clear=True):
        a = _get_user_or_ip(_request({"x-authenticated-user-id": "alice"}))
        b = _get_user_or_ip(_request({"x-authenticated-user-id": "bob"}))
    assert a != b


def test_auth_enabled_without_identity_falls_back_to_ip():
    with patch.dict(os.environ, {"AUTH_ENABLED": "true"}, clear=True):
        assert _get_user_or_ip(_request(host="203.0.113.9")) == "ip:203.0.113.9"
