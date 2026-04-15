from __future__ import annotations

import os

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_STACK_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set RUN_LIVE_STACK_TESTS=1 to run live stack contract tests.",
)
def test_live_chat_metadata_profile_id_matches_system_profile() -> None:
    base_url = os.getenv("LIVE_API_BASE_URL", "http://localhost:8001").rstrip("/")
    chat_token = os.getenv("LIVE_CHAT_ACCESS_TOKEN", "").strip()

    headers = {
        "x-chat-token": chat_token,
        "content-type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        system_response = client.get(f"{base_url}/api/system-config", headers={"x-chat-token": chat_token})
        assert system_response.status_code == 200, system_response.text

        system_payload = system_response.json()
        profile = system_payload.get("profile") or {}
        expected_profile_id = profile.get("id")
        assert expected_profile_id, system_payload

        chat_response = client.post(
            f"{base_url}/api/chat",
            headers=headers,
            json={
                "message": "Reply with one short sentence.",
                "user_id": "anonymous",
                "language": "en",
            },
        )
        assert chat_response.status_code == 200, chat_response.text

        chat_payload = chat_response.json()
        metadata = chat_payload.get("metadata") or {}
        assert metadata.get("profile_id") == expected_profile_id, chat_payload
