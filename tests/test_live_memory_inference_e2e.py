from __future__ import annotations

import os
import time
from datetime import datetime, UTC

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_STACK_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set RUN_LIVE_STACK_TESTS=1 to run live stack memory E2E tests.",
)
def test_live_memory_short_and_long_term_inference() -> None:
    base_url = os.getenv("LIVE_API_BASE_URL", "http://localhost:8001").rstrip("/")
    chat_token = os.getenv("LIVE_CHAT_ACCESS_TOKEN", "").strip()
    user_id = os.getenv("LIVE_USER_ID", "anonymous").strip() or "anonymous"

    if not chat_token:
        pytest.skip("LIVE_CHAT_ACCESS_TOKEN is required for live memory E2E tests")

    headers = {
        "x-chat-token": chat_token,
        "content-type": "application/json",
    }

    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    short_token = f"SHORT_TOKEN_{stamp}"
    long_token = f"LONG_TOKEN_{stamp}"

    def chat(client: httpx.Client, message: str) -> dict:
        response = client.post(
            f"{base_url}/api/chat",
            headers=headers,
            json={
                "message": message,
                "user_id": user_id,
                "language": "en",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert isinstance(payload, dict), payload
        return payload

    with httpx.Client(timeout=120.0) as client:
        _ = chat(
            client,
            f"For short-term memory verification, remember this exact token: {short_token}. Reply only with ACK_SHORT.",
        )

        time.sleep(1.0)
        short_recall = chat(
            client,
            "What short token did I just ask you to remember? Reply with the exact token only.",
        )
        short_recall_text = str(short_recall.get("response") or "")
        assert short_token in short_recall_text, short_recall

        _ = chat(
            client,
            f"Store this as a persistent long-term memory fact for future turns: {long_token}. Reply only with ACK_LONG.",
        )

        # Allow async memory update flow to complete before querying memories.
        time.sleep(2.0)

        memories_response = client.get(
            f"{base_url}/api/memories",
            headers={"x-chat-token": chat_token},
            params={"query": long_token, "limit": 10, "include_related": "false"},
        )
        assert memories_response.status_code == 200, memories_response.text
        memories_payload = memories_response.json()
        items = memories_payload.get("items") if isinstance(memories_payload, dict) else None
        assert isinstance(items, list), memories_payload
        assert any(long_token in str(item.get("text") or "") for item in items), memories_payload

        time.sleep(1.0)
        long_recall = chat(
            client,
            "What persistent long-term token did I ask you to store? Reply with the exact token only.",
        )
        long_recall_text = str(long_recall.get("response") or "")
        assert long_token in long_recall_text, long_recall

        metadata = long_recall.get("metadata") or {}
        memories_used = metadata.get("memories_used") or []
        assert isinstance(memories_used, list), long_recall
