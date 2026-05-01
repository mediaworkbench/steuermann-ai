from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import memories as memories_module
from backend.routers.memories import router as memories_router


class FakeMemoryBackend:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {
            "mem-1": {
                "point_id": "p-1",
                "payload": {
                    "user_id": "u1",
                    "metadata": {
                        "memory_id": "mem-1",
                        "user_rating": None,
                    },
                },
            },
            "mem-2": {
                "point_id": "p-2",
                "payload": {
                    "user_id": "other-user",
                    "metadata": {
                        "memory_id": "mem-2",
                        "user_rating": None,
                    },
                },
            },
        }

    def find_memory_point(self, memory_id: str) -> Optional[dict[str, Any]]:
        return self.records.get(memory_id)

    def set_memory_user_rating(self, *, point_id: Any, metadata: Optional[dict[str, Any]], rating: int) -> None:
        _ = point_id
        if metadata is not None:
            metadata["user_rating"] = rating


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    fake_backend = FakeMemoryBackend()
    monkeypatch.setattr(memories_module, "load_core_config", lambda: object())
    monkeypatch.setattr(memories_module, "build_memory_backend", lambda cfg: fake_backend)

    app = FastAPI()
    app.include_router(memories_router)
    return TestClient(app), fake_backend


def test_rate_memory_success(client):
    test_client, backend = client

    response = test_client.post("/api/memories/mem-1/rate", json={"rating": 5})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "memory_id": "mem-1", "rating": 5}
    assert backend.records["mem-1"]["payload"]["metadata"]["user_rating"] == 5


@pytest.mark.parametrize("rating", [0, 6, "bad"])
def test_rate_memory_invalid_rating(client, rating):
    test_client, _ = client

    response = test_client.post("/api/memories/mem-1/rate", json={"rating": rating})

    assert response.status_code == 422


def test_rate_memory_not_found(client):
    test_client, _ = client

    response = test_client.post("/api/memories/missing/rate", json={"rating": 3})

    assert response.status_code == 404
    assert response.json()["detail"] == "Memory not found"


def test_rate_memory_wrong_owner_forbidden(client):
    test_client, _ = client

    response = test_client.post("/api/memories/mem-2/rate", json={"rating": 4})

    assert response.status_code == 403
    assert response.json()["detail"] == "Memory does not belong to user"


def test_rate_memory_requires_token_when_configured(client, monkeypatch: pytest.MonkeyPatch):
    test_client, _ = client
    monkeypatch.setenv("CHAT_ACCESS_TOKEN", "secret-token")

    response = test_client.post("/api/memories/mem-1/rate", json={"rating": 4})

    assert response.status_code == 401
