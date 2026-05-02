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
                    "text": "memory one",
                    "metadata": {
                        "memory_id": "mem-1",
                        "created_at": "2026-05-01T00:00:00+00:00",
                        "user_rating": None,
                        "importance_score": 0.7,
                        "is_related": False,
                    },
                },
            },
            "mem-2": {
                "point_id": "p-2",
                "payload": {
                    "user_id": "other-user",
                    "text": "memory two",
                    "metadata": {
                        "memory_id": "mem-2",
                        "created_at": "2026-05-01T00:00:00+00:00",
                        "user_rating": None,
                        "importance_score": 0.3,
                        "is_related": True,
                    },
                },
            },
        }

    def load(
        self,
        user_id: str,
        query: Optional[str] = None,
        top_k: int = 5,
        include_related: bool = False,
        session_id: Optional[str] = None,
    ):
        _ = query, include_related, session_id
        from universal_agentic_framework.memory.backend import MemoryRecord

        out = []
        for record in self.records.values():
            payload = record.get("payload") or {}
            if payload.get("user_id") != user_id:
                continue
            out.append(
                MemoryRecord(
                    user_id=user_id,
                    text=str(payload.get("text") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return out[:top_k]

    def find_memory_point(self, memory_id: str) -> Optional[dict[str, Any]]:
        return self.records.get(memory_id)

    def set_memory_user_rating(self, *, point_id: Any, metadata: Optional[dict[str, Any]], rating: int) -> None:
        _ = point_id
        if metadata is not None:
            metadata["user_rating"] = rating

    def delete_memory(self, *, memory_id: str, user_id: str) -> None:
        record = self.records.get(memory_id)
        if not record:
            return
        owner = ((record.get("payload") or {}).get("user_id"))
        if owner != user_id:
            raise PermissionError("Memory does not belong to user")
        self.records.pop(memory_id, None)


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


def test_list_memories_returns_user_owned_memories(client):
    test_client, _ = client

    response = test_client.get("/api/memories?limit=50")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["memory_id"] == "mem-1"


def test_get_memory_detail_success(client):
    test_client, _ = client

    response = test_client.get("/api/memories/mem-1")

    assert response.status_code == 200
    body = response.json()
    assert body["memory_id"] == "mem-1"
    assert body["text"] == "memory one"


def test_get_memory_detail_forbidden_for_non_owner(client):
    test_client, _ = client

    response = test_client.get("/api/memories/mem-2")

    assert response.status_code == 403


def test_delete_memory_success(client):
    test_client, backend = client

    response = test_client.delete("/api/memories/mem-1")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "memory_id": "mem-1", "deleted": True}
    assert "mem-1" not in backend.records


def test_delete_memory_forbidden_for_non_owner(client):
    test_client, backend = client

    response = test_client.delete("/api/memories/mem-2")

    assert response.status_code == 403
    assert "mem-2" in backend.records


def test_memory_stats_summary(client):
    test_client, _ = client

    response = test_client.get("/api/memories/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["memories"] == 1
    assert "rated_coverage" in body["ratios"]
