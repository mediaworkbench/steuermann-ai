from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import metrics as metrics_module
from backend.routers.metrics import router as metrics_router


class _FakePromClient:
    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs

    async def query(self, promql: str):
        if "langgraph_requests_total" in promql:
            return [{"metric": {"status": "200"}, "value": [0, "42"]}]
        if "langgraph_memory_operations_total" in promql and "operation" in promql:
            return [{"metric": {"operation": "load"}, "value": [0, "5"]}]
        return []

    async def query_range(self, promql: str, start: str, end: str, step: str = "60s"):
        _ = end, step
        start_ts = datetime.fromisoformat(start).timestamp()
        next_ts = start_ts + 86400
        if 'operation="load"' in promql:
            return [{"values": [[start_ts, "2"], [next_ts, "3"]]}]
        if 'operation="update"' in promql:
            return [{"values": [[start_ts, "1"], [next_ts, "1"]]}]
        if 'status="error"' in promql:
            return [{"values": [[start_ts, "0"], [next_ts, "1"]]}]
        if "langgraph_memory_quality_score_sum" in promql:
            return [{"values": [[start_ts, "0.8"], [next_ts, "0.5"]]}]
        return []


def test_memory_trends_endpoint_shape(monkeypatch):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(metrics_module, "PrometheusClient", _FakePromClient)

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    response = client.get("/api/analytics/memory-trends?days=2")

    assert response.status_code == 200
    body = response.json()
    assert body["period_days"] == 2
    assert len(body["trends"]) == 2
    assert set(body["trends"][0].keys()) == {
        "date",
        "loads",
        "updates",
        "errors",
        "error_rate",
        "avg_quality_score",
    }
    assert set(body["totals"].keys()) == {"loads", "updates", "errors", "error_rate"}


def test_memory_trends_error_rate_calculation(monkeypatch):
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(metrics_module, "PrometheusClient", _FakePromClient)

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    response = client.get("/api/analytics/memory-trends?days=2")

    assert response.status_code == 200
    trends = response.json()["trends"]
    # day 2 has loads=3, updates=1, errors=1 => 25% error rate
    assert trends[1]["error_rate"] == 25.0
