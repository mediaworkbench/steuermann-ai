from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx


logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")


class PrometheusClient:
    """Client for querying Prometheus metrics."""

    def __init__(self, base_url: str = PROMETHEUS_URL) -> None:
        self.base_url = base_url.rstrip("/")

    async def query(self, promql: str) -> list[Dict[str, Any]]:
        """Execute a PromQL query and return results."""
        url = f"{self.base_url}/api/v1/query"
        params = {"query": promql}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data.get("status") != "success":
                logger.warning(f"Prometheus query failed: {data.get('error', 'unknown error')}")
                return []

            return data.get("data", {}).get("result", [])
        except httpx.RequestError as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return []

    async def query_range(
        self, promql: str, start: str, end: str, step: str = "60s"
    ) -> list[Dict[str, Any]]:
        """Execute a PromQL query over a time range."""
        url = f"{self.base_url}/api/v1/query_range"
        params = {"query": promql, "start": start, "end": end, "step": step}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data.get("status") != "success":
                logger.warning(f"Prometheus query_range failed: {data.get('error', 'unknown error')}")
                return []

            return data.get("data", {}).get("result", [])
        except httpx.RequestError as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return []


def extract_value(result: Dict[str, Any]) -> Optional[float]:
    """Extract the numeric value from a Prometheus query result."""
    if not result or "value" not in result:
        return None
    try:
        return float(result["value"][1])
    except (IndexError, ValueError, TypeError):
        return None


def extract_values_dict(results: list[Dict[str, Any]]) -> Dict[str, float]:
    """Extract a dictionary of label combinations to values from query results."""
    output = {}
    for result in results:
        value = extract_value(result)
        if value is not None:
            labels = result.get("metric", {})
            # Create a key from the metric name and labels
            metric_name = labels.get("__name__", "unknown")
            label_parts = [f"{k}={v}" for k, v in labels.items() if k != "__name__"]
            key = f"{metric_name}{{" + ",".join(label_parts) + "}" if label_parts else metric_name
            output[key] = value
    return output
