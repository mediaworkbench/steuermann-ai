"""Chart analysis tool — extract structured data from charts and visualizations."""

import json
import re
import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from universal_agentic_framework.tools.vision_utils import (
    _build_data_url,
    _build_request_payload,
    _load_vision_api_config,
    _resolve_local_image,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a data visualization analyst. Analyze this chart or graph image and return a JSON object "
    "with the following fields: "
    "chart_type (string, e.g. bar, line, pie, scatter, histogram, area, heatmap), "
    "title (string or null), "
    "x_axis (object with fields: label (string or null), unit (string or null)), "
    "y_axis (object with fields: label (string or null), unit (string or null)), "
    "series (array of objects with fields: name (string), data_points (array of approximate values or labels)), "
    "key_observations (array of strings describing notable trends, peaks, or patterns). "
    "Use null for fields not visible in the chart. "
    "Return ONLY the JSON object, no prose, no markdown fences."
)

_USER_PROMPT = "Extract all structured data and observations from this chart or data visualization."


class AnalyzeChartInput(BaseModel):
    """Input for chart data extraction."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )


class AnalyzeChartTool(BaseTool):
    """Extract structured data from charts and data visualizations using the configured vision model."""

    name: str = "analyze_chart_tool"
    description: str = (
        "Extract structured data from a chart, graph, or data visualization image. "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Returns JSON with chart type, axis labels, data series, and key observations. "
        "Use when the user asks to analyze, read, or extract data from a chart, graph, or plot. "
        "Trigger phrases: analyze chart, read the graph, extract data from plot, "
        "what does the chart show, describe the trend, bar chart, line graph, pie chart, "
        "Diagramm analysieren, Grafik auslesen, was zeigt das Diagramm."
    )
    args_schema: type[BaseModel] = AnalyzeChartInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _run(self, image_source: str, **kwargs) -> str:
        """Extract chart data synchronously."""
        try:
            api_base, bare_model, temperature, api_key = _load_vision_api_config()
            if not api_base:
                return "Error: Vision model api_base is not configured."

            if image_source.startswith(("http://", "https://")):
                with httpx.Client(timeout=30.0) as client:
                    fetch_resp = client.get(image_source, follow_redirects=True)
                    fetch_resp.raise_for_status()
                    image_bytes = fetch_resp.content
                    mime_type = fetch_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            else:
                image_bytes, mime_type = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            data_url = _build_data_url(image_bytes, mime_type)
            payload = _build_request_payload(
                bare_model, temperature, _USER_PROMPT, data_url,
                max_tokens=2048, system_prompt=_SYSTEM_PROMPT,
            )

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return self._clean_json_output(raw)

        except Exception as exc:
            logger.error("analyze_chart_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing chart: {exc}"

    async def _arun(self, image_source: str, **kwargs) -> str:
        """Extract chart data asynchronously."""
        try:
            api_base, bare_model, temperature, api_key = _load_vision_api_config()
            if not api_base:
                return "Error: Vision model api_base is not configured."

            if image_source.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=30.0) as client:
                    fetch_resp = await client.get(image_source, follow_redirects=True)
                    fetch_resp.raise_for_status()
                    image_bytes = fetch_resp.content
                    mime_type = fetch_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            else:
                image_bytes, mime_type = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            data_url = _build_data_url(image_bytes, mime_type)
            payload = _build_request_payload(
                bare_model, temperature, _USER_PROMPT, data_url,
                max_tokens=2048, system_prompt=_SYSTEM_PROMPT,
            )

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return self._clean_json_output(raw)

        except Exception as exc:
            logger.error("analyze_chart_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing chart: {exc}"

    @staticmethod
    def _clean_json_output(raw: str) -> str:
        """Strip markdown fences if the model wrapped the JSON anyway."""
        stripped = raw.strip()
        fenced = re.match(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
        if fenced:
            stripped = fenced.group(1).strip()
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        return stripped
