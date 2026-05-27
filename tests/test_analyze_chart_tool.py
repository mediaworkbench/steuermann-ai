"""Tests for analyze_chart_tool."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.tools.analyze_chart.tool import (
    AnalyzeChartInput,
    AnalyzeChartTool,
)

_VISION_CONFIG_PATCH = "universal_agentic_framework.tools.analyze_chart.tool._load_vision_api_config"
_FAKE_CONFIG = ("http://localhost:1234/v1", "google/gemma-4-e4b", "0.2", "no-key")

_SAMPLE_JSON = json.dumps({
    "chart_type": "line",
    "title": "Monthly Revenue 2025",
    "x_axis": {"label": "Month", "unit": None},
    "y_axis": {"label": "Revenue", "unit": "EUR"},
    "series": [{"name": "Revenue", "data_points": [100, 150, 120, 180]}],
    "key_observations": ["Revenue peaked in April", "Overall upward trend"],
})


# ── Input schema ───────────────────────────────────────────────────────────────


class TestAnalyzeChartInput:
    def test_url_source_accepted(self):
        inp = AnalyzeChartInput(image_source="https://example.com/chart.png")
        assert inp.image_source == "https://example.com/chart.png"

    def test_local_path_accepted(self):
        inp = AnalyzeChartInput(image_source="/tmp/chart.png")
        assert inp.image_source == "/tmp/chart.png"

    def test_image_source_required(self):
        with pytest.raises(Exception):
            AnalyzeChartInput()


# ── JSON cleaning ──────────────────────────────────────────────────────────────


class TestCleanJsonOutput:
    def test_plain_json_unchanged(self):
        raw = '{"chart_type": "bar"}'
        assert AnalyzeChartTool._clean_json_output(raw) == '{"chart_type": "bar"}'

    def test_strips_json_fence(self):
        raw = '```json\n{"chart_type": "bar"}\n```'
        assert AnalyzeChartTool._clean_json_output(raw) == '{"chart_type": "bar"}'

    def test_invalid_json_returned_as_is(self):
        raw = "This is a bar chart showing..."
        assert AnalyzeChartTool._clean_json_output(raw) == raw


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestAnalyzeChartToolSync:
    def _make_tool(self, tmp_path):
        return AnalyzeChartTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_run_with_url_returns_chart_json(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\xff\xd8\xff"
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": _SAMPLE_JSON}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client.post.return_value = api_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/chart.jpg")
        parsed = json.loads(result)
        assert parsed["chart_type"] == "line"
        assert parsed["title"] == "Monthly Revenue 2025"

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_system_prompt_present_in_payload(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\xff\xd8\xff"
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client.post.return_value = api_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        tool._run(image_source="https://example.com/chart.jpg")

        call_json = mock_client.post.call_args[1]["json"]
        roles = [m["role"] for m in call_json["messages"]]
        assert roles[0] == "system"
        assert "visualization" in call_json["messages"][0]["content"].lower()

    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_path_traversal_returns_error(self, _mock_cfg, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(outside))
        assert "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestAnalyzeChartToolAsync:
    def _make_tool(self, tmp_path):
        return AnalyzeChartTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_arun_returns_chart_json(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\x89PNG\r\n"
        fetch_resp.headers = {"content-type": "image/png"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": _SAMPLE_JSON}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=fetch_resp)
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/chart.png")
        parsed = json.loads(result)
        assert len(parsed["key_observations"]) == 2

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_fenced_json_stripped(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "chart.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        fenced = f"```json\n{_SAMPLE_JSON}\n```"
        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": fenced}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        parsed = json.loads(result)
        assert parsed["chart_type"] == "line"

    @pytest.mark.asyncio
    @patch(_VISION_CONFIG_PATCH, side_effect=ValueError("llm.roles.vision is not configured"))
    async def test_vision_not_configured_returns_error(self, _mock_cfg, tmp_path):
        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/a.jpg")
        assert "Error" in result

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_api_error_returns_error_string(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "chart.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        assert "Error" in result
        assert "network error" in result


# ── Registration / structural ──────────────────────────────────────────────────


class TestAnalyzeChartToolRegistration:
    def test_tool_name(self):
        assert AnalyzeChartTool().name == "analyze_chart_tool"

    def test_tool_yaml_present(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "universal_agentic_framework" / "tools" / "analyze_chart" / "tool.yaml"
        )
        assert yaml_path.is_file()

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "tools.yaml"
        assert "analyze_chart_tool" in config_path.read_text()

    def test_default_max_image_bytes(self):
        assert AnalyzeChartTool().max_image_bytes == 10 * 1024 * 1024
