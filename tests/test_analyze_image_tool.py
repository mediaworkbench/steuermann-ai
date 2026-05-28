"""Tests for analyze_image_tool."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.tools.analyze_image.tool import (
    AnalyzeImageInput,
    AnalyzeImageTool,
)

# Patch target for _load_vision_api_config — keeps tests independent of real config.
_VISION_CONFIG_PATCH = "universal_agentic_framework.tools.analyze_image.tool._load_vision_api_config"
_FAKE_CONFIG = ("http://localhost:1234/v1", "google/gemma-4-e4b", "0.2", "no-key")


# ── Input schema ───────────────────────────────────────────────────────────────


class TestAnalyzeImageInput:
    def test_default_prompt(self):
        inp = AnalyzeImageInput(image_source="https://example.com/photo.jpg")
        assert inp.prompt == "Describe this image in detail."

    def test_url_source_accepted(self):
        inp = AnalyzeImageInput(image_source="https://example.com/photo.jpg")
        assert inp.image_source == "https://example.com/photo.jpg"

    def test_local_path_accepted(self):
        inp = AnalyzeImageInput(image_source="/tmp/test/image.png")
        assert inp.image_source == "/tmp/test/image.png"

    def test_custom_prompt(self):
        inp = AnalyzeImageInput(image_source="https://example.com/a.jpg", prompt="What text is visible?")
        assert inp.prompt == "What text is visible?"


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestAnalyzeImageToolSync:
    def _make_tool(self, tmp_path):
        return AnalyzeImageTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_run_with_url_returns_analysis(self, _mock_cfg, mock_client_cls, tmp_path):
        # Set up fetch response (GET) and vision API response (POST)
        fetch_resp = Mock()
        fetch_resp.content = b"\xff\xd8\xff"
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "A cat."}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client.post.return_value = api_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/photo.jpg")
        assert result == "A cat."

    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_path_traversal_returns_error_string(self, _mock_cfg, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(outside))
        assert "Error" in result
        assert "outside" in result.lower() or "error" in result.lower()

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_image_too_large_returns_error(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"x" * (6 * 1024 * 1024)
        fetch_resp.headers = {"content-type": "image/png"}
        fetch_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = AnalyzeImageTool(attachments_base_dir=str(tmp_path), max_image_bytes=5 * 1024 * 1024)
        result = tool._run(image_source="https://example.com/big.jpg")
        assert "too large" in result.lower() or "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestAnalyzeImageToolAsync:
    def _make_tool(self, tmp_path):
        return AnalyzeImageTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_url_source_fetches_and_returns_analysis(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\x89PNG\r\n"
        fetch_resp.headers = {"content-type": "image/png"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "A dog."}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=fetch_resp)
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/dog.png")
        assert result == "A dog."

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_local_path_reads_file_and_returns_analysis(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "A landscape."}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        assert result == "A landscape."

        # Verify the POST payload included a base64 data URL
        call_json = mock_client.post.call_args[1]["json"]
        content_block = call_json["messages"][0]["content"]
        image_block = next(b for b in content_block if b["type"] == "image_url")
        assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_path_outside_attachments_dir_returns_error(self, _mock_cfg, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(outside))
        assert "Error" in result

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_vision_api_error_returns_error_string(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        assert "Error" in result
        assert "connection refused" in result

    @pytest.mark.asyncio
    async def test_vision_role_not_configured_returns_error(self, tmp_path):
        mock_config = Mock()
        mock_config.llm.roles.vision = None

        with patch(
            "universal_agentic_framework.tools.analyze_image.tool._load_vision_api_config",
            side_effect=ValueError("llm.roles.vision is not configured in the active profile."),
        ):
            tool = self._make_tool(tmp_path)
            result = await tool._arun(image_source="https://example.com/photo.jpg")
        assert "Error" in result
        assert "vision" in result.lower()


# ── Registration / structural ──────────────────────────────────────────────────


class TestAnalyzeImageToolRegistration:
    def test_tool_name(self):
        tool = AnalyzeImageTool()
        assert tool.name == "analyze_image_tool"

    def test_tool_yaml_present(self):
        yaml_path = Path(__file__).parent.parent / "universal_agentic_framework" / "tools" / "analyze_image" / "tool.yaml"
        assert yaml_path.is_file(), f"tool.yaml not found at {yaml_path}"

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "profiles" / "starter" / "tools.yaml"
        content = config_path.read_text()
        assert "analyze_image_tool" in content

    def test_default_attachments_base_dir(self):
        tool = AnalyzeImageTool()
        assert tool.attachments_base_dir == "/tmp/steuermann-ai/chat-workspaces"

    def test_default_max_image_bytes(self):
        tool = AnalyzeImageTool()
        assert tool.max_image_bytes == 10 * 1024 * 1024
