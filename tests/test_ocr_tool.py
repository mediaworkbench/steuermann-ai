"""Tests for ocr_tool."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.tools.ocr.tool import OcrInput, OcrTool

_VISION_CONFIG_PATCH = "universal_agentic_framework.tools.ocr.tool._load_vision_api_config"
_FAKE_CONFIG = ("http://localhost:1234/v1", "google/gemma-4-e4b", "0.2", "no-key")


# ── Input schema ───────────────────────────────────────────────────────────────


class TestOcrInput:
    def test_url_source_accepted(self):
        inp = OcrInput(image_source="https://example.com/whiteboard.jpg")
        assert inp.image_source == "https://example.com/whiteboard.jpg"

    def test_local_path_accepted(self):
        inp = OcrInput(image_source="/tmp/test/note.png")
        assert inp.image_source == "/tmp/test/note.png"

    def test_image_source_required(self):
        with pytest.raises(Exception):
            OcrInput()


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestOcrToolSync:
    def _make_tool(self, tmp_path):
        return OcrTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_run_with_url_returns_extracted_text(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\xff\xd8\xff"
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "Hello World\nLine 2"}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client.post.return_value = api_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/whiteboard.jpg")
        assert result == "Hello World\nLine 2"

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_system_prompt_included_in_payload(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\xff\xd8\xff"
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "text"}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client.post.return_value = api_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        tool._run(image_source="https://example.com/a.jpg")

        call_json = mock_client.post.call_args[1]["json"]
        roles = [m["role"] for m in call_json["messages"]]
        assert roles[0] == "system"
        assert "OCR" in call_json["messages"][0]["content"]

    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_path_traversal_returns_error_string(self, _mock_cfg, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(outside))
        assert "Error" in result

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_image_too_large_returns_error(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"x" * (6 * 1024 * 1024)
        fetch_resp.headers = {"content-type": "image/jpeg"}
        fetch_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = fetch_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = OcrTool(attachments_base_dir=str(tmp_path), max_image_bytes=5 * 1024 * 1024)
        result = tool._run(image_source="https://example.com/big.jpg")
        assert "too large" in result.lower() or "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestOcrToolAsync:
    def _make_tool(self, tmp_path):
        return OcrTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_url_source_returns_extracted_text(self, _mock_cfg, mock_client_cls, tmp_path):
        fetch_resp = Mock()
        fetch_resp.content = b"\x89PNG\r\n"
        fetch_resp.headers = {"content-type": "image/png"}
        fetch_resp.raise_for_status.return_value = None

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "Invoice #123"}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=fetch_resp)
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/note.png")
        assert result == "Invoice #123"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_local_file_reads_and_returns_text(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        api_resp = Mock()
        api_resp.json.return_value = {"choices": [{"message": {"content": "Meeting notes: ..."}}]}
        api_resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=api_resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        assert result == "Meeting notes: ..."

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
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source=str(img))
        assert "Error" in result
        assert "timeout" in result


# ── Registration / structural ──────────────────────────────────────────────────


class TestOcrToolRegistration:
    def test_tool_name(self):
        assert OcrTool().name == "ocr_tool"

    def test_tool_yaml_present(self):
        yaml_path = Path(__file__).parent.parent / "universal_agentic_framework" / "tools" / "ocr" / "tool.yaml"
        assert yaml_path.is_file()

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "profiles" / "starter" / "tools.yaml"
        assert "ocr_tool" in config_path.read_text()

    def test_default_attachments_base_dir(self):
        assert OcrTool().attachments_base_dir == "/tmp/steuermann-ai/chat-workspaces"

    def test_default_max_image_bytes(self):
        assert OcrTool().max_image_bytes == 10 * 1024 * 1024
