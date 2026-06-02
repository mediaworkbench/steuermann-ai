"""Tests for analyze_document_tool."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.tools.analyze_document.tool import (
    AnalyzeDocumentInput,
    AnalyzeDocumentTool,
)

_VISION_CONFIG_PATCH = "universal_agentic_framework.tools.analyze_document.tool._load_vision_api_config"
_FAKE_CONFIG = ("http://localhost:1234/v1", "google/gemma-4-e4b", "0.2", "no-key")

_SAMPLE_JSON = json.dumps({
    "document_type": "invoice",
    "vendor": "ACME Corp",
    "date": "2026-05-25",
    "total": "250.00",
    "currency": "EUR",
    "line_items": [{"description": "Widget", "quantity": "2", "unit_price": "125.00", "amount": "250.00"}],
    "notes": None,
})


# ── Input schema ───────────────────────────────────────────────────────────────


class TestAnalyzeDocumentInput:
    def test_defaults(self):
        inp = AnalyzeDocumentInput(image_source="https://example.com/inv.jpg")
        assert inp.document_type == "auto"

    def test_document_type_hint_accepted(self):
        inp = AnalyzeDocumentInput(image_source="https://example.com/inv.jpg", document_type="invoice")
        assert inp.document_type == "invoice"

    def test_image_source_required(self):
        with pytest.raises(Exception):
            AnalyzeDocumentInput()


# ── User prompt building ────────────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_auto_type_has_no_hint(self):
        tool = AnalyzeDocumentTool()
        prompt = tool._build_user_prompt("auto")
        assert "invoice" not in prompt.lower()
        assert "receipt" not in prompt.lower()

    def test_invoice_hint_included(self):
        tool = AnalyzeDocumentTool()
        prompt = tool._build_user_prompt("invoice")
        assert "invoice" in prompt.lower()

    def test_unknown_type_treated_as_auto(self):
        tool = AnalyzeDocumentTool()
        prompt = tool._build_user_prompt("unknown_type")
        assert "Extract all structured data" in prompt


# ── JSON cleaning ──────────────────────────────────────────────────────────────


class TestCleanJsonOutput:
    def test_plain_json_unchanged(self):
        raw = '{"a": 1}'
        assert AnalyzeDocumentTool._clean_json_output(raw) == '{"a": 1}'

    def test_strips_json_fence(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert AnalyzeDocumentTool._clean_json_output(raw) == '{"a": 1}'

    def test_strips_plain_fence(self):
        raw = "```\n{\"a\": 1}\n```"
        assert AnalyzeDocumentTool._clean_json_output(raw) == '{"a": 1}'

    def test_invalid_json_returned_as_is(self):
        raw = "not json at all"
        assert AnalyzeDocumentTool._clean_json_output(raw) == "not json at all"


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestAnalyzeDocumentToolSync:
    def _make_tool(self, tmp_path):
        return AnalyzeDocumentTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_run_with_url_returns_json(self, _mock_cfg, mock_client_cls, tmp_path):
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
        result = tool._run(image_source="https://example.com/inv.jpg")
        parsed = json.loads(result)
        assert parsed["vendor"] == "ACME Corp"

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
        tool._run(image_source="https://example.com/inv.jpg")

        call_json = mock_client.post.call_args[1]["json"]
        roles = [m["role"] for m in call_json["messages"]]
        assert roles[0] == "system"

    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    def test_path_traversal_returns_error(self, _mock_cfg, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(outside))
        assert "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestAnalyzeDocumentToolAsync:
    def _make_tool(self, tmp_path):
        return AnalyzeDocumentTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_arun_returns_json_string(self, _mock_cfg, mock_client_cls, tmp_path):
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
        result = await tool._arun(image_source="https://example.com/receipt.png")
        parsed = json.loads(result)
        assert parsed["document_type"] == "invoice"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch(_VISION_CONFIG_PATCH, return_value=_FAKE_CONFIG)
    async def test_fenced_json_stripped(self, _mock_cfg, mock_client_cls, tmp_path):
        img = tmp_path / "doc.jpg"
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
        assert parsed["vendor"] == "ACME Corp"

    @pytest.mark.asyncio
    @patch(_VISION_CONFIG_PATCH, side_effect=ValueError("llm.roles.vision is not configured"))
    async def test_vision_not_configured_returns_error(self, _mock_cfg, tmp_path):
        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/a.jpg")
        assert "Error" in result


# ── Registration / structural ──────────────────────────────────────────────────


class TestAnalyzeDocumentToolRegistration:
    def test_tool_name(self):
        assert AnalyzeDocumentTool().name == "analyze_document_tool"

    def test_tool_yaml_present(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "universal_agentic_framework" / "tools" / "analyze_document" / "tool.yaml"
        )
        assert yaml_path.is_file()

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "profiles" / "starter" / "tools.yaml"
        assert "analyze_document_tool" in config_path.read_text()

    def test_default_max_image_bytes(self):
        assert AnalyzeDocumentTool().max_image_bytes == 10 * 1024 * 1024
