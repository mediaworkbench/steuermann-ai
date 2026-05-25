"""Tests for read_barcodes_tool."""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from universal_agentic_framework.tools.read_barcodes.tool import (
    ReadBarcodesInput,
    ReadBarcodesTool,
    _decode_barcodes,
)


# ── Input schema ───────────────────────────────────────────────────────────────


class TestReadBarcodesInput:
    def test_url_accepted(self):
        inp = ReadBarcodesInput(image_source="https://example.com/qr.png")
        assert inp.image_source == "https://example.com/qr.png"

    def test_local_path_accepted(self):
        inp = ReadBarcodesInput(image_source="/tmp/barcode.jpg")
        assert inp.image_source == "/tmp/barcode.jpg"

    def test_image_source_required(self):
        with pytest.raises(Exception):
            ReadBarcodesInput()


# ── Helper: _decode_barcodes ───────────────────────────────────────────────────


class TestDecodeBarcodes:
    def test_pyzbar_not_installed_returns_error(self):
        with patch.dict("sys.modules", {"pyzbar": None, "pyzbar.pyzbar": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'pyzbar'")):
                result = _decode_barcodes(b"\x89PNG")
                assert "error" in result

    def test_empty_image_returns_no_codes(self):
        try:
            from PIL import Image
            from pyzbar import pyzbar as _pyzbar_check  # noqa: F401
        except ImportError:
            pytest.skip("pyzbar or Pillow not installed")

        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(buf, format="PNG")
        result = _decode_barcodes(buf.getvalue())
        assert result["found"] is False
        assert result["codes"] == []

    def test_mocked_barcode_decoded(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        FakeRect = MagicMock()
        FakeRect.left = 10
        FakeRect.top = 20
        FakeRect.width = 80
        FakeRect.height = 80

        FakeCode = MagicMock()
        FakeCode.type = "QR_CODE"
        FakeCode.data = b"https://example.com"
        FakeCode.rect = FakeRect

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="PNG")

        # Patch at the import site inside _decode_barcodes, not via the pyzbar package
        # (pyzbar may not be installed on the dev host — it is a Docker dependency).
        fake_pyzbar_module = MagicMock()
        fake_pyzbar_module.decode.return_value = [FakeCode]

        with patch.dict("sys.modules", {"pyzbar": MagicMock(), "pyzbar.pyzbar": fake_pyzbar_module}):
            result = _decode_barcodes(buf.getvalue())

        assert result["found"] is True
        assert len(result["codes"]) == 1
        assert result["codes"][0]["type"] == "QR_CODE"
        assert result["codes"][0]["data"] == "https://example.com"
        assert result["codes"][0]["position"]["left"] == 10


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestReadBarcodesToolSync:
    def _make_tool(self, tmp_path):
        return ReadBarcodesTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    @patch("universal_agentic_framework.tools.read_barcodes.tool._decode_barcodes")
    def test_run_with_url_returns_json(self, mock_decode, mock_client_cls, tmp_path):
        mock_decode.return_value = {
            "found": True,
            "codes": [{"type": "EAN13", "data": "1234567890123", "position": {"left": 5, "top": 5, "width": 90, "height": 90}}],
        }

        resp = Mock()
        resp.content = b"\x89PNG"
        resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/product.png")

        parsed = json.loads(result)
        assert parsed["found"] is True
        assert parsed["codes"][0]["data"] == "1234567890123"

    def test_path_traversal_returns_error(self, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(outside))
        assert "Error" in result

    @patch("httpx.Client")
    def test_image_too_large_returns_error(self, mock_client_cls, tmp_path):
        resp = Mock()
        resp.content = b"x" * (6 * 1024 * 1024)
        resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = ReadBarcodesTool(attachments_base_dir=str(tmp_path), max_image_bytes=5 * 1024 * 1024)
        result = tool._run(image_source="https://example.com/big.jpg")
        assert "too large" in result.lower() or "Error" in result

    @patch("httpx.Client")
    def test_fetch_error_returns_error_string(self, mock_client_cls, tmp_path):
        mock_client = Mock()
        mock_client.get.side_effect = Exception("connection refused")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/qr.png")
        assert "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestReadBarcodesToolAsync:
    def _make_tool(self, tmp_path):
        return ReadBarcodesTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("universal_agentic_framework.tools.read_barcodes.tool._decode_barcodes")
    async def test_arun_with_url_returns_json(self, mock_decode, mock_client_cls, tmp_path):
        mock_decode.return_value = {"found": False, "codes": []}

        resp = Mock()
        resp.content = b"\x89PNG"
        resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/qr.png")

        parsed = json.loads(result)
        assert parsed["found"] is False

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_api_error_returns_error_string(self, mock_client_cls, tmp_path):
        mock_client = Mock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/qr.png")
        assert "Error" in result


# ── Registration / structural ──────────────────────────────────────────────────


class TestReadBarcodesToolRegistration:
    def test_tool_name(self):
        assert ReadBarcodesTool().name == "read_barcodes_tool"

    def test_tool_yaml_present(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "universal_agentic_framework" / "tools" / "read_barcodes" / "tool.yaml"
        )
        assert yaml_path.is_file()

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "tools.yaml"
        assert "read_barcodes_tool" in config_path.read_text()

    def test_default_max_image_bytes(self):
        assert ReadBarcodesTool().max_image_bytes == 10 * 1024 * 1024
