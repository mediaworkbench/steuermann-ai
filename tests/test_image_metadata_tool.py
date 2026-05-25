"""Tests for image_metadata_tool."""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from universal_agentic_framework.tools.image_metadata.tool import (
    ImageMetadataInput,
    ImageMetadataTool,
    _decode_exif,
    _extract_metadata,
    _safe_value,
)


# ── Input schema ───────────────────────────────────────────────────────────────


class TestImageMetadataInput:
    def test_url_accepted(self):
        inp = ImageMetadataInput(image_source="https://example.com/photo.jpg")
        assert inp.image_source == "https://example.com/photo.jpg"

    def test_local_path_accepted(self):
        inp = ImageMetadataInput(image_source="/tmp/photo.jpg")
        assert inp.image_source == "/tmp/photo.jpg"

    def test_image_source_required(self):
        with pytest.raises(Exception):
            ImageMetadataInput()


# ── Helper: _safe_value ────────────────────────────────────────────────────────


class TestSafeValue:
    def test_int_passthrough(self):
        assert _safe_value(42) == 42

    def test_string_passthrough(self):
        assert _safe_value("hello") == "hello"

    def test_bytes_decoded(self):
        result = _safe_value(b"hello")
        assert result == "hello"

    def test_tuple_becomes_list(self):
        result = _safe_value((1, 2, 3))
        assert result == [1, 2, 3]

    def test_none_passthrough(self):
        assert _safe_value(None) is None

    def test_unknown_type_stringified(self):
        class Weird:
            def __str__(self):
                return "weird"
        assert _safe_value(Weird()) == "weird"


# ── Helper: _extract_metadata ──────────────────────────────────────────────────


class TestExtractMetadata:
    def test_basic_fields_present(self):
        import io
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        # Create a minimal in-memory JPEG
        buf = io.BytesIO()
        img = Image.new("RGB", (100, 50), color=(255, 0, 0))
        img.save(buf, format="JPEG")
        buf.seek(0)
        image_bytes = buf.read()

        meta = _extract_metadata(image_bytes, "test.jpg")
        assert meta["filename"] == "test.jpg"
        assert meta["width"] == 100
        assert meta["height"] == 50
        assert meta["format"] == "JPEG"
        assert meta["mode"] == "RGB"

    def test_pillow_missing_returns_error(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'PIL'")):
                result = _extract_metadata(b"\xff\xd8\xff", "photo.jpg")
                assert "error" in result


# ── Sync _run() ────────────────────────────────────────────────────────────────


class TestImageMetadataToolSync:
    def _make_tool(self, tmp_path):
        return ImageMetadataTool(attachments_base_dir=str(tmp_path))

    @patch("httpx.Client")
    def test_run_with_url_returns_json(self, mock_client_cls, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        buf = io.BytesIO()
        Image.new("RGB", (200, 100)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        resp = Mock()
        resp.content = jpeg_bytes
        resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source="https://example.com/photo.jpg")
        parsed = json.loads(result)
        assert parsed["width"] == 200
        assert parsed["height"] == 100

    def test_run_with_local_file_returns_json(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img_path = tmp_path / "photo.jpg"
        buf = io.BytesIO()
        Image.new("RGB", (50, 75)).save(buf, format="JPEG")
        img_path.write_bytes(buf.getvalue())

        tool = self._make_tool(tmp_path)
        result = tool._run(image_source=str(img_path))
        parsed = json.loads(result)
        assert parsed["filename"] == "photo.jpg"
        assert parsed["width"] == 50

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

        tool = ImageMetadataTool(attachments_base_dir=str(tmp_path), max_image_bytes=5 * 1024 * 1024)
        result = tool._run(image_source="https://example.com/big.jpg")
        assert "too large" in result.lower() or "Error" in result


# ── Async _arun() ──────────────────────────────────────────────────────────────


class TestImageMetadataToolAsync:
    def _make_tool(self, tmp_path):
        return ImageMetadataTool(attachments_base_dir=str(tmp_path))

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_arun_with_url_returns_json(self, mock_client_cls, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        buf = io.BytesIO()
        Image.new("RGB", (320, 240)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        resp = Mock()
        resp.content = png_bytes
        resp.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/photo.png")
        parsed = json.loads(result)
        assert parsed["width"] == 320

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_api_error_returns_error_string(self, mock_client_cls, tmp_path):
        mock_client = Mock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        tool = self._make_tool(tmp_path)
        result = await tool._arun(image_source="https://example.com/photo.jpg")
        assert "Error" in result


# ── Registration / structural ──────────────────────────────────────────────────


class TestImageMetadataToolRegistration:
    def test_tool_name(self):
        assert ImageMetadataTool().name == "image_metadata_tool"

    def test_tool_yaml_present(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "universal_agentic_framework" / "tools" / "image_metadata" / "tool.yaml"
        )
        assert yaml_path.is_file()

    def test_tool_in_tools_config(self):
        config_path = Path(__file__).parent.parent / "config" / "tools.yaml"
        assert "image_metadata_tool" in config_path.read_text()

    def test_default_max_image_bytes(self):
        assert ImageMetadataTool().max_image_bytes == 10 * 1024 * 1024
