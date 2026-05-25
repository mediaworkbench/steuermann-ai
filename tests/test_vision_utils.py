"""Tests for shared vision utility helpers."""

import base64
from pathlib import Path

import pytest

from universal_agentic_framework.tools.vision_utils import (
    _build_data_url,
    _build_request_payload,
    _resolve_local_image,
)


class TestBuildDataUrl:
    def test_jpeg_bytes(self):
        result = _build_data_url(b"\xff\xd8\xff", "image/jpeg")
        assert result.startswith("data:image/jpeg;base64,")
        decoded = base64.b64decode(result.split(",", 1)[1])
        assert decoded == b"\xff\xd8\xff"

    def test_png_mime_type(self):
        result = _build_data_url(b"\x89PNG", "image/png")
        assert result.startswith("data:image/png;base64,")

    def test_empty_bytes(self):
        result = _build_data_url(b"", "image/jpeg")
        assert result == "data:image/jpeg;base64,"


class TestResolveLocalImage:
    def test_valid_file_inside_base_dir(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        data, mime = _resolve_local_image(str(img), str(tmp_path))
        assert data == b"\xff\xd8\xff"
        assert mime == "image/jpeg"

    def test_path_traversal_rejected(self, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("sensitive")
        with pytest.raises(ValueError, match="outside the allowed attachments directory"):
            _resolve_local_image(str(outside), str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_local_image(str(tmp_path / "missing.jpg"), str(tmp_path))

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path):
        img = tmp_path / "photo.bin"
        img.write_bytes(b"\x00\x01\x02")
        _data, mime = _resolve_local_image(str(img), str(tmp_path))
        assert mime == "image/jpeg"

    def test_png_extension_inferred(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")
        _data, mime = _resolve_local_image(str(img), str(tmp_path))
        assert mime == "image/png"


class TestBuildRequestPayload:
    def test_no_system_prompt(self):
        payload = _build_request_payload("gemma", "0.2", "Describe this.", "data:image/jpeg;base64,abc", 512)
        assert payload["model"] == "gemma"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 512
        msgs = payload["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_with_system_prompt(self):
        payload = _build_request_payload(
            "gemma", "0.1", "Extract text.", "data:image/png;base64,xyz", 1024,
            system_prompt="You are an OCR engine.",
        )
        msgs = payload["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are an OCR engine."}
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_image_and_text_blocks(self):
        payload = _build_request_payload("m", "0.2", "Hello", "data:image/jpeg;base64,abc", 256)
        content = payload["messages"][0]["content"]
        types = [b["type"] for b in content]
        assert "text" in types
        assert "image_url" in types

    def test_temperature_string_converted_to_float(self):
        payload = _build_request_payload("m", "0.3", "p", "data:", 128)
        assert isinstance(payload["temperature"], float)
