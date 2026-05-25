"""Image metadata tool — extract EXIF and file metadata from images using Pillow."""

import io
import json
from pathlib import Path
from typing import Optional

import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from universal_agentic_framework.tools.vision_utils import _resolve_local_image

logger = structlog.get_logger()

# EXIF tag names from the TAGS dict in Pillow
_EXIF_GPS_TAG = 34853  # GPSInfo IFD pointer


def _decode_exif(exif_data) -> dict:
    """Convert raw Pillow ExifData / dict to a serializable dict."""
    try:
        from PIL.ExifTags import GPSTAGS, TAGS
    except ImportError:
        return {}

    result: dict = {}
    if exif_data is None:
        return result

    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, str(tag_id))

        if tag_id == _EXIF_GPS_TAG and isinstance(value, dict):
            gps: dict = {}
            for gps_tag_id, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                gps[gps_tag_name] = _safe_value(gps_value)
            result["GPS"] = gps
            continue

        result[tag_name] = _safe_value(value)

    return result


def _safe_value(v):
    """Return a JSON-serializable representation of an EXIF value."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v.hex()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, tuple):
        return list(_safe_value(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _safe_value(val) for k, val in v.items()}
    return str(v)


def _extract_metadata(image_bytes: bytes, filename: str) -> dict:
    """Open image bytes with Pillow and extract file + EXIF metadata."""
    try:
        from PIL import Image
    except ImportError:
        return {"error": "Pillow is not installed. Run: poetry add Pillow"}

    with Image.open(io.BytesIO(image_bytes)) as img:
        meta: dict = {
            "filename": filename,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
        }
        dpi = img.info.get("dpi")
        if dpi:
            meta["dpi"] = list(dpi)

        exif = img.getexif()
        if exif:
            decoded = _decode_exif(exif)
            if decoded:
                meta["exif"] = decoded
        else:
            meta["exif"] = {}

    return meta


class ImageMetadataInput(BaseModel):
    """Input for image metadata extraction."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )


class ImageMetadataTool(BaseTool):
    """Extract EXIF and file metadata from images using Pillow. No vision model required."""

    name: str = "image_metadata_tool"
    description: str = (
        "Extract metadata from an image file — file format, dimensions, DPI, and EXIF data "
        "such as camera model, capture date, GPS coordinates. "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Does NOT use the vision model; metadata is read directly from the file. "
        "Use when the user asks about when a photo was taken, where it was taken, "
        "what camera was used, what resolution it is, or EXIF information. "
        "Trigger phrases: when was this photo taken, where was this taken, what camera, "
        "what resolution, photo metadata, EXIF, GPS location of photo, "
        "wann wurde das Foto aufgenommen, welche Kamera, Bildgröße, GPS-Daten."
    )
    args_schema: type[BaseModel] = ImageMetadataInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _run(self, image_source: str, **kwargs) -> str:
        """Extract image metadata synchronously."""
        try:
            if image_source.startswith(("http://", "https://")):
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(image_source, follow_redirects=True)
                    resp.raise_for_status()
                    image_bytes = resp.content
                filename = Path(image_source.split("?")[0]).name or "image"
            else:
                image_bytes, _ = _resolve_local_image(image_source, self.attachments_base_dir)
                filename = Path(image_source).name

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            meta = _extract_metadata(image_bytes, filename)
            return json.dumps(meta, ensure_ascii=False, indent=2)

        except Exception as exc:
            logger.error("image_metadata_tool failed", error=str(exc), image_source=image_source)
            return f"Error extracting metadata: {exc}"

    async def _arun(self, image_source: str, **kwargs) -> str:
        """Extract image metadata asynchronously."""
        try:
            if image_source.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(image_source, follow_redirects=True)
                    resp.raise_for_status()
                    image_bytes = resp.content
                filename = Path(image_source.split("?")[0]).name or "image"
            else:
                image_bytes, _ = _resolve_local_image(image_source, self.attachments_base_dir)
                filename = Path(image_source).name

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            meta = _extract_metadata(image_bytes, filename)
            return json.dumps(meta, ensure_ascii=False, indent=2)

        except Exception as exc:
            logger.error("image_metadata_tool failed", error=str(exc), image_source=image_source)
            return f"Error extracting metadata: {exc}"
