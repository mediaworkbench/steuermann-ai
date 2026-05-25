"""Barcode and QR code reader tool — decode barcodes from images using pyzbar + Pillow."""

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


def _decode_barcodes(image_bytes: bytes) -> dict:
    """Decode all barcodes and QR codes in the image bytes."""
    try:
        from PIL import Image
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError as exc:
        missing = "pyzbar" if "pyzbar" in str(exc) else "Pillow"
        return {
            "error": (
                f"{missing} is not available. "
                "Ensure pyzbar is installed (poetry add pyzbar) and "
                "libzbar0 is installed in the container (apt-get install -y libzbar0)."
            )
        }

    with Image.open(io.BytesIO(image_bytes)) as img:
        decoded = pyzbar_decode(img)

    codes = []
    for item in decoded:
        codes.append({
            "type": item.type,
            "data": item.data.decode("utf-8", errors="replace"),
            "position": {
                "left": item.rect.left,
                "top": item.rect.top,
                "width": item.rect.width,
                "height": item.rect.height,
            },
        })

    return {"found": len(codes) > 0, "codes": codes}


class ReadBarcodesInput(BaseModel):
    """Input for barcode and QR code decoding."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )


class ReadBarcodesTool(BaseTool):
    """Decode barcodes and QR codes from images using pyzbar. No vision model required."""

    name: str = "read_barcodes_tool"
    description: str = (
        "Decode barcodes and QR codes from an image. "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Returns JSON with decoded data, barcode type, and position for each code found. "
        "Does NOT use the vision model; decoding is deterministic using pyzbar. "
        "Use when the user asks to scan or read a barcode, QR code, or product code. "
        "Trigger phrases: scan barcode, read QR code, decode QR, what does this QR code say, "
        "scan this code, product barcode, QR-Code scannen, Barcode lesen, Code auslesen."
    )
    args_schema: type[BaseModel] = ReadBarcodesInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _run(self, image_source: str, **kwargs) -> str:
        """Decode barcodes synchronously."""
        try:
            if image_source.startswith(("http://", "https://")):
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(image_source, follow_redirects=True)
                    resp.raise_for_status()
                    image_bytes = resp.content
            else:
                image_bytes, _ = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            result = _decode_barcodes(image_bytes)
            return json.dumps(result, ensure_ascii=False)

        except Exception as exc:
            logger.error("read_barcodes_tool failed", error=str(exc), image_source=image_source)
            return f"Error reading barcodes: {exc}"

    async def _arun(self, image_source: str, **kwargs) -> str:
        """Decode barcodes asynchronously."""
        try:
            if image_source.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(image_source, follow_redirects=True)
                    resp.raise_for_status()
                    image_bytes = resp.content
            else:
                image_bytes, _ = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            result = _decode_barcodes(image_bytes)
            return json.dumps(result, ensure_ascii=False)

        except Exception as exc:
            logger.error("read_barcodes_tool failed", error=str(exc), image_source=image_source)
            return f"Error reading barcodes: {exc}"
