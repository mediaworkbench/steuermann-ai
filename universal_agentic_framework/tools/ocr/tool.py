"""OCR tool — extract text from images using the configured vision model."""

from typing import ClassVar, Optional

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
    "You are an OCR engine. Extract all text visible in this image exactly as it appears. "
    "Output only the extracted text, preserving line breaks and formatting. "
    "Do not describe the image or add commentary. If no text is visible, output: [No text found]"
)


class OcrInput(BaseModel):
    """Input for OCR text extraction."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )


class OcrTool(BaseTool):
    """Extract text from images using the configured vision model."""

    name: str = "ocr_tool"
    description: str = (
        "Extract text from an image using OCR (Optical Character Recognition). "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Use when the user asks to read, transcribe, or extract text visible in an image — "
        "such as text on whiteboards, receipts, handwritten notes, screenshots, or signs. "
        "Trigger phrases: read the text, extract text, what does it say, transcribe, OCR, "
        "what text is in the image, read the writing."
    )
    args_schema: type[BaseModel] = OcrInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _run(self, image_source: str, **kwargs) -> str:
        """Extract text from image synchronously."""
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
                bare_model, temperature, "Extract all text from this image.", data_url,
                max_tokens=4096, system_prompt=_SYSTEM_PROMPT,
            )

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error("ocr_tool failed", error=str(exc), image_source=image_source)
            return f"Error extracting text: {exc}"

    async def _arun(self, image_source: str, **kwargs) -> str:
        """Extract text from image asynchronously."""
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
                bare_model, temperature, "Extract all text from this image.", data_url,
                max_tokens=4096, system_prompt=_SYSTEM_PROMPT,
            )

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error("ocr_tool failed", error=str(exc), image_source=image_source)
            return f"Error extracting text: {exc}"
