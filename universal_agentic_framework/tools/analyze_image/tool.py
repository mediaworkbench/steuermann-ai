"""Vision model image analysis tool."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional

import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = structlog.get_logger()

_DEFAULT_PROMPT = "Describe this image in detail."


class AnalyzeImageInput(BaseModel):
    """Input for image analysis."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )
    prompt: str = Field(
        default=_DEFAULT_PROMPT,
        description="What to analyze or look for in the image.",
    )


def _resolve_local_image(image_source: str, base_dir: str) -> tuple[bytes, str]:
    """Read a local image file and validate it is inside base_dir."""
    path = Path(image_source).resolve()
    base = Path(base_dir).resolve()
    if not str(path).startswith(str(base) + os.sep) and path != base:
        raise ValueError(
            f"Path '{image_source}' is outside the allowed attachments directory."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    return path.read_bytes(), mime_type


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"


def _load_vision_api_config() -> tuple[str, str, str, Optional[str]]:
    """Return (api_base, bare_model, temperature, api_key) for the vision role."""
    from universal_agentic_framework.config import load_core_config

    config = load_core_config()
    vision_role = getattr(getattr(config.llm, "roles", None), "vision", None)
    if vision_role is None:
        raise ValueError("llm.roles.vision is not configured in the active profile.")
    provider = config.llm.get_role_provider("vision")
    api_base = str(provider.api_base or "").rstrip("/")
    model_name = config.llm.get_role_model_name("vision", "en")
    bare_model = model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
    return api_base, bare_model, str(provider.temperature or 0.2), provider.api_key


def _build_request_payload(bare_model: str, temperature: str, prompt: str, data_url: str, max_tokens: int) -> dict:
    return {
        "model": bare_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": float(temperature),
        "max_tokens": max_tokens,
    }


class AnalyzeImageTool(BaseTool):
    """Analyze images using the configured vision model."""

    name: str = "analyze_image_tool"
    description: str = (
        "Analyze an image using the vision model. "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Use when the user asks to describe, identify, read text from, or interpret an image."
    )
    args_schema: type[BaseModel] = AnalyzeImageInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _run(
        self,
        image_source: str,
        prompt: str = _DEFAULT_PROMPT,
        **kwargs,
    ) -> str:
        """Analyze an image synchronously."""
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
            payload = _build_request_payload(bare_model, temperature, prompt, data_url, max_tokens=2048)

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error("analyze_image_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing image: {exc}"

    async def _arun(
        self,
        image_source: str,
        prompt: str = _DEFAULT_PROMPT,
        **kwargs,
    ) -> str:
        """Analyze an image asynchronously."""
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
            payload = _build_request_payload(bare_model, temperature, prompt, data_url, max_tokens=2048)

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error("analyze_image_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing image: {exc}"
