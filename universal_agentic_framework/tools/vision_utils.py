"""Shared helpers for vision model tools."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional


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


def _build_request_payload(
    bare_model: str,
    temperature: str,
    prompt: str,
    data_url: str,
    max_tokens: int,
    *,
    system_prompt: Optional[str] = None,
) -> dict:
    """Build an OpenAI-compatible multimodal chat completions payload.

    Pass system_prompt to inject a specialized instruction (e.g. for OCR or
    document extraction). When None, no system message is added and the prompt
    acts as the sole user turn — preserving the original analyze_image behavior.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    })
    return {
        "model": bare_model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
    }
