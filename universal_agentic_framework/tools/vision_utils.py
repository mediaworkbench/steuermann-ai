"""Shared helpers for vision model tools."""

import asyncio
import base64
import ipaddress
import mimetypes
import os
import socket
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

# Redirect hops followed (each one re-validated) when fetching a remote image.
_MAX_IMAGE_REDIRECTS = 3


class ImageTooLargeError(ValueError):
    """A remote image exceeded the tool's byte limit (raised before buffering it all)."""


def _resolve_host_ips(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (separate function so tests can stub DNS)."""
    return [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _assert_public_url(url: str) -> None:
    """Reject anything but http(s) to a publicly routable host.

    The model chooses the URL (from the user's message or from fetched web content), and
    the tool runs inside the compose network — without this, ``http://qdrant:6333`` or
    ``http://langgraph:8000`` are one prompt away. Every resolved address must be global
    (no private, loopback, link-local, CGNAT or reserved ranges). Residual risk: DNS can
    rebind between this check and the connect.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Only public http(s) image URLs are allowed: {url!r}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addresses = _resolve_host_ips(parts.hostname, port)
    except OSError as exc:
        raise ValueError(f"Cannot resolve image host {parts.hostname!r}: {exc}") from exc
    if not addresses:
        raise ValueError(f"Cannot resolve image host {parts.hostname!r}")
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise ValueError(f"Refusing to fetch image from non-public address {parts.hostname!r}")


def _image_too_large(size: int, max_bytes: int) -> ImageTooLargeError:
    return ImageTooLargeError(f"Image is too large ({size:,}+ bytes; limit is {max_bytes:,} bytes).")


def _declared_length(response: Any) -> Optional[int]:
    try:
        return int(response.headers.get("content-length"))
    except (TypeError, ValueError):
        return None


def _mime_from(response: Any) -> str:
    return (response.headers.get("content-type") or "image/jpeg").split(";")[0].strip()


def fetch_image_url(client: Any, url: str, max_bytes: int) -> tuple[bytes, str]:
    """GET a remote image through ``client`` (an ``httpx.Client``): SSRF-guarded on
    every redirect hop, and streamed so an oversized body is cut off at ``max_bytes``
    instead of being buffered whole. Returns ``(bytes, mime_type)``."""
    for _ in range(_MAX_IMAGE_REDIRECTS + 1):
        _assert_public_url(url)
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            declared = _declared_length(response)
            if declared is not None and declared > max_bytes:
                raise _image_too_large(declared, max_bytes)
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise _image_too_large(len(body), max_bytes)
            return bytes(body), _mime_from(response)
    raise ValueError(f"Too many redirects fetching image (max {_MAX_IMAGE_REDIRECTS})")


async def afetch_image_url(client: Any, url: str, max_bytes: int) -> tuple[bytes, str]:
    """Async twin of :func:`fetch_image_url` for an ``httpx.AsyncClient``."""
    for _ in range(_MAX_IMAGE_REDIRECTS + 1):
        await asyncio.to_thread(_assert_public_url, url)  # DNS lookup off the event loop
        async with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            declared = _declared_length(response)
            if declared is not None and declared > max_bytes:
                raise _image_too_large(declared, max_bytes)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise _image_too_large(len(body), max_bytes)
            return bytes(body), _mime_from(response)
    raise ValueError(f"Too many redirects fetching image (max {_MAX_IMAGE_REDIRECTS})")


def _resolve_local_file(file_path: str, base_dir: str) -> Path:
    """Resolve a local file path and validate it is confined inside base_dir.

    Unlike _resolve_local_image this does not assert an image MIME type, making
    it suitable for arbitrary file types (e.g. CSV workspace documents).

    Confinement here is per workspace root, which holds every user's files. Per-user
    isolation is enforced before the tool runs: the tool-calling nodes only accept a
    local path that is one of this turn's own attachment / document ``stored_path``s
    (``check_local_path_args`` in orchestration/helpers/tool_call_args.py).
    """
    path = Path(file_path).resolve()
    base = Path(base_dir).resolve()
    if not str(path).startswith(str(base) + os.sep) and path != base:
        raise ValueError(
            f"Path '{file_path}' is outside the allowed workspace directory."
        )
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


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
    """Return (api_base, model, temperature, api_key) for the vision role."""
    from universal_agentic_framework.config import load_core_config

    config = load_core_config()
    vision_role = getattr(getattr(config.llm, "roles", None), "vision", None)
    if vision_role is None:
        raise ValueError("llm.roles.vision is not configured in the active profile.")
    provider = config.llm.get_role_provider("vision")
    api_base = str(provider.api_base or "").rstrip("/")
    model_name = config.llm.get_role_model_name("vision", "en")
    temp = provider.temperature if provider.temperature is not None else 0.2
    return api_base, model_name, str(temp), provider.api_key


def _build_request_payload(
    model: str,
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
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
    }
