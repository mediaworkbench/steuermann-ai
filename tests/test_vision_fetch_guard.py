"""SSRF guard + streamed byte cap for the vision tools' remote image fetch."""
from __future__ import annotations

import httpx
import pytest

from universal_agentic_framework.tools import vision_utils
from universal_agentic_framework.tools.vision_utils import (
    ImageTooLargeError,
    afetch_image_url,
    fetch_image_url,
)

_DNS = {
    "cdn.example.com": ["93.184.216.34"],
    "qdrant": ["172.18.0.5"],
    "metadata.internal": ["169.254.169.254"],
    "rebind.example.com": ["93.184.216.34", "127.0.0.1"],
}


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    monkeypatch.setattr(vision_utils, "_resolve_host_ips", lambda host, port: _DNS[host])


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _png(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})


def test_public_url_is_fetched():
    with _client(_png) as client:
        assert fetch_image_url(client, "https://cdn.example.com/a.png", 1024) == (b"\x89PNG", "image/png")


@pytest.mark.parametrize(
    "url",
    [
        "http://qdrant:6333/collections",
        "http://metadata.internal/latest/meta-data",
        "http://rebind.example.com/a.png",  # any non-public address taints the host
        "file:///etc/passwd",
        "ftp://cdn.example.com/a.png",
    ],
)
def test_non_public_or_non_http_urls_are_refused(url):
    calls = []
    with _client(lambda r: calls.append(r) or _png(r)) as client:
        with pytest.raises(ValueError):
            fetch_image_url(client, url, 1024)
    assert calls == []  # refused before any request is sent


def test_redirect_to_internal_host_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://qdrant:6333/"})

    with _client(handler) as client:
        with pytest.raises(ValueError, match="non-public"):
            fetch_image_url(client, "https://cdn.example.com/a.png", 1024)


def test_relative_redirect_is_followed_and_revalidated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old.png":
            return httpx.Response(301, headers={"location": "/new.png"})
        return _png(request)

    with _client(handler) as client:
        assert fetch_image_url(client, "https://cdn.example.com/old.png", 1024)[0] == b"\x89PNG"


def test_redirect_loop_is_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://cdn.example.com/again"})

    with _client(handler) as client:
        with pytest.raises(ValueError, match="Too many redirects"):
            fetch_image_url(client, "https://cdn.example.com/a.png", 1024)


def test_declared_oversize_body_is_rejected_up_front():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 2048, headers={"content-type": "image/png"})

    with _client(handler) as client:
        with pytest.raises(ImageTooLargeError, match="too large"):
            fetch_image_url(client, "https://cdn.example.com/big.png", 1024)


def test_undeclared_oversize_stream_is_cut_off():
    def chunks():
        for _ in range(100):
            yield b"x" * 512

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks(), headers={"content-type": "image/png"})

    with _client(handler) as client:
        with pytest.raises(ImageTooLargeError):
            fetch_image_url(client, "https://cdn.example.com/endless.png", 1024)


@pytest.mark.asyncio
async def test_async_fetch_applies_the_same_guard():
    transport = httpx.MockTransport(_png)
    async with httpx.AsyncClient(transport=transport) as client:
        assert await afetch_image_url(client, "https://cdn.example.com/a.png", 1024) == (b"\x89PNG", "image/png")
        with pytest.raises(ValueError, match="non-public"):
            await afetch_image_url(client, "http://qdrant:6333/", 1024)
