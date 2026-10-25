"""Mock adapters for the vision tools' streamed image fetch (``vision_utils.fetch_image_url``).

The tests build a plain ``Mock`` response (``.content`` + ``.headers``); these wrap it as
the ``client.stream(...)`` context manager the fetch helper consumes.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _prime(response):
    response.is_redirect = False
    if not isinstance(getattr(response, "headers", None), dict):
        response.headers = {}  # a real response always carries headers
    body = response.content
    response.iter_bytes = lambda: iter([body])

    async def _aiter():
        yield body

    response.aiter_bytes = _aiter
    return response


def as_stream(response) -> MagicMock:
    """``httpx.Client.stream`` return value yielding ``response``."""
    cm = MagicMock()
    cm.__enter__.return_value = _prime(response)
    cm.__exit__.return_value = False
    return cm


def as_astream(response) -> MagicMock:
    """``httpx.AsyncClient.stream`` return value yielding ``response``."""
    cm = MagicMock()
    cm.__aenter__.return_value = _prime(response)
    cm.__aexit__.return_value = False
    return cm
