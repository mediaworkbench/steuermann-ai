"""Bounded reads for multipart uploads.

``await UploadFile.read()`` pulls the whole (disk-spooled) part into memory before
any size check can run; reading in chunks lets an oversized upload be rejected as
soon as it crosses the limit.
"""
from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

_CHUNK_BYTES = 1024 * 1024


async def read_upload_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read ``file`` fully, raising 413 once it exceeds ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {max_bytes:,} bytes).",
            )
        chunks.append(chunk)
    return b"".join(chunks)
