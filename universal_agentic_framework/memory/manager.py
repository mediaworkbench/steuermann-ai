"""Minimal in-memory memory manager stub. Long-term semantic memory is handled by Mem0MemoryBackend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MemoryRecord:
    user_id: str
    text: str
    metadata: Optional[dict] = None


class InMemoryMemoryManager:
    def __init__(self):
        self._store: Dict[str, List[MemoryRecord]] = {}

    def load(self, user_id: str, query: Optional[str] = None, top_k: int = 5) -> List[MemoryRecord]:
        records = self._store.get(user_id, [])
        if query:
            filtered = [r for r in records if query.lower() in r.text.lower()]
        else:
            filtered = records
        return filtered[:top_k]

    def upsert(
        self,
        user_id: str,
        text: str,
        metadata: Optional[dict] = None,
        messages: Optional[list] = None,
        digest_chain: Optional[list[dict]] = None,
    ) -> MemoryRecord:
        payload = dict(metadata or {})
        if digest_chain:
            payload.setdefault("digest_chain", digest_chain)
            payload.setdefault(
                "digest_chain_ids",
                [d.get("digest_id") for d in digest_chain if isinstance(d, dict) and d.get("digest_id")],
            )
            payload.setdefault("digest_chain_length", len(digest_chain))
        rec = MemoryRecord(user_id=user_id, text=text, metadata=payload)
        self._store.setdefault(user_id, []).append(rec)
        return rec

    def clear(self, user_id: str) -> None:
        self._store.pop(user_id, None)
