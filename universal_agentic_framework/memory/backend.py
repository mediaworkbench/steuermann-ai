from __future__ import annotations

from typing import List, Optional, Protocol


class MemoryRecord:
    user_id: str
    text: str
    metadata: Optional[dict]

    def __init__(self, user_id: str, text: str, metadata: Optional[dict] = None):
        self.user_id = user_id
        self.text = text
        self.metadata = metadata or {}


class MemoryBackend(Protocol):
    def load(self, user_id: str, query: Optional[str] = None, top_k: int = 5) -> List[MemoryRecord]:
        ...

    def upsert(self, user_id: str, text: str, metadata: Optional[dict] = None) -> MemoryRecord:
        ...

    def clear(self, user_id: str) -> None:
        ...
