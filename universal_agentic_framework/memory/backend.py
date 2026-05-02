from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class MemoryRatingBackend(Protocol):
    def find_memory_point(self, memory_id: str) -> Optional[dict[str, Any]]:
        ...

    def set_memory_user_rating(
        self,
        *,
        point_id: Any,
        metadata: Optional[dict[str, Any]],
        rating: int,
    ) -> None:
        ...


@runtime_checkable
class MemoryDeleteBackend(Protocol):
    def delete_memory(self, *, memory_id: str, user_id: str) -> None:
        ...
