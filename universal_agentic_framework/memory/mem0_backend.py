from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import structlog

from .backend import MemoryBackend, MemoryRecord
from .importance import MemoryImportanceScorer
from .linking import MemoryCoOccurrenceTracker


logger = structlog.get_logger(__name__)


class Mem0MemoryBackend(MemoryBackend):
    """Mem0 OSS adapter in embedded mode with Qdrant vector storage.

    This adapter preserves the repository's memory contract and metadata shape,
    while delegating storage/search operations to Mem0.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection_prefix: str,
        embedding_model: str,
        dimension: int,
        embedding_remote_endpoint: Optional[str],
        llm_model: str,
        llm_api_base: Optional[str],
        llm_temperature: float,
        llm_max_tokens: Optional[int],
        llm_api_key: Optional[str],
        search_limit: int = 10,
        custom_instructions: Optional[str] = None,
        enable_importance_scoring: bool = True,
        enable_co_occurrence_tracking: bool = True,
        client: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        self.collection_name = f"{collection_prefix}_memory"
        self.search_limit = max(1, int(search_limit))
        self._client_override = client
        self._embedder_override = embedder

        self._importance_scorer = MemoryImportanceScorer() if enable_importance_scoring else None
        self._co_occurrence_tracker = (
            MemoryCoOccurrenceTracker() if enable_co_occurrence_tracking else None
        )

        # Cache metadata/rating for robust compatibility when SDK update semantics differ.
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._text_cache: Dict[str, str] = {}
        self._owner_cache: Dict[str, str] = {}
        self._rating_overrides: Dict[str, int] = {}

        config: Dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": host,
                    "port": int(port),
                    "collection_name": self.collection_name,
                    "embedding_model_dims": int(dimension),
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "temperature": float(llm_temperature),
                },
            },
        }

        if llm_max_tokens is not None:
            config["llm"]["config"]["max_tokens"] = int(llm_max_tokens)

        if embedding_remote_endpoint:
            config["embedder"]["config"]["openai_base_url"] = embedding_remote_endpoint

        if llm_api_base:
            config["llm"]["config"]["openai_base_url"] = llm_api_base

        # OpenAI-compatible clients require an api_key value even when local
        # providers ignore it. Use a non-empty fallback when not configured.
        effective_api_key = llm_api_key or "not-needed"
        config["embedder"]["config"]["api_key"] = effective_api_key
        config["llm"]["config"]["api_key"] = effective_api_key

        if custom_instructions:
            config["custom_instructions"] = custom_instructions

        if self._client_override is not None:
            self._memory = self._client_override
        else:
            try:
                from mem0 import Memory

                self._memory = Memory.from_config(config)
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Mem0 memory backend: {exc}") from exc

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _extract_results(self, response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, dict):
            raw = response.get("results")
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
            return []
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        return []

    def _merge_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(metadata)

        if memory_id in self._metadata_cache:
            merged = {**self._metadata_cache[memory_id], **merged}

        if "memory_id" not in merged:
            merged["memory_id"] = memory_id

        if "created_at" not in merged:
            merged["created_at"] = self._now_iso()

        if "access_count" not in merged:
            merged["access_count"] = 0

        if memory_id in self._rating_overrides:
            merged["user_rating"] = int(self._rating_overrides[memory_id])

        self._metadata_cache[memory_id] = dict(merged)
        return merged

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        memory_id = str(item.get("id") or item.get("memory_id") or uuid.uuid4().hex[:12])
        text = str(item.get("memory") or item.get("text") or item.get("data") or "")

        metadata = dict(item.get("metadata") or {})
        if item.get("created_at") and "created_at" not in metadata:
            metadata["created_at"] = item.get("created_at")
        if item.get("updated_at") and "updated_at" not in metadata:
            metadata["last_accessed"] = item.get("updated_at")

        # Canonical contract ID is the provider ID returned by Mem0.
        metadata["memory_id"] = memory_id
        metadata = self._merge_metadata(memory_id, metadata)

        raw_score = item.get("score", 1.0)
        try:
            score = float(raw_score) if raw_score is not None else 1.0
        except (TypeError, ValueError):
            score = 1.0

        return {
            "memory_id": memory_id,
            "text": text,
            "metadata": metadata,
            "score": score,
            "user_id": item.get("user_id"),
        }

    def _fetch_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._memory.get(memory_id=memory_id)
        except TypeError:
            try:
                response = self._memory.get(memory_id)
            except Exception:
                response = None
        except Exception:
            response = None

        if isinstance(response, dict):
            if "id" not in response:
                response["id"] = memory_id
            return self._normalize_item(response)

        return None

    def _fetch_memories_by_ids(self, user_id: str, memory_ids: List[str]) -> List[MemoryRecord]:
        out: List[MemoryRecord] = []
        for memory_id in memory_ids:
            normalized = self._fetch_by_id(memory_id)
            if normalized is None:
                continue
            self._owner_cache[memory_id] = user_id
            out.append(
                MemoryRecord(
                    user_id=user_id,
                    text=normalized["text"],
                    metadata=dict(normalized["metadata"], is_related=True),
                )
            )
        return out

    def _search_memories(self, query: str, user_id: str, limit: int) -> Any:
        """Search using Mem0 v3 filters API with backward-compatible fallback."""
        try:
            return self._memory.search(query, filters={"user_id": user_id}, top_k=limit)
        except TypeError:
            # Older Mem0 signatures accepted entity ids as top-level kwargs.
            try:
                return self._memory.search(query, user_id=user_id, top_k=limit)
            except TypeError:
                return self._memory.search(query, user_id=user_id, limit=limit)
        except ValueError as exc:
            # Guard against partial upgrades where top-level entity kwargs are rejected.
            if "Top-level entity parameters" in str(exc):
                logger.warning("mem0_search_requires_filters", error=str(exc))
            raise

    def _get_all_memories(self, user_id: str, limit: int) -> Any:
        """Get all memories using Mem0 v3 filters API with backward-compatible fallback."""
        try:
            return self._memory.get_all(filters={"user_id": user_id}, limit=limit)
        except TypeError:
            try:
                return self._memory.get_all(user_id=user_id, top_k=limit)
            except TypeError:
                return self._memory.get_all(user_id=user_id, limit=limit)
        except ValueError as exc:
            if "Top-level entity parameters" in str(exc):
                logger.warning("mem0_get_all_requires_filters", error=str(exc))
            raise

    def _delete_all_memories(self, user_id: str) -> None:
        """Delete all memories using Mem0 v3 filters API with backward-compatible fallback."""
        try:
            self._memory.delete_all(filters={"user_id": user_id})
        except TypeError:
            self._memory.delete_all(user_id=user_id)
        except ValueError as exc:
            if "Top-level entity parameters" in str(exc):
                logger.warning("mem0_delete_all_requires_filters", error=str(exc))
            raise

    def load(
        self,
        user_id: str,
        query: Optional[str] = None,
        top_k: int = 5,
        include_related: bool = False,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        records: List[Dict[str, Any]] = []

        if query:
            response = self._search_memories(query, user_id=user_id, limit=max(top_k * 2, self.search_limit))
            records = [self._normalize_item(item) for item in self._extract_results(response)]
            if not records:
                # Retrieval fallback: when semantic search returns no hits,
                # return recent memories to maintain conversational continuity.
                logger.info(
                    "mem0_search_no_hits_fallback_to_recent",
                    user_id=user_id,
                    top_k=top_k,
                    search_limit=self.search_limit,
                )
                recent_response = self._get_all_memories(
                    user_id=user_id,
                    limit=max(top_k * 2, self.search_limit),
                )
                records = [self._normalize_item(item) for item in self._extract_results(recent_response)]
                records.sort(
                    key=lambda item: str(item["metadata"].get("created_at") or ""),
                    reverse=True,
                )
        else:
            response = self._get_all_memories(user_id=user_id, limit=max(top_k * 2, self.search_limit))
            records = [self._normalize_item(item) for item in self._extract_results(response)]
            records.sort(
                key=lambda item: str(item["metadata"].get("created_at") or ""),
                reverse=True,
            )

        if self._importance_scorer:
            ranked = self._importance_scorer.rank_memories(records, min_score=0.0)
        else:
            ranked = records

        primary = ranked[:top_k]
        out: List[MemoryRecord] = []

        for item in primary:
            metadata = dict(item["metadata"])
            if self._importance_scorer:
                metadata = self._importance_scorer.update_access_metadata(metadata)
                metadata["importance_score"] = float(item.get("importance", item.get("score", 0.0)))
            memory_id = str(metadata["memory_id"])
            self._metadata_cache[memory_id] = dict(metadata)
            self._text_cache[memory_id] = item["text"]
            self._owner_cache[memory_id] = user_id
            out.append(MemoryRecord(user_id=user_id, text=item["text"], metadata=metadata))

        if include_related and self._co_occurrence_tracker and len(out) > 1:
            active_session = session_id or user_id
            primary_ids = [m.metadata.get("memory_id") for m in out if m.metadata.get("memory_id")]
            self._co_occurrence_tracker.record_co_occurrence(primary_ids, active_session)

            related_ids: Set[str] = set()
            for primary_id in primary_ids:
                related = self._co_occurrence_tracker.get_related_memories(primary_id, top_k=5)
                for item in related:
                    related_id = item["memory_id"]
                    if related_id not in primary_ids:
                        related_ids.add(related_id)

            if related_ids:
                out.extend(self._fetch_memories_by_ids(user_id, list(related_ids)))

        return out

    def upsert(self, user_id: str, text: str, metadata: Optional[dict] = None, messages: Optional[List[Dict[str, Any]]] = None) -> MemoryRecord:
        payload = dict(metadata or {})
        payload.setdefault("created_at", self._now_iso())
        payload.setdefault("access_count", 0)

        # Use structured exchange messages when provided (richer context for Mem0 inference),
        # otherwise wrap the summary text as a single user message.
        message_payload = messages if messages else [{"role": "user", "content": text}]

        add_response = None
        try:
            # infer=True: Mem0's LLM extracts, deduplicates and merges facts automatically.
            add_response = self._memory.add(
                message_payload,
                user_id=user_id,
                metadata=payload,
                infer=True,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "mem0_infer_failed",
                error=str(exc),
                user_id=user_id,
                exc_info=True,
            )
        except Exception as exc:
            _err = str(exc).lower()
            if any(kw in _err for kw in ("json", "parse", "decode", "extract", "format", "invalid")):
                logger.warning("mem0_infer_failed", error=str(exc), user_id=user_id)
            else:
                raise

        if add_response is None:
            # Graceful degradation: store summary text verbatim without inference.
            logger.info("mem0_infer_fallback_verbatim", user_id=user_id)
            verbatim_payload = [{"role": "user", "content": text}]
            try:
                add_response = self._memory.add(verbatim_payload, user_id=user_id, metadata=payload, infer=False)
            except Exception as exc:
                logger.error("mem0_verbatim_fallback_failed", error=str(exc), user_id=user_id)
                add_response = {}

        memory_id = uuid.uuid4().hex[:12]
        if isinstance(add_response, dict):
            if isinstance(add_response.get("id"), str):
                memory_id = add_response["id"]
            elif isinstance(add_response.get("memory_id"), str):
                memory_id = add_response["memory_id"]
            elif isinstance(add_response.get("memory_ids"), list) and add_response.get("memory_ids"):
                first_id = add_response.get("memory_ids")[0]
                if isinstance(first_id, str):
                    memory_id = first_id
            else:
                results = add_response.get("results")
                if isinstance(results, list) and results:
                    first = results[0]
                    if isinstance(first, dict) and isinstance(first.get("id"), str):
                        memory_id = first["id"]

        payload["memory_id"] = memory_id
        self._metadata_cache[memory_id] = dict(payload)
        self._text_cache[memory_id] = text
        self._owner_cache[memory_id] = user_id

        return MemoryRecord(user_id=user_id, text=text, metadata=payload)

    def clear(self, user_id: str) -> None:
        response = self._get_all_memories(user_id=user_id, limit=10_000)
        memory_ids = [
            self._normalize_item(item)["memory_id"]
            for item in self._extract_results(response)
        ]

        self._delete_all_memories(user_id=user_id)

        for memory_id in memory_ids:
            self._metadata_cache.pop(memory_id, None)
            self._text_cache.pop(memory_id, None)
            self._owner_cache.pop(memory_id, None)
            self._rating_overrides.pop(memory_id, None)

    def find_memory_point(self, memory_id: str) -> Optional[dict[str, Any]]:
        normalized = self._fetch_by_id(memory_id)
        if normalized is None:
            return None

        owner_user_id = normalized.get("user_id") or self._owner_cache.get(memory_id)
        return {
            "point_id": normalized["memory_id"],
            "payload": {
                "user_id": owner_user_id,
                "text": normalized["text"],
                "metadata": normalized["metadata"],
            },
        }

    def delete_memory(self, *, memory_id: str, user_id: str) -> None:
        point = self.find_memory_point(memory_id)
        if point is None:
            return

        payload = point.get("payload") or {}
        owner_user_id = payload.get("user_id")
        if owner_user_id and owner_user_id != user_id:
            raise PermissionError("Memory does not belong to user")

        try:
            self._memory.delete(memory_id=memory_id)
        except TypeError:
            self._memory.delete(memory_id)

        self._metadata_cache.pop(memory_id, None)
        self._text_cache.pop(memory_id, None)
        self._owner_cache.pop(memory_id, None)
        self._rating_overrides.pop(memory_id, None)

    def set_memory_user_rating(
        self,
        *,
        point_id: Any,
        metadata: Optional[dict[str, Any]],
        rating: int,
    ) -> None:
        memory_id = str(point_id)
        updated = dict(metadata or {})
        updated["user_rating"] = int(rating)

        self._rating_overrides[memory_id] = int(rating)
        self._metadata_cache[memory_id] = self._merge_metadata(memory_id, updated)

        text = self._text_cache.get(memory_id)
        if not text:
            fetched = self._fetch_by_id(memory_id)
            if fetched:
                text = fetched["text"]

        if not text:
            return

        # Persist rating metadata across Mem0 OSS SDK signature variations.
        update_attempts = [
            {"memory_id": memory_id, "data": text, "metadata": updated},
            {"memory_id": memory_id, "new_memory": text, "metadata": updated},
            {"memory_id": memory_id, "metadata": updated},
            {"memory_id": memory_id, "data": text},
            {"memory_id": memory_id, "new_memory": text},
        ]

        for kwargs in update_attempts:
            try:
                self._memory.update(**kwargs)
                return
            except TypeError:
                continue
            except Exception:
                continue

        logger.warning("mem0_rating_persist_failed", memory_id=memory_id)
