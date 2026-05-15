"""Mem0 OSS adapter: long-memory backend with Qdrant persistence.

SOURCE OF TRUTH OWNERSHIP:
- Primary: Mem0 + Qdrant vector store
- Adapter caches (for SDK transition robustness):
  - _metadata_cache: Record metadata keyed by memory ID
  - _text_cache: Full text keyed by memory ID
  - _owner_cache: User ID ownership keyed by memory ID
  - _rating_overrides: Manual rating adjustments for fallback

METADATA CONSISTENCY MODEL (CRITICAL):
- Write path: upsert() populates caches after Mem0 write succeeds
- Read path: load() uses Mem0 results; populates/uses caches for consistency
- Invalidation: Caches cleared on delete(); NOT automatically invalidated on reads
- Fallback: If Mem0 SDK semantics change, caches provide bridge during transition

DIGEST CHAIN HANDLING:
- Input: update_memory_node() passes digest_chain (list of digest dicts)
- Storage: Digest metadata embedded in memory record metadata field
- Output: load() retrieves and returns with digest metadata intact
- Validation: Unit test required to verify digest persists end-to-end (checkpoint #7)

RATING SIGNAL FLOW:
- rate() stores override + emits retrieval feedback signal
- Signal consumed by metrics.track_memory_rated_after_retrieval()
- Analytics endpoint calculates coverage: (rated / retrieved) over time window

KNOWN ISSUES (to address in Phase 3):
- 4 separate maps without locking; potential race condition under high concurrency
- Unbounded cache growth; no eviction policy yet (planned: LRU or bounded)
- Missing PostgreSQL co_occurrence_edges table for knowledge graph persistence

See: docs/technical_architecture.md (Memory Architecture) for full context
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import structlog

from universal_agentic_framework.embeddings import normalize_embedding_model_name

from .backend import MemoryBackend, MemoryRecord
from .importance import MemoryImportanceScorer
from .linking import MemoryCoOccurrenceTracker


logger = structlog.get_logger(__name__)


class Mem0MemoryBackend(MemoryBackend):
    """Mem0 OSS adapter in embedded mode with Qdrant vector storage.

    This adapter preserves the repository's memory contract and metadata shape,
    while delegating storage/search operations to Mem0.
    """

    # Keep Mem0 extraction prompts bounded for local models with smaller context windows.
    _INFER_MAX_TOTAL_CHARS = 2800
    _INFER_MAX_MESSAGE_CHARS = 900
    _TRUNCATION_SUFFIX = "\n[truncated]"

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
        llm_provider: str = "openai",
        infer_enabled: bool = True,
        search_limit: int = 10,
        custom_instructions: Optional[str] = None,
        enable_importance_scoring: bool = True,
        enable_co_occurrence_tracking: bool = True,
        client: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        self.collection_name = f"{collection_prefix}_memory"
        self.search_limit = max(1, int(search_limit))
        self.infer_enabled = bool(infer_enabled)
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

        normalized_embedding_model = normalize_embedding_model_name(embedding_model)

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
                    "model": normalized_embedding_model,
                },
            },
        }

        effective_api_key = llm_api_key or "not-needed"

        if llm_provider == "lmstudio":
            # LM Studio requires json_schema instead of json_object for response_format.
            # Strip the "openai/" LiteLLM prefix that the main LLM stack uses — LM Studio
            # expects the raw model name (e.g. "liquid/lfm2-24b-a2b").
            lmstudio_model = llm_model.removeprefix("openai/")
            config["llm"] = {
                "provider": "lmstudio",
                "config": {
                    "model": lmstudio_model,
                    "temperature": float(llm_temperature),
                    "lmstudio_base_url": llm_api_base or "http://localhost:1234/v1",
                    "lmstudio_response_format": {
                        "type": "json_schema",
                        "json_schema": {"type": "object", "schema": {}},
                    },
                },
            }
        else:
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "temperature": float(llm_temperature),
                    "api_key": effective_api_key,
                },
            }

        if llm_max_tokens is not None:
            config["llm"]["config"]["max_tokens"] = int(llm_max_tokens)

        if embedding_remote_endpoint:
            config["embedder"]["config"]["openai_base_url"] = embedding_remote_endpoint

        config["embedder"]["config"]["api_key"] = effective_api_key
        if llm_provider != "lmstudio":
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

    def _stringify_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_text = item.get("text") or item.get("content") or item.get("value") or ""
                    if item_text:
                        parts.append(str(item_text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or content.get("value") or "")
        return str(content)

    def _normalize_message_payload(self, messages: Optional[List[Dict[str, Any]]], text: str) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = self._stringify_message_content(item.get("content"))
            normalized.append({"role": role, "content": content})
        if normalized:
            return normalized
        return [{"role": "user", "content": text}]

    def _truncate_for_infer(self, content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        clip = max(0, limit - len(self._TRUNCATION_SUFFIX))
        return (content[:clip]).rstrip() + self._TRUNCATION_SUFFIX

    def _build_infer_payload(self, messages: List[Dict[str, str]], summary_text: str) -> List[Dict[str, str]]:
        """Compact conversational payload to stay within local model context constraints."""
        trimmed: List[Dict[str, str]] = []
        remaining = self._INFER_MAX_TOTAL_CHARS

        # Preserve recency by walking backwards and prepending accepted items.
        for item in reversed(messages):
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            if not content:
                continue

            if remaining <= 0:
                break

            per_message_limit = min(self._INFER_MAX_MESSAGE_CHARS, remaining)
            clipped = self._truncate_for_infer(content, per_message_limit)
            if not clipped:
                continue

            trimmed.insert(0, {"role": role, "content": clipped})
            remaining -= len(clipped)

        if trimmed:
            return trimmed

        # Always provide at least the summary text when message payload is empty/fully trimmed.
        return [{"role": "user", "content": self._truncate_for_infer(summary_text, self._INFER_MAX_MESSAGE_CHARS)}]

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
        message_payload = self._normalize_message_payload(messages, text)
        infer_payload = self._build_infer_payload(message_payload, text)

        add_response = None
        if self.infer_enabled:
            try:
                # infer=True: Mem0's LLM extracts, deduplicates and merges facts automatically.
                add_response = self._memory.add(
                    infer_payload,
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
        else:
            logger.info("mem0_infer_disabled_storing_verbatim", user_id=user_id)

        # Detect silent inference failure: Mem0 swallows LLM errors internally (e.g. when
        # the local model rejects `response_format=json_object`) and returns an empty result
        # without raising. Treat any empty response as a failed inference and fall back.
        def _is_empty_response(r: Any) -> bool:
            if r is None:
                return True
            if isinstance(r, list) and len(r) == 0:
                return True
            if isinstance(r, dict):
                results = r.get("results")
                if results is not None and isinstance(results, list) and len(results) == 0:
                    return True
            return False

        if _is_empty_response(add_response):
            # Graceful degradation: store summary text verbatim without inference.
            if self.infer_enabled:
                logger.warning(
                    "mem0_infer_fallback_verbatim",
                    user_id=user_id,
                    reason="empty_or_failed_inference",
                )
            else:
                logger.info(
                    "mem0_infer_fallback_verbatim",
                    user_id=user_id,
                    reason="infer_disabled",
                )
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
