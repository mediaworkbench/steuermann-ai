from __future__ import annotations

import uuid
import time
from datetime import datetime, timezone
from typing import Any, List, Optional, Set

from .backend import MemoryBackend, MemoryRecord
from .importance import MemoryImportanceScorer
from .linking import MemoryCoOccurrenceTracker
from universal_agentic_framework.embeddings import build_embedding_provider, EmbeddingProvider


class QdrantMemoryBackend(MemoryBackend):
    """Qdrant-backed memory store with remote API-based embeddings.

    Designed for dependency injection: you can pass a preconfigured client and
    embedder for testing. If not provided, the backend will construct them from
    the given parameters.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_prefix: str = "framework",
        embedding_model: Optional[str] = None,
        client: Optional[object] = None,
        embedder: Optional[EmbeddingProvider] = None,
        distance: str = "Cosine",
        dimension: Optional[int] = None,
        enable_importance_scoring: bool = True,
        enable_co_occurrence_tracking: bool = True,
        fork_name: str = "default",
        embedding_provider_type: str = "remote",
        embedding_remote_endpoint: Optional[str] = None,
    ) -> None:
        self.collection_name = f"{collection_prefix}_memory"
        self.enable_importance_scoring = enable_importance_scoring
        self.enable_co_occurrence_tracking = enable_co_occurrence_tracking
        self.fork_name = fork_name
        
        # Store configured host/port for REST API calls (QdrantClient doesn't expose these)
        self._host = host or "localhost"
        self._port = port or 6333

        # Build client lazily if not provided
        self._client = client or self._build_client(host, port)
        # Build embedder lazily if not provided (now using EmbeddingProvider abstraction)
        self._embedder = embedder or self._build_embedder(
            embedding_model,
            embedding_provider_type,
            embedding_remote_endpoint,
            dimension=dimension,
        )
        
        # Initialize importance scorer if enabled
        if self.enable_importance_scoring:
            self._importance_scorer = MemoryImportanceScorer()
        else:
            self._importance_scorer = None
        
        # Initialize co-occurrence tracker if enabled
        if self.enable_co_occurrence_tracking:
            self._co_occurrence_tracker = MemoryCoOccurrenceTracker()
        else:
            self._co_occurrence_tracker = None

        # Ensure collection exists
        if dimension is None:
            # Try to infer dimension from the model by encoding a dummy string
            try:
                test_vec = self._embedder.encode(["dim"])[0]
                dimension = len(test_vec)
            except Exception:
                dimension = 384  # sensible default
        self._ensure_collection(self.collection_name, dimension=dimension, distance=distance)

    # --- helpers ------------------------------------------------------------
    def _build_client(self, host: Optional[str], port: Optional[int]):
        from qdrant_client import QdrantClient

        if host and port:
            return QdrantClient(host=host, port=port, check_compatibility=False)
        return QdrantClient(check_compatibility=False)

    def _build_embedder(
        self,
        model_name: Optional[str],
        provider_type: str = "remote",
        remote_endpoint: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> EmbeddingProvider:
        """Build embedding provider (remote API-based)."""
        model = model_name or "text-embedding-granite-embedding-278m-multilingual"
        dimension = dimension or 768  # Granite default
        return build_embedding_provider(
            model_name=model,
            dimension=dimension,
            provider_type=provider_type,
            remote_endpoint=remote_endpoint,
        )

    def _to_vector_list(self, vector) -> list[float]:
        """Normalize embedding vectors to plain Python list for Qdrant/JSON APIs."""
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)

    def _ensure_collection(self, name: str, *, dimension: int, distance: str):
        """Ensure collection exists. Try multiple approaches for compatibility."""
        import json
        import httpx
        import structlog
        logger = structlog.get_logger()
        
        # Use configured host/port (not from client, which doesn't expose them)
        url = f"http://{self._host}:{self._port}"
        
        try:
            # Try REST API first to check if collection exists
            check_url = f"{url}/collections/{name}"
            resp = httpx.get(check_url, timeout=5.0)
            if resp.status_code == 200:
                # Collection exists
                logger.info("Collection exists", name=name, status=resp.status_code)
                return
            logger.info("Collection check failed (REST API)", name=name, status=resp.status_code)
        except Exception as e:
            logger.info("Collection check exception", name=name, error=str(e))
        
        # Use QdrantClient to create collection (more reliable than REST API)
        try:
            from qdrant_client.models import Distance, VectorParams

            # Create collection with proper distance metric
            logger.info("Creating collection via QdrantClient", name=name, dimension=dimension, distance=distance)
            
            try:
                # Try to get collection first
                self._client.get_collection(name)
                logger.info("Collection exists via QdrantClient", name=name)
            except Exception as get_err:
                # Create if doesn't exist
                logger.info("Collection get_collection failed (expected), attempting create", name=name, error=str(get_err)[:100])
                try:
                    # Use COSINE as default distance metric (enum name differs by qdrant-client version)
                    distance_value = getattr(Distance, "COSINE", None) or getattr(Distance, "Cosine")
                    params = VectorParams(size=dimension, distance=distance_value)
                    logger.info("VectorParams created", size=dimension, distance_type="Cosine")
                    self._client.create_collection(collection_name=name, vectors_config=params)
                    logger.info("Collection created successfully via QdrantClient", name=name, dimension=dimension)
                except Exception as create_err:
                    logger.error("Failed to create collection", name=name, error=str(create_err)[:200], exc_info=True)
                    raise
        except Exception as e:
            logger.error("QdrantClient collection operation failed", name=name, error=str(e)[:200], exc_info=True)

    # --- API ----------------------------------------------------------------
    def load(
        self,
        user_id: str,
        query: Optional[str] = None,
        top_k: int = 5,
        include_related: bool = False,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Load memories from Qdrant with optional related memory expansion.
        
        Args:
            user_id: User identifier
            query: Search query (if None, returns latest memories)
            top_k: Number of primary memories to retrieve
            include_related: If True, fetch related memories via co-occurrence graph
            session_id: Session identifier for tracking co-occurrences
            
        Returns:
            List of memory records (primary + related if enabled)
        """
        if not query:
            # If no query, return latest entries by filtering payload; fake clients may handle this
            try:
                # Attempt to scroll all points and filter by user_id
                points, _ = self._client.scroll(self.collection_name, limit=top_k)
                out: List[MemoryRecord] = []
                for p in points:
                    payload = getattr(p, "payload", {})
                    if payload.get("user_id") == user_id:
                        # Update access metadata
                        metadata = payload.get("metadata", {})
                        if self._importance_scorer:
                            metadata = self._importance_scorer.update_access_metadata(metadata)
                        out.append(MemoryRecord(user_id=user_id, text=payload.get("text", ""), metadata=metadata))
                        if len(out) >= top_k:
                            break
                return out
            except Exception:
                return []

        # With query: embed and search with filter
        import structlog
        logger = structlog.get_logger()
        
        vector = self._to_vector_list(self._embedder.encode([query])[0])
        logger.info("Memory search started", user_id=user_id, query=query[:50], collection=self.collection_name)

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
            logger.info("Attempting QdrantClient search", user_id=user_id, collection=self.collection_name)
            
            # Use the correct Qdrant API - it's 'query_points' in newer versions or 'search' via REST
            try:
                query_response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=top_k * 2,
                    query_filter=flt
                )
                results = getattr(query_response, "points", None)
                if results is None or not isinstance(results, (list, tuple)):
                    raise TypeError("query_points returned unsupported response shape")
            except AttributeError:
                # Fallback to legacy Qdrant client API first.
                try:
                    results = self._client.search(
                        collection_name=self.collection_name,
                        query_vector=vector,
                        limit=top_k * 2,
                        query_filter=flt,
                    )
                except Exception:
                    # Final fallback to REST API for compatibility.
                    import httpx
                    url = f"http://{self._host}:{self._port}/collections/{self.collection_name}/points/search"
                    response = httpx.post(
                        url,
                        json={
                            "vector": vector,
                            "limit": top_k * 2,
                            "filter": {
                                "must": [{"key": "user_id", "match": {"value": user_id}}]
                            },
                            "with_payload": True
                        },
                        timeout=10.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Convert REST API response to match QdrantClient format
                        from types import SimpleNamespace
                        results = []
                        for item in data.get("result", []):
                            result = SimpleNamespace(
                                score=item.get("score", 0.0),
                                payload=item.get("payload", {})
                            )
                            results.append(result)
                    else:
                        logger.error("REST API search failed", status=response.status_code, response=response.text[:200])
                        results = []
            except Exception:
                # Fallback to legacy Qdrant client API first.
                try:
                    results = self._client.search(
                        collection_name=self.collection_name,
                        query_vector=vector,
                        limit=top_k * 2,
                        query_filter=flt,
                    )
                except Exception:
                    # Final fallback to REST API for compatibility.
                    import httpx
                    url = f"http://{self._host}:{self._port}/collections/{self.collection_name}/points/search"
                    response = httpx.post(
                        url,
                        json={
                            "vector": vector,
                            "limit": top_k * 2,
                            "filter": {
                                "must": [{"key": "user_id", "match": {"value": user_id}}]
                            },
                            "with_payload": True
                        },
                        timeout=10.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Convert REST API response to match QdrantClient format
                        from types import SimpleNamespace
                        results = []
                        for item in data.get("result", []):
                            result = SimpleNamespace(
                                score=item.get("score", 0.0),
                                payload=item.get("payload", {})
                            )
                            results.append(result)
                    else:
                        logger.error("REST API search failed", status=response.status_code, response=response.text[:200])
                        results = []
            
            logger.info("QdrantClient search returned", user_id=user_id, result_count=len(results) if results else 0)
            
            # Build search results in format expected by importance scorer
            search_results = []
            for r in results:
                payload = getattr(r, "payload", {})
                search_results.append({
                    "text": payload.get("text", ""),
                    "score": getattr(r, "score", 0.0),
                    "metadata": payload.get("metadata", {}),
                    "payload": payload,  # Keep for later
                })
            
            # Apply importance scoring if enabled
            if self._importance_scorer and search_results:
                start_time = time.time()
                current_time = datetime.now(timezone.utc)
                ranked = self._importance_scorer.rank_memories(search_results, current_time, min_score=0.0)
                
                # Track importance ranking duration
                try:
                    from ..monitoring.metrics import track_memory_importance_ranking, track_memory_quality, track_memory_age
                    duration = time.time() - start_time
                    track_memory_importance_ranking(self.fork_name, duration)
                    
                    # Track quality scores and ages
                    for item in ranked[:top_k]:
                        if "importance_score" in item["metadata"]:
                            track_memory_quality(self.fork_name, item["metadata"]["importance_score"])
                        if "created_at" in item["metadata"]:
                            try:
                                created = datetime.fromisoformat(item["metadata"]["created_at"])
                                age_days = (current_time - created).total_seconds() / 86400
                                track_memory_age(self.fork_name, age_days)
                            except Exception:
                                pass
                except ImportError:
                    pass  # Metrics not available
                
                # Update access metadata for top results
                out: List[MemoryRecord] = []
                primary_memory_ids: List[str] = []
                for item in ranked[:top_k]:  # Take top_k after reranking
                    metadata = item["metadata"]
                    metadata = self._importance_scorer.update_access_metadata(metadata)
                    memory_id = metadata.get("memory_id")
                    if memory_id:
                        primary_memory_ids.append(memory_id)
                    out.append(MemoryRecord(user_id=user_id, text=item["text"], metadata=metadata))
                
                # Track co-occurrences and fetch related memories if enabled
                if include_related and self._co_occurrence_tracker and primary_memory_ids and session_id:
                    # Record co-occurrence of primary memories
                    self._co_occurrence_tracker.record_co_occurrence(
                        primary_memory_ids,
                        session_id=session_id,
                        timestamp=current_time
                    )
                    
                    # Fetch related memories
                    related_ids: Set[str] = set()
                    for memory_id in primary_memory_ids:
                        related = self._co_occurrence_tracker.get_related_memories(
                            memory_id,
                            current_time=current_time,
                            top_k=5,  # Max 5 related per primary memory
                        )
                        for item in related:
                            rel_id = item["memory_id"]
                            if rel_id not in primary_memory_ids:  # Don't duplicate primary memories
                                related_ids.add(rel_id)
                    
                    # Retrieve related memories from Qdrant
                    if related_ids:
                        related_memories = self._fetch_memories_by_ids(user_id, list(related_ids))
                        out.extend(related_memories)
                        
                        # Track metric
                        try:
                            from ..monitoring.metrics import track_related_memories
                            track_related_memories(self.fork_name, len(related_memories))
                        except ImportError:
                            pass
                    
                    # Update graph statistics
                    try:
                        from ..monitoring.metrics import track_memory_graph_statistics
                        stats = self._co_occurrence_tracker.get_graph_statistics()
                        track_memory_graph_statistics(self.fork_name, stats["node_count"], stats["edge_count"])
                    except ImportError:
                        pass
                
                return out
            else:
                # Fallback: return raw results without importance scoring
                out: List[MemoryRecord] = []
                for r in results[:top_k]:
                    payload = getattr(r, "payload", {})
                    out.append(MemoryRecord(user_id=user_id, text=payload.get("text", ""), metadata=payload.get("metadata", {})))
                return out
        except Exception as e:
            logger.error("Memory search failed", user_id=user_id, error=str(e), exc_info=True)
            return []

    def _fetch_memories_by_ids(self, user_id: str, memory_ids: List[str]) -> List[MemoryRecord]:
        """Fetch specific memories by their IDs.
        
        Args:
            user_id: User identifier
            memory_ids: List of memory IDs to fetch
            
        Returns:
            List of memory records
        """
        if not memory_ids:
            return []
        
        try:
            # Retrieve points by scrolling and filtering by memory_id
            # Note: Qdrant doesn't have a native "get by payload field" method,
            # so we scroll and filter in-memory (acceptable for small batches)
            points, _ = self._client.scroll(
                self.collection_name,
                scroll_filter=None,  # No filter, we'll filter in-memory
                limit=1000,  # Reasonable upper bound
            )
            
            results: List[MemoryRecord] = []
            for p in points:
                payload = getattr(p, "payload", {})
                if payload.get("user_id") == user_id:
                    metadata = payload.get("metadata", {})
                    memory_id = metadata.get("memory_id")
                    if memory_id and memory_id in memory_ids:
                        # Update access metadata
                        if self._importance_scorer:
                            metadata = self._importance_scorer.update_access_metadata(metadata)
                        results.append(MemoryRecord(user_id=user_id, text=payload.get("text", ""), metadata=metadata))
                        
                        # Stop if we found all requested memories
                        if len(results) >= len(memory_ids):
                            break
            
            return results
        except Exception:
            return []

    def upsert(self, user_id: str, text: str, metadata: Optional[dict] = None) -> MemoryRecord:
        """Upsert a memory record into Qdrant."""
        import structlog
        logger = structlog.get_logger()
        
        logger.info("Upsert started", user_id=user_id, text_length=len(text))
        
        vector = self._to_vector_list(self._embedder.encode([text])[0])
        logger.info("Vector encoded", vector_size=len(vector), vector_sample=vector[:3])
        
        # Generate point ID
        # For REST API: use full UUID string format
        # For QdrantClient: will use integer derived from UUID
        uuid_obj = uuid.uuid4()
        pid_uuid = str(uuid_obj)  # String UUID for REST API
        pid_int = int(uuid_obj.int % (2**63 - 1))  # Integer for QdrantClient (unsigned 64-bit)
        
        # Initialize metadata with importance scoring fields
        if metadata is None:
            metadata = {}
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        if "access_count" not in metadata:
            metadata["access_count"] = 0
        if "memory_id" not in metadata:
            metadata["memory_id"] = str(uuid.uuid4().hex[:12])  # Stable memory ID for co-occurrence tracking
        
        payload = {"user_id": user_id, "text": text, "metadata": metadata}
        logger.info("Payload prepared", payload_keys=list(payload.keys()), memory_id=metadata.get("memory_id"))

        # Use configured host/port (not from client, which doesn't expose them)
        url = f"http://{self._host}:{self._port}"
        logger.info("Qdrant URL configured", url=url, collection=self.collection_name)
        
        try:
            # Try REST API first (more reliable)
            import httpx
            upsert_url = f"{url}/collections/{self.collection_name}/points?wait=true"
            payload_data = {
                "points": [
                    {
                        "id": pid_uuid,  # Use UUID string for REST API
                        "vector": vector,
                        "payload": payload
                    }
                ]
            }
            logger.info("Attempting REST API upsert", url=upsert_url, point_id=pid_uuid)
            resp = httpx.put(upsert_url, json=payload_data, timeout=10.0)
            logger.info("REST API response", status_code=resp.status_code, response_text=resp.text[:200] if resp.text else "")
            
            if resp.status_code in (200, 201):
                logger.info("REST API upsert succeeded", status_code=resp.status_code)
                return MemoryRecord(user_id=user_id, text=text, metadata=metadata)
            # If 404, collection doesn't exist - create it
            if resp.status_code == 404:
                logger.info("Collection not found (404), creating it")
                self._ensure_collection(self.collection_name, dimension=len(vector), distance="Cosine")
                resp = httpx.put(upsert_url, json=payload_data, timeout=10.0)
                logger.info("Retry after collection creation", status_code=resp.status_code)
                if resp.status_code in (200, 201):
                    logger.info("REST API upsert succeeded after collection creation")
                    return MemoryRecord(user_id=user_id, text=text, metadata=metadata)
        except Exception as e:
            logger.warning("REST API upsert failed", error=str(e))

        # Fallback: try QdrantClient
        logger.info("Attempting QdrantClient upsert fallback")
        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(id=pid_int, vector=vector, payload=payload)  # Use integer ID for QdrantClient
            self._client.upsert(self.collection_name, points=[point])
            logger.info("QdrantClient upsert succeeded")
        except Exception as e:
            logger.error("QdrantClient upsert failed", error=str(e))
            # Last fallback
            try:
                self._client.upsert(self.collection_name, points=[{"id": pid_int, "vector": vector, "payload": payload}])
                logger.info("Last fallback upsert succeeded")
            except Exception as e2:
                logger.error("All upsert attempts failed", error=str(e2))

        logger.info("Upsert complete, returning MemoryRecord")
        return MemoryRecord(user_id=user_id, text=text, metadata=metadata)

    def clear(self, user_id: str) -> None:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
            self._client.delete(self.collection_name, query_filter=flt)
        except Exception:
            # Ignore if client doesn't support filtered delete
            pass

    def find_memory_point(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Find a memory point by stable metadata.memory_id.

        Returns a dict with point_id and payload when found.
        """
        if not memory_id:
            return None

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            flt = Filter(
                must=[
                    FieldCondition(key="metadata.memory_id", match=MatchValue(value=memory_id)),
                ]
            )
            points, _ = self._client.scroll(
                self.collection_name,
                scroll_filter=flt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                point = points[0]
                return {
                    "point_id": getattr(point, "id", None),
                    "payload": getattr(point, "payload", {}) or {},
                }
        except Exception:
            pass

        # Compatibility fallback for clients/backends that do not support filtered scroll.
        try:
            points, _ = self._client.scroll(self.collection_name, limit=2000, with_payload=True, with_vectors=False)
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                metadata = payload.get("metadata", {}) or {}
                if metadata.get("memory_id") == memory_id:
                    return {
                        "point_id": getattr(point, "id", None),
                        "payload": payload,
                    }
        except Exception:
            return None

        return None

    def set_memory_user_rating(
        self,
        *,
        point_id: Any,
        metadata: Optional[dict[str, Any]],
        rating: int,
    ) -> None:
        """Persist user_rating in metadata for an existing point."""
        updated_metadata = dict(metadata or {})
        updated_metadata["user_rating"] = int(rating)

        # Primary path: qdrant-client set_payload API.
        try:
            self._client.set_payload(
                collection_name=self.collection_name,
                payload={"metadata": updated_metadata},
                points=[point_id],
            )
            return
        except Exception:
            pass

        # Fallback path: REST API payload update.
        import httpx

        url = f"http://{self._host}:{self._port}/collections/{self.collection_name}/points/payload?wait=true"
        payload_data = {
            "payload": {"metadata": updated_metadata},
            "points": [point_id],
        }
        response = httpx.put(url, json=payload_data, timeout=10.0)
        response.raise_for_status()
