"""Vector database backend for semantic cache search using Qdrant.

Enables efficient similarity search for paraphrased queries instead of O(n) iteration.
Stores query embeddings and metadata for semantic cache lookups.
"""

from typing import Optional, List, Dict, Any
import logging
import httpx
import time

from universal_agentic_framework.monitoring import metrics
from universal_agentic_framework.embeddings import build_embedding_provider, EmbeddingProvider

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

logger = logging.getLogger(__name__)


class QdrantCacheVectorBackend:
    """Qdrant vector store for semantic cache search.
    
    Stores query embeddings alongside crew result cache entries.
    Enables efficient similarity search for paraphrased queries.
    
    Collections:
    - {collection_prefix}_cache: semantic query embeddings for cache lookups
    
    Attributes:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of Qdrant collection
        dimension: Vector dimension (default 384)
        similarity_threshold: Minimum similarity score for matches
    """
    
    def __init__(
        self,
        collection_prefix: str = "cache",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "",
        dimension: int = 768,
        similarity_threshold: float = 0.85,
        fork_name: str = "default",
        embedding_provider_type: str = "remote",
        embedding_remote_endpoint: Optional[str] = None,
    ) -> None:
        """Initialize Qdrant cache vector backend.
        
        Args:
            collection_prefix: Prefix for collection name
            host: Qdrant server host
            port: Qdrant server port
            embedding_model: Embedding model name
            dimension: Vector dimension (must match model output)
            similarity_threshold: Minimum cosine similarity for matches
            fork_name: Fork name for metrics/logging
            embedding_provider_type: Embedding provider type (remote-only)
            embedding_remote_endpoint: Required remote endpoint for embedding requests
        """
        if not HAS_QDRANT:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")
        
        self.collection_name = f"{collection_prefix}_cache"
        self.host = host
        self.port = port
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.fork_name = fork_name
        
        # Initialize client
        self._client = QdrantClient(
            host=host,
            port=port,
            timeout=5,
            check_compatibility=False,
        )
        
        # Initialize embedder using EmbeddingProvider abstraction
        self._embedder: Optional[EmbeddingProvider] = None
        try:
            self._embedder = build_embedding_provider(
                model_name=embedding_model,
                dimension=dimension,
                provider_type=embedding_provider_type,
                remote_endpoint=embedding_remote_endpoint,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embedding provider: {e}")
            self._embedder = None
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(
            f"Initialized QdrantCacheVectorBackend for {fork_name} "
            f"(collection={self.collection_name}, threshold={similarity_threshold})"
        )
    
    def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with proper schema.
        
        Uses REST API first (more reliable), then falls back to QdrantClient.
        Drops and recreates collection if dimensions don't match.
        """
        try:
            # Extract host/port from client
            host = getattr(self._client, 'host', self.host)
            port = getattr(self._client, 'port', self.port)
            url = f"http://{host}:{port}"
            
            # Check if collection exists first
            collection_exists = False
            collection_info = None
            
            try:
                # Try REST API first
                check_url = f"{url}/collections/{self.collection_name}"
                resp = httpx.get(check_url, timeout=5.0)
                if resp.status_code == 200:
                    collection_exists = True
                    collection_info = resp.json()
            except httpx.RequestError:
                # Try QdrantClient fallback
                try:
                    collection_info = self._client.get_collection(self.collection_name)
                    collection_exists = True
                except Exception:
                    collection_exists = False
            
            if collection_exists and collection_info:
                # Check if dimensions match
                try:
                    # Extract dimension from collection info
                    if isinstance(collection_info, dict):
                        current_dimension = collection_info.get("config", {}).get("params", {}).get("vectors", {}).get("size")
                    else:
                        # QdrantClient returns object with config attribute
                        current_dimension = getattr(collection_info.config.params.vectors, 'size', None)
                    
                    if current_dimension and current_dimension != self.dimension:
                        # Drop and recreate with correct dimensions
                        logger.warning(
                            f"Collection {self.collection_name} has dimension {current_dimension}, "
                            f"expected {self.dimension}. Dropping and recreating..."
                        )
                        try:
                            self._client.delete_collection(self.collection_name)
                            logger.info(f"Dropped collection {self.collection_name}")
                        except Exception as e:
                            logger.error(f"Failed to drop collection: {e}")
                        collection_exists = False
                    else:
                        logger.debug(f"Qdrant collection {self.collection_name} exists with correct dimension")
                        return
                except Exception as e:
                    logger.debug(f"Could not verify collection dimension: {e}")
                    # If we can't verify dimensions, assume it's OK
                    if collection_exists:
                        return
            
            # Create via REST API (more reliable)
            if not collection_exists:
                try:
                    create_url = f"{url}/collections/{self.collection_name}"
                    payload = {
                        "vectors": {
                            "size": self.dimension,
                            "distance": "Cosine"
                        }
                    }
                    resp = httpx.put(create_url, json=payload, timeout=10.0)
                    if resp.status_code in (200, 201):
                        logger.info(f"Created Qdrant collection via REST: {self.collection_name} (dim={self.dimension})")
                        return
                except httpx.RequestError as e:
                    logger.debug(f"REST API creation failed: {e}, trying QdrantClient")
                
                # Fallback: Create via QdrantClient
                try:
                    self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.dimension,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection via client: {self.collection_name} (dim={self.dimension})")
                except Exception as e:
                    logger.error(f"Failed to create collection via client: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            raise

    
    def store_embedding(
        self,
        query: str,
        embedding: List[float],
        crew_name: str,
        user_id: str,
        language: str,
        result_hash: str,
        ttl_seconds: int = 3600,
    ) -> bool:
        """Store query embedding with metadata for later similarity search.
        
        Args:
            query: Original query text
            embedding: Query embedding vector (384-dim)
            crew_name: Name of crew that executed query
            user_id: User ID
            language: Query language code
            result_hash: Hash of cached result (for reference)
            ttl_seconds: Time-to-live for cache entry
        
        Returns:
            True if successfully stored
        """
        if not embedding:
            return False
        
        try:
            # Ensure embedding is a proper list
            if isinstance(embedding, (int, float)):
                logger.warning(f"Received scalar value instead of embedding vector: {embedding}")
                return False
            
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                try:
                    embedding = list(embedding)
                except TypeError:
                    logger.warning(f"Cannot convert embedding to list: {type(embedding)}")
                    return False
            
            # Validate embedding is a list of floats with correct dimension
            if not isinstance(embedding, list) or len(embedding) != self.dimension:
                logger.warning(f"Invalid embedding: expected list of {self.dimension}, got {type(embedding)} with len {len(embedding) if isinstance(embedding, (list, tuple)) else 'N/A'}")
                return False
            
            # Create point with metadata
            point_id = int(time.time() * 1000000) % (2**31 - 1)  # Unique ID from timestamp
            
            payload = {
                "query": query,
                "crew_name": crew_name,
                "user_id": user_id,
                "language": language,
                "result_hash": result_hash,
                "created_at": time.time(),
                "ttl_seconds": ttl_seconds,
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            # Upsert point (insert or update)
            self._client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            # Update collection size metric
            try:
                collection_size = self.get_collection_size()
                fork_name = "default"  # Will be overridden by caller if available
                metrics.update_vector_db_collection_size(
                    fork_name=fork_name,
                    collection_name=self.collection_name,
                    size=collection_size
                )
            except Exception as metric_error:
                logger.debug(f"Failed to update collection size metric: {metric_error}")
            
            logger.debug(
                f"Stored embedding for query '{query[:50]}...' "
                f"(crew={crew_name}, user={user_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
    
    def search_similar(
        self,
        embedding: List[float],
        crew_name: str,
        user_id: str,
        language: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar query embeddings in Qdrant.
        
        Args:
            embedding: Query embedding to search for
            crew_name: Filter by crew name
            user_id: Filter by user ID
            language: Filter by language
            top_k: Number of results to return
            similarity_threshold: Override default threshold
        
        Returns:
            List of similar queries with scores and metadata
        """
        if not embedding:
            return []
        
        # Ensure embedding is a proper list
        if isinstance(embedding, (int, float)):
            logger.warning(f"Received scalar value instead of embedding vector: {embedding}")
            return []
        
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except TypeError:
                logger.warning(f"Cannot convert embedding to list: {type(embedding)}")
                return []
        
        # Validate embedding
        if not isinstance(embedding, list) or len(embedding) != self.dimension:
            logger.warning(f"Invalid embedding: expected list of {self.dimension}, got {type(embedding)} with len {len(embedding) if isinstance(embedding, (list, tuple)) else 'N/A'}")
            return []
        
        threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Build filter for crew/user/language
            filter_conditions = [
                FieldCondition(key="crew_name", match=MatchValue(value=crew_name)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="language", match=MatchValue(value=language)),
            ]
            query_filter = Filter(must=filter_conditions)
            
            # Search Qdrant with timing. Newer qdrant-client exposes `query_points`
            # while older versions used `search`.
            search_start = time.time()
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                results = getattr(response, "points", response)
            else:
                results = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=threshold,
                )
            search_duration = time.time() - search_start
            
            # Track search duration metric
            fork_name = "default"  # Will be overridden by caller if available
            metrics.track_vector_db_search(fork_name, search_duration)
            
            # Extract results
            matches = []
            for result in results:
                if result.score >= threshold:
                    payload = result.payload
                    matches.append({
                        "query": payload.get("query"),
                        "score": result.score,
                        "result_hash": payload.get("result_hash"),
                        "crew_name": payload.get("crew_name"),
                        "user_id": payload.get("user_id"),
                        "language": payload.get("language"),
                        "created_at": payload.get("created_at"),
                    })
            
            logger.debug(
                f"Found {len(matches)} similar queries for crew {crew_name} "
                f"(user {user_id}, threshold={threshold:.2f})"
            )
            return matches
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            return []
    
    def cleanup_expired(self, current_time: Optional[float] = None) -> int:
        """Clean up expired cache entries from vector store.
        
        Args:
            current_time: Current timestamp (defaults to time.time())
        
        Returns:
            Number of points deleted
        """
        if current_time is None:
            current_time = time.time()
        
        try:
            # Scroll through all points and delete expired ones
            deleted_count = 0
            scroll_result = self._client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            
            while scroll_result:
                points, next_offset = scroll_result
                
                # Find expired points
                expired_ids = []
                for point in points:
                    payload = point.payload
                    created_at = payload.get("created_at", 0)
                    ttl_seconds = payload.get("ttl_seconds", 3600)
                    expiry_time = created_at + ttl_seconds
                    
                    if current_time > expiry_time:
                        expired_ids.append(point.id)
                
                # Delete expired points
                if expired_ids:
                    self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=expired_ids,
                    )
                    deleted_count += len(expired_ids)
                    logger.debug(f"Deleted {len(expired_ids)} expired embeddings")
                
                # Continue scrolling if there are more points
                if next_offset is None:
                    break
                
                scroll_result = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired embeddings from {self.collection_name}")
                
                # Track cleanup metrics
                fork_name = "default"  # Will be overridden by caller if available
                metrics.track_vector_db_cleanup(fork_name, deleted_count)
            
            return deleted_count
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_collection_size(self) -> int:
        """Get total points in cache collection."""
        try:
            collection_info = self._client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Failed to get collection size: {e}")
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all entries from cache collection (use carefully)."""
        try:
            self._client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.warning(f"Cleared collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant backend is healthy and accessible."""
        try:
            # Try to get collection info
            self._client.get_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
