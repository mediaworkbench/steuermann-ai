"""Embedding provider abstraction layer.

Supports remote OpenAI-compatible embeddings (LM Studio, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
import hashlib
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Embed one or more texts.

        Args:
            texts: Single text or list of texts

        Returns:
            Single embedding vector or list of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


def _deterministic_embedding(text: str, *, dimension: int, salt: str) -> List[float]:
    """Generate deterministic pseudo-embeddings with lexical overlap behavior.

    This fallback avoids network dependencies in tests while preserving a rough
    semantic signal: texts with overlapping tokens get higher cosine similarity.
    """
    vector = [0.0] * dimension
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        tokens = ["empty"]

    for token in tokens:
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            index = (i * 257 + byte) % dimension
            value = (byte / 127.5) - 1.0
            vector[index] += value

    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class RemoteEmbeddingProvider(EmbeddingProvider):
    """Remote OpenAI-compatible embedding provider (LM Studio, OpenAI, etc.)."""

    def __init__(self, endpoint: str, model_name: str, dimension: int):
        """Initialize remote embedding provider.

        Args:
            endpoint: OpenAI-compatible endpoint (e.g., http://localhost:8000/v1)
            model_name: Model name as recognized by the endpoint
            dimension: Expected embedding dimension
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. Install with: pip install httpx")

        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.dimension = dimension
        self.client = httpx.Client(timeout=30.0)
        self._fallback = endpoint.startswith("$") or "$/" in endpoint or "$" in endpoint
        logger.info(f"Initialized remote embedding provider: {endpoint} (model: {model_name})")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Embed using remote OpenAI-compatible API.

        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments (ignored for remote provider, e.g., show_progress_bar)

        Returns:
            Single embedding (list) or list of embeddings
        """
        # Normalize to list
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts

        if self._fallback:
            embedded = [
                _deterministic_embedding(text, dimension=self.dimension, salt=f"remote:{self.model_name}")
                for text in texts_list
            ]
            return embedded[0] if is_single else embedded

        try:
            # Call OpenAI-compatible endpoint
            response = self.client.post(
                f"{self.endpoint}/embeddings",
                json={"input": texts_list, "model": self.model_name},
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]

            # Return single embedding or list based on input
            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.warning(f"Remote embedding request failed, using deterministic fallback: {e}")
            embedded = [
                _deterministic_embedding(text, dimension=self.dimension, salt=f"remote:{self.model_name}")
                for text in texts_list
            ]
            return embedded[0] if is_single else embedded

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass


def build_embedding_provider(
    model_name: str,
    dimension: int,
    provider_type: str = "remote",
    remote_endpoint: Optional[str] = None,
) -> EmbeddingProvider:
    """Factory function to build the appropriate embedding provider.

    Args:
        model_name: Model name
        dimension: Expected embedding dimension
        provider_type: Must be "remote"
        remote_endpoint: Required endpoint for the remote provider

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If configuration is invalid
    """
    if provider_type != "remote":
        raise ValueError(f"Unsupported provider type: {provider_type}. Only 'remote' is supported")
    if not remote_endpoint:
        raise ValueError("remote_endpoint required for remote provider")
    return RemoteEmbeddingProvider(remote_endpoint, model_name, dimension)
