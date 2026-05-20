"""Embedding provider abstraction layer.

Supports remote OpenAI-compatible embeddings (LM Studio, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
import time
from typing import List, Optional, Union
import logging

import httpx

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = {503}


class EmbeddingProviderUnavailableError(RuntimeError):
    """Raised when the remote embedding provider is unreachable after all retries."""
    pass


def normalize_embedding_model_name(model_name: str) -> str:
    """Normalize embedding model IDs for OpenAI-compatible /embeddings endpoints.

    The chat stack may use LiteLLM-style IDs such as ``openai/<model>``.
    Embedding endpoints (LM Studio/OpenAI-compatible) typically expect raw model IDs.
    """
    return (model_name or "").removeprefix("openai/").strip()


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


class RemoteEmbeddingProvider(EmbeddingProvider):
    """Remote OpenAI-compatible embedding provider (LM Studio, OpenAI, etc.)."""

    def __init__(self, endpoint: str, model_name: str, dimension: int):
        """Initialize remote embedding provider.

        Args:
            endpoint: OpenAI-compatible endpoint (e.g., http://localhost:8000/v1)
            model_name: Model name as recognized by the endpoint
            dimension: Expected embedding dimension

        Raises:
            ValueError: If endpoint contains an unresolved environment variable placeholder.
        """
        if endpoint.startswith("$"):
            raise ValueError(
                f"Embedding endpoint '{endpoint}' looks like an unresolved environment variable. "
                "Check that the relevant env var is set and loaded before initializing the provider."
            )

        self.endpoint = endpoint.rstrip("/")
        requested_model_name = model_name
        self.model_name = normalize_embedding_model_name(model_name)
        self.dimension = dimension
        self.client = httpx.Client(timeout=30.0)
        if requested_model_name != self.model_name:
            logger.info(
                "Normalized embedding model ID for remote provider: %s -> %s",
                requested_model_name,
                self.model_name,
            )
        logger.info(f"Initialized remote embedding provider: {endpoint} (model: {self.model_name})")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Embed using remote OpenAI-compatible API.

        Retries up to 3 times with exponential backoff on transient errors
        (connection refused, timeout, HTTP 503). Non-retryable HTTP errors propagate
        immediately. Raises EmbeddingProviderUnavailableError after all retries are
        exhausted.

        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments (ignored for remote provider)

        Returns:
            Single embedding (list) or list of embeddings

        Raises:
            EmbeddingProviderUnavailableError: Provider unreachable after retries.
            httpx.HTTPStatusError: Non-retryable HTTP error from the provider.
        """
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts

        retry_delays = [1.0, 2.0, 4.0]
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate([0.0] + retry_delays):
            if delay:
                logger.warning(
                    "Embedding request retry %d/4 in %.0f s (endpoint: %s)",
                    attempt, delay, self.endpoint,
                )
                time.sleep(delay)
            try:
                response = self.client.post(
                    f"{self.endpoint}/embeddings",
                    json={"input": texts_list, "model": self.model_name},
                )
                if response.status_code in TRANSIENT_STATUS_CODES:
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                    logger.warning(
                        "Embedding request returned %d (attempt %d/4, endpoint: %s)",
                        response.status_code, attempt + 1, self.endpoint,
                    )
                    continue
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings[0] if is_single else embeddings

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "Embedding request failed (attempt %d/4, endpoint: %s): %s",
                    attempt + 1, self.endpoint, e,
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in TRANSIENT_STATUS_CODES:
                    raise
                last_exc = e
                logger.warning(
                    "Embedding request returned %d (attempt %d/4, endpoint: %s)",
                    e.response.status_code, attempt + 1, self.endpoint,
                )

        raise EmbeddingProviderUnavailableError(
            f"Embedding provider unreachable after 4 attempts (endpoint: {self.endpoint}, model: {self.model_name})"
        ) from last_exc

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
        ValueError: If configuration is invalid or endpoint is unresolved.
    """
    if provider_type != "remote":
        raise ValueError(f"Unsupported provider type: {provider_type}. Only 'remote' is supported")
    if not remote_endpoint:
        raise ValueError("remote_endpoint required for remote provider")
    return RemoteEmbeddingProvider(remote_endpoint, model_name, dimension)
