from typing import Any, List, Tuple

from universal_agentic_framework.memory.qdrant_backend import QdrantMemoryBackend


class _FakePoint:
    def __init__(self, pid: str, vector: List[float], payload: dict):
        self.id = pid
        self.vector = vector
        self.payload = payload


class _FakeClient:
    """Minimal in-memory stand-in for QdrantClient."""

    def __init__(self) -> None:
        self._collections = {}

    def get_collection(self, name: str):
        return self._collections.get(name)

    def create_collection(self, collection_name: str, vectors_config: Any):
        self._collections.setdefault(collection_name, {"points": []})

    def upsert(self, collection_name: str, points: List[Any]):
        col = self._collections.setdefault(collection_name, {"points": []})
        for p in points:
            if isinstance(p, dict):
                col["points"].append(_FakePoint(p["id"], p.get("vector", []), p.get("payload", {})))
            else:
                col["points"].append(_FakePoint(p.id, getattr(p, "vector", []), getattr(p, "payload", {})))

    def scroll(self, collection_name: str, limit: int = 10) -> Tuple[List[_FakePoint], None]:
        col = self._collections.get(collection_name, {"points": []})
        pts = col["points"][:limit]
        return pts, None


class _FakeEmbedder:
    def encode(self, texts):
        # Return 384-dim zero vectors as numpy arrays to match real embedder behavior
        import numpy as np
        return [np.array([0.0] * 384) for _ in texts]


def test_qdrant_backend_upsert_and_load_without_query():
    client = _FakeClient()
    embedder = _FakeEmbedder()
    backend = QdrantMemoryBackend(
        collection_prefix="smoketest",
        client=client,
        embedder=embedder,
        dimension=384,
    )

    backend.upsert("u1", "alpha", {"k": 1})
    backend.upsert("u1", "beta", {"k": 2})
    backend.upsert("u2", "gamma", {"k": 3})

    loaded = backend.load("u1", query=None, top_k=10)
    texts = [r.text for r in loaded]

    assert "alpha" in texts and "beta" in texts
    assert "gamma" not in texts
