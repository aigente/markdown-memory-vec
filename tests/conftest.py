"""
Shared test fixtures for markdown-memory-vec tests.

All tests use Fake/Mock implementations — no real models are downloaded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import pytest
from memory_vec.interfaces import VectorRecord, VectorSearchResult


class FakeEmbedder:
    """Fake embedder for testing — returns a fixed-dimension vector."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        # Simple deterministic embedding based on text length
        return [float(len(text) % (i + 1)) / max(1, i + 1) for i in range(self._dim)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


class FakeVecStore:
    """Fake vector store for testing — stores records in memory and returns preset results."""

    def __init__(self, search_results: Optional[List[VectorSearchResult]] = None):
        self._records: List[VectorRecord] = []
        self._search_results = search_results or []

    def add(self, records: Sequence[VectorRecord]) -> None:
        self._records.extend(records)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        return self._search_results[:top_k]

    def delete(self, ids: Sequence[str]) -> None:
        self._records = [r for r in self._records if r.id not in ids]

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=4)


@pytest.fixture
def fake_vec_store() -> FakeVecStore:
    return FakeVecStore()


def make_search_result(
    distance: float = 0.2,
    importance: float = 0.7,
    last_accessed: Optional[str] = None,
    file_path: str = "semantic/test.md",
    chunk_text: str = "Test chunk",
    chunk_index: int = 0,
    memory_type: str = "semantic",
    tags: Optional[list[str]] = None,
) -> VectorSearchResult:
    """Helper to create a VectorSearchResult with metadata."""
    return VectorSearchResult(
        id=str(uuid4()),
        distance=distance,
        metadata={
            "file_path": file_path,
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "importance": importance,
            "tags": tags or [],
            "memory_type": memory_type,
            "last_accessed": last_accessed,
        },
    )
