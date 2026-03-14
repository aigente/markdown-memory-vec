"""
Interfaces for vector infrastructure components.

These are the contracts that concrete implementations (store, embedder) must fulfill.
This file serves as the integration point between components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class VectorSearchResult:
    """Raw result from a vector KNN search."""

    id: str
    distance: float  # Lower = more similar (L2) or higher = more similar (cosine)
    metadata: Dict[str, Any]


@dataclass
class VectorRecord:
    """A record stored in the vector store."""

    id: str
    embedding: List[float]
    metadata: Dict[str, Any]


class IEmbedder(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into vectors.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        ...


class ISqliteVecStore(ABC):
    """Interface for sqlite-vec based vector storage.

    Concrete implementations use sqlite-vec for KNN search over embedding vectors.
    """

    @abstractmethod
    def add(self, records: Sequence[VectorRecord]) -> None:
        """Add records to the vector store.

        Args:
            records: Sequence of VectorRecord to add.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform KNN search.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            List of VectorSearchResult sorted by relevance.
        """
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete records by IDs.

        Args:
            ids: IDs of records to delete.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Delete all records from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of records in the store."""
        ...
