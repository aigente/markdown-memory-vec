"""
Hybrid search service combining semantic similarity, importance weighting, and temporal decay.

The hybrid retrieval formula:
    score = α × semantic_similarity(query, memory)    # sqlite-vec KNN
          + β × importance_weight(memory.importance)    # frontmatter
          + γ × temporal_decay(memory.last_accessed)    # frontmatter

    temporal_decay = exp(-λ × days_since_access)
    Default weights: α=0.6, β=0.2, γ=0.2, λ=0.05
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a hybrid search combining semantic, importance, and temporal signals."""

    file_path: str
    chunk_text: str
    chunk_index: int
    semantic_score: float  # 0.0-1.0, cosine similarity
    importance: float  # 0.0-1.0, from frontmatter
    temporal_decay: float  # 0.0-1.0, exp(-λ × days)
    hybrid_score: float  # Weighted combination
    memory_type: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    last_accessed: Optional[datetime] = None


class HybridSearchService:
    """
    Hybrid search service that combines:
    1. Semantic similarity (via sqlite-vec KNN search)
    2. Importance weighting (from memory frontmatter)
    3. Temporal decay (based on last access time)

    The combination formula is:
        score = α × semantic + β × importance + γ × temporal_decay

    All imports of vector infrastructure (ISqliteVecStore, IEmbedder) are optional
    to support graceful degradation when sqlite-vec is not installed.
    """

    def __init__(
        self,
        vec_store: Any,  # ISqliteVecStore — typed as Any for optional import safety
        embedder: Any,  # IEmbedder — typed as Any for optional import safety
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        decay_lambda: float = 0.05,
    ):
        """
        Initialize the hybrid search service.

        Args:
            vec_store: Vector store implementing ISqliteVecStore interface.
            embedder: Embedding model implementing IEmbedder interface.
            alpha: Weight for semantic similarity (default 0.6).
            beta: Weight for importance score (default 0.2).
            gamma: Weight for temporal decay (default 0.2).
            decay_lambda: Decay rate for temporal scoring (default 0.05).

        Raises:
            ValueError: If weights don't sum to approximately 1.0.
        """
        weight_sum = alpha + beta + gamma
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got α={alpha} + β={beta} + γ={gamma} = {weight_sum}")

        self.vec_store = vec_store
        self.embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_lambda = decay_lambda

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic, importance, and temporal signals.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            memory_type: Optional filter by memory type (e.g., "semantic", "episodic").
            min_score: Minimum hybrid score threshold (0.0-1.0).

        Returns:
            List of SearchResult sorted by hybrid_score descending.
        """
        if not query.strip():
            return []

        # Step 1: Embed the query
        try:
            query_embedding = self.embedder.embed(query)
        except Exception:
            logger.warning("Failed to embed query, returning empty results", exc_info=True)
            return []

        # Step 2: KNN search via vector store
        # Request more candidates than top_k to allow for filtering and re-ranking
        candidate_k = min(top_k * 3, 100)
        filter_metadata: Optional[Dict[str, Any]] = None
        if memory_type:
            filter_metadata = {"memory_type": memory_type}

        try:
            raw_results = self.vec_store.search(
                query_embedding=query_embedding,
                top_k=candidate_k,
                filter_metadata=filter_metadata,
            )
        except Exception:
            logger.warning("Vector search failed, returning empty results", exc_info=True)
            return []

        # Step 3: Compute hybrid scores
        results: list[SearchResult] = []
        for raw in raw_results:
            metadata = raw.metadata or {}

            # Extract fields from metadata
            file_path = metadata.get("file_path", "")
            chunk_text = metadata.get("chunk_text", "")
            chunk_index = metadata.get("chunk_index", 0)
            importance = float(metadata.get("importance", 0.5))
            tags = metadata.get("tags", [])
            mem_type = metadata.get("memory_type")
            last_accessed_str = metadata.get("last_accessed")

            # Parse last_accessed
            last_accessed: Optional[datetime] = None
            if last_accessed_str:
                try:
                    last_accessed = datetime.fromisoformat(str(last_accessed_str))
                except (ValueError, TypeError):
                    last_accessed = None

            # Normalize semantic score: convert cosine distance to similarity.
            # sqlite-vec with distance_metric=cosine returns distance in [0, 2]:
            #   0 = identical, 1 = orthogonal, 2 = opposite.
            # Similarity = 1 - distance maps to [-1, 1]; we clamp to [0, 1].
            semantic_score = max(0.0, min(1.0, 1.0 - raw.distance))

            # Compute temporal decay
            temporal_decay = self.compute_temporal_decay(last_accessed)

            # Clamp importance to [0, 1]
            importance = max(0.0, min(1.0, importance))

            # Compute hybrid score
            hybrid_score = self.compute_hybrid_score(semantic_score, importance, temporal_decay)

            if hybrid_score >= min_score:
                results.append(
                    SearchResult(
                        file_path=file_path,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        semantic_score=semantic_score,
                        importance=importance,
                        temporal_decay=temporal_decay,
                        hybrid_score=hybrid_score,
                        memory_type=mem_type,
                        tags=tags if isinstance(tags, list) else [],
                        last_accessed=last_accessed,
                    )
                )

        # Step 4: Sort by hybrid score and return top_k
        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    def compute_temporal_decay(self, last_accessed: Optional[datetime]) -> float:
        """
        Compute temporal decay factor: exp(-λ × days_since_access).

        Args:
            last_accessed: When the memory was last accessed. If None, returns 0.5
                          (neutral — neither penalized nor boosted).

        Returns:
            A float in [0, 1] where 1.0 means "just accessed" and approaches 0.0
            for very old memories.
        """
        if last_accessed is None:
            return 0.5  # Neutral default for memories without access time

        now = datetime.now(timezone.utc)

        # Ensure last_accessed is timezone-aware
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        delta = now - last_accessed
        days_since_access = max(0.0, delta.total_seconds() / 86400.0)

        return math.exp(-self.decay_lambda * days_since_access)

    def compute_hybrid_score(
        self,
        semantic_score: float,
        importance: float,
        temporal_decay: float,
    ) -> float:
        """
        Compute the weighted hybrid score.

        Args:
            semantic_score: Cosine similarity score [0, 1].
            importance: Importance weight from frontmatter [0, 1].
            temporal_decay: Temporal decay factor [0, 1].

        Returns:
            Weighted hybrid score = α × semantic + β × importance + γ × temporal_decay.
        """
        return self.alpha * semantic_score + self.beta * importance + self.gamma * temporal_decay
