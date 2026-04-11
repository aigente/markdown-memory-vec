"""
Hybrid search service combining semantic similarity, FTS5 keyword search,
importance weighting, and temporal decay.

Supports three search modes:
- ``vector_only``: semantic similarity via sqlite-vec KNN (original behaviour)
- ``fts_only``: keyword matching via SQLite FTS5 (BM25)
- ``hybrid`` (default): both vector and FTS5, fused via Reciprocal Rank Fusion (RRF)

The final scoring formula (all modes):
    score = α × relevance_score + β × importance + γ × temporal_decay

Where *relevance_score* is:
- ``vector_only``: cosine similarity
- ``fts_only``: normalised BM25
- ``hybrid``: normalised RRF score

RRF formula: ``RRF(d) = Σ 1 / (k + rank_i(d))`` for each ranking *i* containing *d*.
Default RRF constant ``k = 60``.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SearchMode(enum.Enum):
    """Search mode for :class:`HybridSearchService`."""

    VECTOR_ONLY = "vector_only"
    FTS_ONLY = "fts_only"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Result of a hybrid search combining semantic, FTS5, importance, and temporal signals."""

    file_path: str
    chunk_text: str
    chunk_index: int
    semantic_score: float  # 0.0-1.0, cosine similarity (0 if fts_only)
    importance: float  # 0.0-1.0, from frontmatter
    temporal_decay: float  # 0.0-1.0, exp(-λ × days)
    hybrid_score: float  # Weighted combination
    memory_type: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    last_accessed: Optional[datetime] = None
    fts_score: float = 0.0  # Normalised BM25 score (0 if vector_only)
    rrf_score: float = 0.0  # Normalised RRF score (only in hybrid mode)


# ---------------------------------------------------------------------------
# RRF utility
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion scores from multiple ranked lists.

    Parameters
    ----------
    ranked_lists:
        Each element is an ordered list of document IDs (best first).
    k:
        RRF constant (default 60).  Higher *k* reduces the influence of
        high-ranking outliers.

    Returns
    -------
    A dict mapping document ID → RRF score (higher = more relevant).
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridSearchService:
    """
    Hybrid search service that combines:
    1. Semantic similarity (via sqlite-vec KNN search)
    2. FTS5 keyword matching (via SQLite FTS5 / BM25)
    3. Importance weighting (from memory frontmatter)
    4. Temporal decay (based on last access time)

    Supports three modes:
    - ``vector_only``: original behaviour (semantic KNN only)
    - ``fts_only``: FTS5 keyword search only
    - ``hybrid`` (default): RRF fusion of vector + FTS5

    The combination formula is:
        score = α × relevance + β × importance + γ × temporal_decay

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
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid search service.

        Args:
            vec_store: Vector store implementing ISqliteVecStore interface.
                       Must also expose ``search_fts(query, top_k)`` and
                       ``get_meta(rowid)`` for hybrid / FTS-only modes.
            embedder: Embedding model implementing IEmbedder interface.
            alpha: Weight for relevance score (default 0.6).
            beta: Weight for importance score (default 0.2).
            gamma: Weight for temporal decay (default 0.2).
            decay_lambda: Decay rate for temporal scoring (default 0.05).
            rrf_k: RRF constant (default 60).

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
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_score: float = 0.0,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[SearchResult]:
        """
        Perform search combining semantic, FTS5, importance, and temporal signals.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            memory_type: Optional filter by memory type (e.g., "semantic", "episodic").
            min_score: Minimum hybrid score threshold (0.0-1.0).
            mode: Search mode — ``hybrid`` (default), ``vector_only``, or ``fts_only``.

        Returns:
            List of SearchResult sorted by hybrid_score descending.
        """
        if not query.strip():
            return []

        if mode == SearchMode.VECTOR_ONLY:
            return self._search_vector(query, top_k, memory_type, min_score)
        elif mode == SearchMode.FTS_ONLY:
            return self._search_fts(query, top_k, memory_type, min_score)
        else:
            return self._search_hybrid(query, top_k, memory_type, min_score)

    # ------------------------------------------------------------------
    # Vector-only search (original behaviour)
    # ------------------------------------------------------------------

    def _search_vector(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[str],
        min_score: float,
    ) -> list[SearchResult]:
        """Semantic-only search via vector KNN."""
        # Step 1: Embed the query
        try:
            query_embedding = self.embedder.embed(query)
        except Exception:
            logger.warning("Failed to embed query, returning empty results", exc_info=True)
            return []

        # Step 2: KNN search via vector store
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
            r = self._build_result_from_metadata(metadata, raw.distance, 0.0, 0.0)
            # Relevance = semantic_score
            r.hybrid_score = self.compute_hybrid_score(r.semantic_score, r.importance, r.temporal_decay)
            if r.hybrid_score >= min_score:
                results.append(r)

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # FTS-only search
    # ------------------------------------------------------------------

    def _search_fts(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[str],
        min_score: float,
    ) -> list[SearchResult]:
        """Keyword-only search via FTS5."""
        if not hasattr(self.vec_store, "search_fts"):
            logger.warning("Vec store does not support FTS5; falling back to vector search")
            return self._search_vector(query, top_k, memory_type, min_score)

        candidate_k = min(top_k * 3, 100)
        fts_raw = self.vec_store.search_fts(query, top_k=candidate_k)
        if not fts_raw:
            return []

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(r.bm25_score for r in fts_raw) if fts_raw else 1.0
        max_bm25 = max(max_bm25, 1e-9)

        results: list[SearchResult] = []
        for raw in fts_raw:
            meta = self.vec_store.get_meta(raw.rowid)
            if meta is None:
                continue
            if memory_type and meta.get("memory_type") != memory_type:
                continue

            fts_norm = raw.bm25_score / max_bm25
            r = self._build_result_from_metadata(meta, distance=None, fts_score=fts_norm, rrf_score=0.0)
            # Relevance = fts_score
            r.hybrid_score = self.compute_hybrid_score(fts_norm, r.importance, r.temporal_decay)
            if r.hybrid_score >= min_score:
                results.append(r)

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Hybrid search (vector + FTS5 with RRF fusion)
    # ------------------------------------------------------------------

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[str],
        min_score: float,
    ) -> list[SearchResult]:
        """Hybrid search: vector KNN + FTS5 fused via RRF."""
        candidate_k = min(top_k * 3, 100)

        # --- Vector candidates ---
        vec_ids: list[str] = []
        vec_distances: dict[str, float] = {}
        try:
            query_embedding = self.embedder.embed(query)
            filter_metadata: Optional[Dict[str, Any]] = None
            if memory_type:
                filter_metadata = {"memory_type": memory_type}
            vec_raw = self.vec_store.search(
                query_embedding=query_embedding,
                top_k=candidate_k,
                filter_metadata=filter_metadata,
            )
            for raw in vec_raw:
                vec_ids.append(raw.id)
                vec_distances[raw.id] = raw.distance
        except Exception:
            logger.warning("Vector search failed in hybrid mode, using FTS only", exc_info=True)

        # --- FTS5 candidates ---
        fts_ids: list[str] = []
        fts_scores: dict[str, float] = {}
        if hasattr(self.vec_store, "search_fts"):
            try:
                fts_raw = self.vec_store.search_fts(query, top_k=candidate_k)
                max_bm25 = max((r.bm25_score for r in fts_raw), default=1.0)
                max_bm25 = max(max_bm25, 1e-9)
                for raw in fts_raw:
                    doc_id = str(raw.rowid)
                    fts_ids.append(doc_id)
                    fts_scores[doc_id] = raw.bm25_score / max_bm25
            except Exception:
                logger.warning("FTS5 search failed in hybrid mode, using vector only", exc_info=True)

        # If both failed, give up
        if not vec_ids and not fts_ids:
            return []

        # If only one source succeeded, fall through to RRF with a single list
        # (RRF with one list == just that list's ranking, which is correct)

        # --- RRF fusion ---
        rrf_raw = reciprocal_rank_fusion([vec_ids, fts_ids], k=self.rrf_k)
        if not rrf_raw:
            return []

        # Normalise RRF scores to [0, 1]
        max_rrf = max(rrf_raw.values()) if rrf_raw else 1.0
        max_rrf = max(max_rrf, 1e-9)

        # Collect all candidate doc IDs, ranked by RRF
        all_ids = sorted(rrf_raw.keys(), key=lambda d: rrf_raw[d], reverse=True)

        results: list[SearchResult] = []
        for doc_id in all_ids:
            meta = self.vec_store.get_meta(int(doc_id))
            if meta is None:
                continue
            if memory_type and meta.get("memory_type") != memory_type:
                continue

            distance = vec_distances.get(doc_id)
            fts_score = fts_scores.get(doc_id, 0.0)
            rrf_norm = rrf_raw[doc_id] / max_rrf

            r = self._build_result_from_metadata(
                meta,
                distance=distance,
                fts_score=fts_score,
                rrf_score=rrf_norm,
            )
            # Relevance = normalised RRF score
            r.hybrid_score = self.compute_hybrid_score(rrf_norm, r.importance, r.temporal_decay)
            if r.hybrid_score >= min_score:
                results.append(r)
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result_from_metadata(
        self,
        metadata: Dict[str, Any],
        distance: Optional[float],
        fts_score: float,
        rrf_score: float,
    ) -> SearchResult:
        """Build a :class:`SearchResult` from a metadata dict.

        The ``hybrid_score`` field is set to 0.0 — the caller must fill it in.
        """
        file_path = metadata.get("file_path", "")
        chunk_text = metadata.get("chunk_text", "")
        chunk_index = metadata.get("chunk_index", 0)
        importance = max(0.0, min(1.0, float(metadata.get("importance", 0.5))))
        tags = metadata.get("tags", [])
        mem_type = metadata.get("memory_type")
        last_accessed_str = metadata.get("last_accessed")

        last_accessed: Optional[datetime] = None
        if last_accessed_str:
            try:
                last_accessed = datetime.fromisoformat(str(last_accessed_str))
            except (ValueError, TypeError):
                last_accessed = None

        semantic_score = 0.0
        if distance is not None:
            semantic_score = max(0.0, min(1.0, 1.0 - distance))

        temporal_decay = self.compute_temporal_decay(last_accessed)

        return SearchResult(
            file_path=file_path,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            semantic_score=semantic_score,
            importance=importance,
            temporal_decay=temporal_decay,
            hybrid_score=0.0,  # Caller fills this in
            memory_type=mem_type,
            tags=tags if isinstance(tags, list) else [],
            last_accessed=last_accessed,
            fts_score=fts_score,
            rrf_score=rrf_score,
        )

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
        relevance_score: float,
        importance: float,
        temporal_decay: float,
    ) -> float:
        """
        Compute the weighted hybrid score.

        Args:
            relevance_score: Relevance score [0, 1] — cosine similarity in
                vector_only mode, normalised BM25 in fts_only, or normalised
                RRF in hybrid mode.
            importance: Importance weight from frontmatter [0, 1].
            temporal_decay: Temporal decay factor [0, 1].

        Returns:
            Weighted hybrid score = α × relevance + β × importance + γ × temporal_decay.
        """
        return self.alpha * relevance_score + self.beta * importance + self.gamma * temporal_decay
