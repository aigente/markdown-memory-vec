"""
Tests for HybridSearchService: scoring formula, temporal decay, and full search pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from memory_vec.search import HybridSearchService

from .conftest import FakeEmbedder, FakeVecStore, make_search_result

# ============================================================================
# HybridSearchService Scoring Tests
# ============================================================================


class TestHybridSearchServiceScoring:
    """Tests for the hybrid scoring formula."""

    def test_default_weights(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Default weights should be α=0.6, β=0.2, γ=0.2."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        assert svc.alpha == 0.6
        assert svc.beta == 0.2
        assert svc.gamma == 0.2

    def test_weight_validation_rejects_bad_sum(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridSearchService(fake_vec_store, fake_embedder, alpha=0.5, beta=0.5, gamma=0.5)

    def test_custom_weights(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Custom weights should be accepted if they sum to 1.0."""
        svc = HybridSearchService(fake_vec_store, fake_embedder, alpha=0.8, beta=0.1, gamma=0.1)
        assert svc.alpha == 0.8
        assert svc.beta == 0.1
        assert svc.gamma == 0.1

    def test_compute_hybrid_score_default_weights(
        self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore
    ) -> None:
        """Test the hybrid score formula: score = α×sem + β×imp + γ×decay."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        score = svc.compute_hybrid_score(
            relevance_score=0.9,
            importance=0.8,
            temporal_decay=0.7,
        )
        expected = 0.6 * 0.9 + 0.2 * 0.8 + 0.2 * 0.7
        assert abs(score - expected) < 1e-10

    def test_compute_hybrid_score_all_zero(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """All zeros should yield 0."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        score = svc.compute_hybrid_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_compute_hybrid_score_all_one(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """All ones should yield 1.0."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        score = svc.compute_hybrid_score(1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-10


# ============================================================================
# Temporal Decay Tests
# ============================================================================


class TestTemporalDecay:
    """Tests for temporal decay: exp(-λ × days_since_access)."""

    def test_just_accessed(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """A memory just accessed should have decay ≈ 1.0."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        now = datetime.now(timezone.utc)
        decay = svc.compute_temporal_decay(now)
        assert decay > 0.99

    def test_one_day_old(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """A memory 1 day old with λ=0.05 should have decay ≈ exp(-0.05)."""
        svc = HybridSearchService(fake_vec_store, fake_embedder, decay_lambda=0.05)
        one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        decay = svc.compute_temporal_decay(one_day_ago)
        expected = math.exp(-0.05 * 1)
        assert abs(decay - expected) < 0.01

    def test_thirty_days_old(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """A memory 30 days old with λ=0.05 should have decay ≈ exp(-1.5) ≈ 0.22."""
        svc = HybridSearchService(fake_vec_store, fake_embedder, decay_lambda=0.05)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        decay = svc.compute_temporal_decay(thirty_days_ago)
        expected = math.exp(-0.05 * 30)
        assert abs(decay - expected) < 0.01

    def test_none_last_accessed_returns_neutral(
        self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore
    ) -> None:
        """None last_accessed should return neutral 0.5."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        decay = svc.compute_temporal_decay(None)
        assert decay == 0.5

    def test_naive_datetime_treated_as_utc(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Timezone-naive datetime should be treated as UTC."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        naive_recent = datetime.now() - timedelta(seconds=10)
        decay = svc.compute_temporal_decay(naive_recent)
        assert decay > 0.99

    def test_high_lambda_decays_faster(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Higher λ should produce faster decay."""
        svc_slow = HybridSearchService(fake_vec_store, fake_embedder, decay_lambda=0.01)
        svc_fast = HybridSearchService(fake_vec_store, fake_embedder, decay_lambda=0.1)
        ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
        assert svc_slow.compute_temporal_decay(ten_days_ago) > svc_fast.compute_temporal_decay(ten_days_ago)


# ============================================================================
# Full Search Pipeline Tests
# ============================================================================


class TestHybridSearch:
    """Tests for the full search pipeline."""

    def test_empty_query_returns_empty(self, fake_embedder: FakeEmbedder, fake_vec_store: FakeVecStore) -> None:
        """Empty query should return empty results."""
        svc = HybridSearchService(fake_vec_store, fake_embedder)
        results = svc.search("")
        assert results == []

    def test_search_returns_ranked_results(self, fake_embedder: FakeEmbedder) -> None:
        """Results should be sorted by hybrid_score descending."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.3,
                importance=0.5,
                last_accessed=(now - timedelta(days=10)).isoformat(),
                chunk_text="low semantic",
            ),
            make_search_result(
                distance=0.1,
                importance=0.9,
                last_accessed=now.isoformat(),
                chunk_text="high semantic + important + recent",
            ),
            make_search_result(
                distance=0.2,
                importance=0.7,
                last_accessed=(now - timedelta(days=2)).isoformat(),
                chunk_text="medium",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("test query")
        assert len(results) == 3
        # Results should be sorted by hybrid_score descending
        for i in range(len(results) - 1):
            assert results[i].hybrid_score >= results[i + 1].hybrid_score

    def test_search_respects_top_k(self, fake_embedder: FakeEmbedder) -> None:
        """top_k should limit the number of results."""
        results_data = [make_search_result(distance=0.1 * i, chunk_text=f"chunk {i}") for i in range(1, 6)]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("query", top_k=2)
        assert len(results) == 2

    def test_search_respects_min_score(self, fake_embedder: FakeEmbedder) -> None:
        """min_score should filter out low-scoring results."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.9,
                importance=0.1,
                last_accessed=(now - timedelta(days=100)).isoformat(),
                chunk_text="very low score",
            ),
            make_search_result(
                distance=0.1,
                importance=0.9,
                last_accessed=now.isoformat(),
                chunk_text="high score",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("query", min_score=0.5)
        # Only the high-score result should pass
        assert len(results) >= 1
        for r in results:
            assert r.hybrid_score >= 0.5

    def test_search_with_memory_type_filter(self, fake_embedder: FakeEmbedder) -> None:
        """memory_type filter should be passed to vec store."""
        store = FakeVecStore()
        store.search = MagicMock(return_value=[])  # type: ignore[assignment]
        svc = HybridSearchService(store, fake_embedder)

        svc.search("query", memory_type="episodic")
        store.search.assert_called_once()
        call_kwargs = store.search.call_args
        assert call_kwargs[1]["filter_metadata"] == {"memory_type": "episodic"}

    def test_search_result_fields_populated(self, fake_embedder: FakeEmbedder) -> None:
        """SearchResult fields should be properly populated from metadata."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.15,
                importance=0.8,
                last_accessed=now.isoformat(),
                file_path="episodic/diary.md",
                chunk_text="Today's entry",
                chunk_index=2,
                memory_type="episodic",
                tags=["daily", "work"],
            )
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("diary")
        assert len(results) == 1
        r = results[0]
        assert r.file_path == "episodic/diary.md"
        assert r.chunk_text == "Today's entry"
        assert r.chunk_index == 2
        assert r.memory_type == "episodic"
        assert r.tags == ["daily", "work"]
        assert r.importance == 0.8
        assert 0 <= r.semantic_score <= 1
        assert 0 <= r.temporal_decay <= 1
        assert 0 <= r.hybrid_score <= 1
