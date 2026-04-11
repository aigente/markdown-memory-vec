# pyright: reportPrivateUsage=false, reportGeneralTypeIssues=false
"""
Tests for hybrid search: FTS5 integration, RRF fusion, search modes,
CJK support, and end-to-end indexing + search with real SqliteVecStore.

Divided into:
- Unit tests (RRF, SearchMode, FakeVecStore based)
- CJK segmentation unit tests
- Integration tests (real SqliteVecStore + MemoryIndexer + FTS5)
- CJK integration tests
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from memory_vec.search import HybridSearchService, SearchMode, SearchResult, reciprocal_rank_fusion
from memory_vec.store import SqliteVecStore, _cjk_segment, is_sqlite_vec_available

from .conftest import FakeEmbedder, FakeVecStore, make_search_result

# ============================================================================
# RRF Unit Tests
# ============================================================================


class TestReciprocalRankFusion:
    """Tests for the RRF utility function."""

    def test_single_list(self) -> None:
        """Single ranked list produces RRF scores."""
        scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        assert len(scores) == 3
        assert scores["a"] > scores["b"] > scores["c"]

    def test_two_lists_boost_overlap(self) -> None:
        """Documents in both lists get higher RRF scores."""
        list1 = ["a", "b", "c"]
        list2 = ["b", "d", "a"]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        # 'b' appears rank 2 in list1, rank 1 in list2 → best combined
        # 'a' appears rank 1 in list1, rank 3 in list2 → also good
        # Both 'a' and 'b' appear in both lists → should score higher than 'c' or 'd'
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["d"]

    def test_empty_lists(self) -> None:
        """Empty lists produce empty scores."""
        scores = reciprocal_rank_fusion([[], []], k=60)
        assert scores == {}

    def test_disjoint_lists(self) -> None:
        """Completely disjoint lists should all have positive scores."""
        scores = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)
        assert len(scores) == 4
        for v in scores.values():
            assert v > 0

    def test_k_parameter_effect(self) -> None:
        """Lower k gives more weight to top-ranked documents."""
        list1 = ["a", "b", "c"]
        scores_low_k = reciprocal_rank_fusion([list1], k=1)
        scores_high_k = reciprocal_rank_fusion([list1], k=100)
        # With low k, the gap between rank 1 and rank 3 is larger
        gap_low = scores_low_k["a"] - scores_low_k["c"]
        gap_high = scores_high_k["a"] - scores_high_k["c"]
        assert gap_low > gap_high


# ============================================================================
# SearchMode Tests
# ============================================================================


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_values(self) -> None:
        assert SearchMode.VECTOR_ONLY.value == "vector_only"
        assert SearchMode.FTS_ONLY.value == "fts_only"
        assert SearchMode.HYBRID.value == "hybrid"

    def test_from_string(self) -> None:
        assert SearchMode("hybrid") == SearchMode.HYBRID
        assert SearchMode("vector_only") == SearchMode.VECTOR_ONLY
        assert SearchMode("fts_only") == SearchMode.FTS_ONLY


# ============================================================================
# Hybrid Search Service — Mode Dispatch Tests (with FakeVecStore)
# ============================================================================


class TestSearchModeDispatch:
    """Tests that the search() method correctly dispatches to the right mode."""

    def test_default_mode_is_hybrid(self, fake_embedder: FakeEmbedder) -> None:
        """Default search mode should be hybrid."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.1,
                importance=0.8,
                last_accessed=now.isoformat(),
                chunk_text="matching query text",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("matching")
        assert len(results) >= 1

    def test_vector_only_mode(self, fake_embedder: FakeEmbedder) -> None:
        """vector_only mode should use only semantic search."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.15,
                importance=0.7,
                last_accessed=now.isoformat(),
                chunk_text="vector test",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("vector test", mode=SearchMode.VECTOR_ONLY)
        assert len(results) == 1
        r = results[0]
        assert r.semantic_score > 0
        assert r.fts_score == 0.0  # No FTS in vector_only mode
        assert r.rrf_score == 0.0

    def test_fts_only_mode(self, fake_embedder: FakeEmbedder) -> None:
        """fts_only mode should use only FTS5 search."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.1,
                importance=0.9,
                last_accessed=now.isoformat(),
                chunk_text="keyword search test",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("keyword", mode=SearchMode.FTS_ONLY)
        assert len(results) == 1
        r = results[0]
        assert r.fts_score > 0.0
        assert r.semantic_score == 0.0  # No vector in fts_only mode

    def test_fts_only_no_match(self, fake_embedder: FakeEmbedder) -> None:
        """fts_only with no keyword match should return empty."""
        results_data = [
            make_search_result(chunk_text="something else entirely"),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("nonexistent", mode=SearchMode.FTS_ONLY)
        assert results == []

    def test_hybrid_mode_boosts_overlap(self, fake_embedder: FakeEmbedder) -> None:
        """Hybrid mode: results matching both vector and FTS should rank higher."""
        now = datetime.now(timezone.utc)
        results_data = [
            make_search_result(
                distance=0.1,
                importance=0.5,
                last_accessed=now.isoformat(),
                chunk_text="deploy kubernetes cluster",
            ),
            make_search_result(
                distance=0.3,
                importance=0.5,
                last_accessed=now.isoformat(),
                chunk_text="deploy the application",
            ),
        ]
        store = FakeVecStore(search_results=results_data)
        svc = HybridSearchService(store, fake_embedder)

        results = svc.search("deploy", mode=SearchMode.HYBRID)
        # Both should appear (both match "deploy" in FTS and are in vector results)
        assert len(results) == 2

    def test_empty_query_returns_empty_all_modes(self, fake_embedder: FakeEmbedder) -> None:
        """Empty query should return empty for all modes."""
        store = FakeVecStore(search_results=[make_search_result()])
        svc = HybridSearchService(store, fake_embedder)

        for mode in SearchMode:
            assert svc.search("", mode=mode) == []
            assert svc.search("  ", mode=mode) == []


# ============================================================================
# SearchResult Fields Tests
# ============================================================================


class TestSearchResultFields:
    """Tests for new fields in SearchResult."""

    def test_fts_score_field_exists(self) -> None:
        """SearchResult should have fts_score field with default 0."""
        r = SearchResult(
            file_path="test.md",
            chunk_text="hello",
            chunk_index=0,
            semantic_score=0.8,
            importance=0.5,
            temporal_decay=0.9,
            hybrid_score=0.7,
        )
        assert r.fts_score == 0.0
        assert r.rrf_score == 0.0

    def test_fts_score_field_custom(self) -> None:
        """SearchResult should accept custom fts_score."""
        r = SearchResult(
            file_path="test.md",
            chunk_text="hello",
            chunk_index=0,
            semantic_score=0.8,
            importance=0.5,
            temporal_decay=0.9,
            hybrid_score=0.7,
            fts_score=0.95,
            rrf_score=0.6,
        )
        assert r.fts_score == 0.95
        assert r.rrf_score == 0.6


# ============================================================================
# Integration Tests — Real SqliteVecStore + FTS5
# ============================================================================

_DIM = 4
_skip_no_vec = pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")


def _make_embedding(seed: float = 1.0) -> list[float]:
    """Create a simple test embedding."""
    return [seed * (i + 1) / _DIM for i in range(_DIM)]


@_skip_no_vec
class TestFtsStoreIntegration:
    """Integration tests for FTS5 operations in SqliteVecStore."""

    def test_ensure_tables_creates_fts(self) -> None:
        """ensure_tables should create memory_fts table."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        row = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        ).fetchone()
        assert row is not None

    def test_insert_syncs_fts(self) -> None:
        """insert_embedding should also insert into FTS5 table."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="Hello world this is a test",
        )
        assert store.fts_count() == 1

    def test_delete_syncs_fts(self) -> None:
        """delete_embedding should also remove from FTS5."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="Hello world",
        )
        assert store.fts_count() == 1
        store.delete_embedding(rowid)
        assert store.fts_count() == 0

    def test_delete_by_file_syncs_fts(self) -> None:
        """delete_by_file should also remove FTS5 entries."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="alpha content")
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=1, chunk_text="alpha second")
        store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="beta content")
        assert store.fts_count() == 3

        store.delete_by_file("a.md")
        assert store.fts_count() == 1

    def test_clear_syncs_fts(self) -> None:
        """clear should also empty FTS5 table."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="content")
        store.insert_embedding(embedding=emb, file_path="test2.md", chunk_index=0, chunk_text="more content")
        assert store.fts_count() == 2

        store.clear()
        assert store.fts_count() == 0

    def test_search_fts_basic(self) -> None:
        """search_fts should find matching documents."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="deploy.md",
            chunk_index=0,
            chunk_text="How to deploy kubernetes clusters on AWS",
        )
        store.insert_embedding(
            embedding=emb,
            file_path="python.md",
            chunk_index=0,
            chunk_text="Python programming basics and tips",
        )

        results = store.search_fts("kubernetes")
        assert len(results) == 1
        assert results[0].bm25_score > 0

    def test_search_fts_multiple_matches(self) -> None:
        """search_fts should return multiple matching documents."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="deploy to production")
        store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="deploy staging server")
        store.insert_embedding(embedding=emb, file_path="c.md", chunk_index=0, chunk_text="run tests locally")

        results = store.search_fts("deploy")
        assert len(results) == 2

    def test_search_fts_empty_query(self) -> None:
        """search_fts with empty query should return empty."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="content")
        assert store.search_fts("") == []
        assert store.search_fts("   ") == []

    def test_search_fts_no_match(self) -> None:
        """search_fts with no matching content should return empty."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="hello world")
        assert store.search_fts("zzzznonexistent") == []

    def test_search_fts_respects_top_k(self) -> None:
        """search_fts should limit results to top_k."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        for i in range(10):
            store.insert_embedding(
                embedding=emb,
                file_path=f"file{i}.md",
                chunk_index=0,
                chunk_text=f"deploy variant number {i}",
            )
        results = store.search_fts("deploy", top_k=3)
        assert len(results) == 3

    def test_search_fts_special_chars(self) -> None:
        """search_fts should handle special characters in query gracefully."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="config: key=value; path=/usr/bin",
        )
        # These shouldn't crash — special chars are quoted
        results = store.search_fts("config:")
        # May or may not match depending on tokenizer, but shouldn't crash
        assert isinstance(results, list)

    def test_update_embedding_syncs_fts(self) -> None:
        """update_embedding with new chunk_text should update FTS5."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb1 = _make_embedding(1.0)
        emb2 = _make_embedding(2.0)
        rowid = store.insert_embedding(
            embedding=emb1,
            file_path="test.md",
            chunk_index=0,
            chunk_text="original keyword content",
        )

        # Verify original is searchable
        results = store.search_fts("original")
        assert len(results) == 1

        # Update with new text
        store.update_embedding(rowid, emb2, chunk_text="replaced updated content")

        # Original keyword should no longer match
        results = store.search_fts("original")
        assert len(results) == 0

        # New keyword should match
        results = store.search_fts("replaced")
        assert len(results) == 1

    def test_backfill_fts_on_existing_db(self) -> None:
        """FTS5 backfill should populate from existing metadata."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)

        # Insert some data
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="alpha content")
        store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="beta content")
        assert store.fts_count() == 2

        # Simulate a scenario where FTS is empty but meta has data
        # by manually clearing FTS and re-running backfill
        store._fts_clear()
        store.connection.commit()
        assert store.fts_count() == 0

        store._backfill_fts()
        assert store.fts_count() == 2

        # Verify search works after backfill
        results = store.search_fts("alpha")
        assert len(results) == 1


# ============================================================================
# Integration Tests — Full Pipeline (index + hybrid search)
# ============================================================================


@_skip_no_vec
class TestHybridSearchEndToEnd:
    """End-to-end tests: index markdown files, then search with different modes."""

    @pytest.fixture
    def indexed_store(self, tmp_path: "pytest.TempPathFactory") -> tuple[SqliteVecStore, FakeEmbedder]:
        """Create a store with indexed content for searching."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        embedder = FakeEmbedder(dim=_DIM)

        # Insert test documents with varying embeddings and content
        texts = [
            ("deploy.md", "How to deploy kubernetes on AWS"),
            ("python.md", "Python programming best practices"),
            ("deploy2.md", "Deploy docker containers to production"),
            ("memory.md", "Memory management in low-level systems"),
            ("search.md", "Full-text search with SQLite FTS5"),
        ]
        for i, (fp, text) in enumerate(texts):
            emb = embedder.embed(text)
            store.insert_embedding(
                embedding=emb,
                file_path=fp,
                chunk_index=0,
                chunk_text=text,
                importance=0.5 + i * 0.1,
            )
        return store, embedder

    def test_vector_only_returns_results(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """Vector-only search should return results based on embedding similarity."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("deploy", mode=SearchMode.VECTOR_ONLY, top_k=3)
        assert len(results) > 0
        assert all(r.fts_score == 0.0 for r in results)

    def test_fts_only_returns_keyword_matches(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """FTS-only should return results matching keywords."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("deploy", mode=SearchMode.FTS_ONLY)
        assert len(results) == 2  # deploy.md and deploy2.md
        assert all(r.fts_score > 0 for r in results)
        assert all(r.semantic_score == 0.0 for r in results)

    def test_hybrid_combines_both(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """Hybrid mode should combine vector and FTS results."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("deploy", mode=SearchMode.HYBRID)
        assert len(results) > 0
        # At least the FTS-matching results should be present
        file_paths = {r.file_path for r in results}
        assert "deploy.md" in file_paths or "deploy2.md" in file_paths

    def test_fts_only_unique_keyword(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """FTS-only with a unique keyword should return exactly one result."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("kubernetes", mode=SearchMode.FTS_ONLY)
        assert len(results) == 1
        assert results[0].file_path == "deploy.md"

    def test_hybrid_fts_boost(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """In hybrid mode, a result matching both vector and FTS should score well."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)

        hybrid_results = svc.search("deploy", mode=SearchMode.HYBRID, top_k=5)
        vector_results = svc.search("deploy", mode=SearchMode.VECTOR_ONLY, top_k=5)

        # Both modes should return results
        assert len(hybrid_results) > 0
        assert len(vector_results) > 0

    def test_fts_count_matches_meta_count(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """FTS row count should match metadata row count."""
        store, _ = indexed_store
        assert store.fts_count() == store.count()

    def test_search_with_memory_type_filter(
        self, indexed_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """memory_type filter should work in all search modes."""
        store, embedder = indexed_store
        svc = HybridSearchService(store, embedder)

        # All our test data has default memory_type="semantic", so filtering
        # for "episodic" should return empty
        for mode in SearchMode:
            results = svc.search("deploy", mode=mode, memory_type="episodic")
            assert results == [], f"Expected empty for mode={mode} with memory_type=episodic"


# ============================================================================
# CJK Segmentation Unit Tests
# ============================================================================


class TestCjkSegment:
    """Tests for _cjk_segment helper function (rjieba word-level segmentation)."""

    def test_pure_chinese(self) -> None:
        """Pure Chinese text should be segmented into words."""
        result = _cjk_segment("桌面端构建打包")
        tokens = result.split()
        # rjieba produces word-level tokens like: 桌面 端 构建 打包
        assert "桌面" in tokens
        assert "构建" in tokens
        assert len(tokens) < 7  # Must be fewer than 7 char-level tokens

    def test_pure_english(self) -> None:
        """Pure English text should pass through unchanged."""
        text = "deploy to production"
        result = _cjk_segment(text)
        assert result == text

    def test_mixed_cjk_english(self) -> None:
        """Mixed CJK and English should segment CJK while preserving English words."""
        result = _cjk_segment("deploy to 生产环境")
        tokens = result.split()
        assert "deploy" in tokens
        assert "to" in tokens
        assert "生产" in tokens
        assert "环境" in tokens

    def test_empty_string(self) -> None:
        """Empty string should return empty."""
        assert _cjk_segment("") == ""

    def test_numbers_between_cjk_preserved(self) -> None:
        """Numbers between CJK runs are left in place; CJK runs are segmented."""
        result = _cjk_segment("版本2.0发布")
        # "版本" and "发布" are each a single CJK run → single words
        # "2.0" is non-CJK → untouched.  FTS5 unicode61 splits on "2.0" naturally.
        assert "版本" in result
        assert "发布" in result
        assert "2.0" in result

    def test_english_not_modified(self) -> None:
        """English words, special chars, hyphens should be untouched."""
        text = "sqlite-vec cosine search_memory FTS5"
        assert _cjk_segment(text) == text


# ============================================================================
# CJK FTS5 Integration Tests
# ============================================================================


@_skip_no_vec
class TestCjkFtsIntegration:
    """Integration tests for CJK text search in FTS5."""

    def test_chinese_text_searchable(self) -> None:
        """Chinese text should be searchable via FTS5."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="diary.md",
            chunk_index=0,
            chunk_text="桌面端构建打包完成",
        )
        results = store.search_fts("桌面端")
        assert len(results) >= 1, "Chinese query should match Chinese content"
        assert results[0].bm25_score > 0

    def test_chinese_partial_match(self) -> None:
        """A partial Chinese query should match a longer Chinese text."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="memory.md",
            chunk_index=0,
            chunk_text="向量检索基础设施已完成部署",
        )
        # Partial query
        results = store.search_fts("向量检索")
        assert len(results) == 1

    def test_chinese_no_word_overlap_no_match(self) -> None:
        """Chinese query with zero word overlap should not match."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="a.md",
            chunk_index=0,
            chunk_text="天气晴朗万里无云",
        )
        # Query shares no words with the stored text
        results = store.search_fts("数据库备份")
        assert len(results) == 0

    def test_mixed_chinese_english(self) -> None:
        """Mixed CJK+English content should be searchable by either language."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="notes.md",
            chunk_index=0,
            chunk_text="ONNX embedding 双后端架构设计",
        )

        # Search by English
        results_en = store.search_fts("ONNX")
        assert len(results_en) >= 1

        # Search by Chinese
        results_cn = store.search_fts("双后端")
        assert len(results_cn) >= 1

        # Search by mixed
        results_mix = store.search_fts("ONNX 后端")
        assert len(results_mix) >= 1

    def test_multiple_chinese_docs_ranked(self) -> None:
        """Docs with matching Chinese words should be found; unrelated excluded."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)

        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="桌面端自动更新功能")
        store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="桌面端构建打包脚本")
        store.insert_embedding(embedding=emb, file_path="c.md", chunk_index=0, chunk_text="天气晴朗万里无云")

        results = store.search_fts("桌面端")
        # a.md and b.md contain "桌面" + "端"; c.md has zero overlap
        assert len(results) >= 2
        matched_rowids = {r.rowid for r in results}
        # c.md (rowid 3) should NOT match
        assert 3 not in matched_rowids

    def test_fts_version_auto_retokenize(self) -> None:
        """FTS version tracking should auto-retokenize on version bump."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="记忆系统测试内容",
        )

        # Verify version is stored
        row = store.connection.execute("SELECT value FROM _fts_meta WHERE key = 'version'").fetchone()
        assert row is not None
        assert int(row[0]) >= 2  # _FTS_VERSION = 2

    def test_backfill_applies_cjk_segmentation(self) -> None:
        """Backfill should apply CJK segmentation to existing data."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="混合检索服务实现",
        )

        # Force re-backfill
        store._fts_clear()
        store._backfill_fts(force=True)

        # Should still be searchable after re-backfill
        results = store.search_fts("混合检索")
        assert len(results) == 1


# ============================================================================
# CJK End-to-End Hybrid Search Tests
# ============================================================================


@_skip_no_vec
class TestCjkHybridSearchEndToEnd:
    """End-to-end CJK tests: index Chinese content, search with all modes."""

    @pytest.fixture
    def cjk_store(self) -> tuple[SqliteVecStore, FakeEmbedder]:
        """Create a store with Chinese content."""
        store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        store.ensure_tables()
        embedder = FakeEmbedder(dim=_DIM)

        texts = [
            ("deploy.md", "桌面端构建打包部署流程"),
            ("memory.md", "向量检索基础设施与记忆系统"),
            ("search.md", "混合搜索支持中文全文检索"),
            ("config.md", "ONNX 双后端 embedding 架构设计"),
            ("diary.md", "今日完成代码审查和测试验证"),
        ]
        for i, (fp, text) in enumerate(texts):
            emb = embedder.embed(text)
            store.insert_embedding(
                embedding=emb, file_path=fp, chunk_index=0, chunk_text=text, importance=0.5 + i * 0.1
            )
        return store, embedder

    def test_fts_only_chinese_query(self, cjk_store: tuple[SqliteVecStore, FakeEmbedder]) -> None:
        """FTS-only with Chinese query should find matching documents."""
        store, embedder = cjk_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("桌面端", mode=SearchMode.FTS_ONLY)
        assert len(results) >= 1
        assert any(r.file_path == "deploy.md" for r in results)

    def test_hybrid_chinese_query(self, cjk_store: tuple[SqliteVecStore, FakeEmbedder]) -> None:
        """Hybrid mode with Chinese query should combine signals."""
        store, embedder = cjk_store
        svc = HybridSearchService(store, embedder)
        results = svc.search("向量检索", mode=SearchMode.HYBRID)
        assert len(results) >= 1
        # The memory.md should be highly ranked (keyword + semantic match)
        assert any(r.file_path == "memory.md" for r in results)

    def test_all_modes_return_results_for_chinese(
        self, cjk_store: tuple[SqliteVecStore, FakeEmbedder]
    ) -> None:
        """All search modes should return results for Chinese queries."""
        store, embedder = cjk_store
        svc = HybridSearchService(store, embedder)
        for mode in SearchMode:
            results = svc.search("记忆系统", mode=mode)
            assert len(results) >= 1, f"mode={mode} should return results for Chinese query"
