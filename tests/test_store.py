"""
Tests for SqliteVecStore: CRUD operations, search, and metadata management.

Uses real sqlite-vec (in-memory DB) — no mocks needed for store tests.
"""

from __future__ import annotations

import pytest
from memory_vec.store import MemoryVecMeta, SqliteVecStore, content_hash, is_sqlite_vec_available

_skip_no_vec = pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")

_DIM = 4


def _make_embedding(seed: float = 1.0) -> list[float]:
    """Create a simple test embedding."""
    return [seed * (i + 1) / _DIM for i in range(_DIM)]


@pytest.fixture
def store() -> SqliteVecStore:
    """Create an in-memory SqliteVecStore for testing."""
    s = SqliteVecStore(db_path=":memory:", dimension=_DIM)
    s.ensure_tables()
    return s


@_skip_no_vec
class TestStoreCreation:
    """Tests for store initialization and table creation."""

    def test_ensure_tables_creates_schema(self, store: SqliteVecStore) -> None:
        """ensure_tables should create both vec0 and metadata tables."""
        # Check vec0 table exists
        row = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_vec'"
        ).fetchone()
        assert row is not None

        # Check metadata table exists
        row = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_vec_meta'"
        ).fetchone()
        assert row is not None

    def test_ensure_tables_idempotent(self, store: SqliteVecStore) -> None:
        """Calling ensure_tables twice should not error."""
        store.ensure_tables()  # Second call
        assert store.count() == 0

    def test_count_empty(self, store: SqliteVecStore) -> None:
        """Empty store should have count 0."""
        assert store.count() == 0


@_skip_no_vec
class TestStoreInsertAndRetrieve:
    """Tests for insert_embedding and get_meta."""

    def test_insert_and_count(self, store: SqliteVecStore) -> None:
        """Inserting a record should increase count."""
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(
            embedding=emb,
            file_path="test.md",
            chunk_index=0,
            chunk_text="Hello world",
        )
        assert rowid > 0
        assert store.count() == 1

    def test_insert_returns_valid_rowid(self, store: SqliteVecStore) -> None:
        """Each insert should return a unique rowid."""
        emb = _make_embedding(1.0)
        id1 = store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="chunk A")
        id2 = store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="chunk B")
        assert id1 != id2
        assert store.count() == 2

    def test_get_meta_returns_correct_data(self, store: SqliteVecStore) -> None:
        """get_meta should return all fields correctly."""
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(
            embedding=emb,
            file_path="test/file.md",
            chunk_index=3,
            chunk_text="Test content",
            hash_value="abc123",
            importance=0.8,
            memory_type="episodic",
            tags=["tag1", "tag2"],
        )
        meta = store.get_meta(rowid)
        assert meta is not None
        assert meta["file_path"] == "test/file.md"
        assert meta["chunk_index"] == 3
        assert meta["chunk_text"] == "Test content"
        assert meta["content_hash"] == "abc123"
        assert meta["importance"] == 0.8
        assert meta["memory_type"] == "episodic"
        assert meta["tags"] == ["tag1", "tag2"]
        assert meta["created_at"] is not None

    def test_get_meta_nonexistent_returns_none(self, store: SqliteVecStore) -> None:
        """get_meta for non-existent rowid should return None."""
        assert store.get_meta(9999) is None

    def test_auto_hash_when_not_provided(self, store: SqliteVecStore) -> None:
        """If no hash_value is provided, it should be computed automatically."""
        emb = _make_embedding(1.0)
        text = "Test auto hash"
        rowid = store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text=text)
        meta = store.get_meta(rowid)
        assert meta is not None
        assert meta["content_hash"] == content_hash(text)


@_skip_no_vec
class TestStoreDelete:
    """Tests for delete operations."""

    def test_delete_embedding(self, store: SqliteVecStore) -> None:
        """delete_embedding should remove both vec and meta."""
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="content")
        assert store.count() == 1

        store.delete_embedding(rowid)
        assert store.count() == 0
        assert store.get_meta(rowid) is None

    def test_delete_by_file(self, store: SqliteVecStore) -> None:
        """delete_by_file should remove all chunks for a file."""
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="target.md", chunk_index=0, chunk_text="chunk 0")
        store.insert_embedding(embedding=emb, file_path="target.md", chunk_index=1, chunk_text="chunk 1")
        store.insert_embedding(embedding=emb, file_path="other.md", chunk_index=0, chunk_text="other")
        assert store.count() == 3

        deleted = store.delete_by_file("target.md")
        assert deleted == 2
        assert store.count() == 1

    def test_delete_by_file_nonexistent(self, store: SqliteVecStore) -> None:
        """delete_by_file for non-existent file should return 0."""
        assert store.delete_by_file("nonexistent.md") == 0

    def test_clear(self, store: SqliteVecStore) -> None:
        """clear should remove all records."""
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="a.md", chunk_index=0, chunk_text="a")
        store.insert_embedding(embedding=emb, file_path="b.md", chunk_index=0, chunk_text="b")
        assert store.count() == 2

        store.clear()
        assert store.count() == 0

    def test_delete_via_interface(self, store: SqliteVecStore) -> None:
        """ISqliteVecStore.delete() should work with string IDs."""
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="content")
        store.delete([str(rowid)])
        assert store.count() == 0


@_skip_no_vec
class TestStoreSearch:
    """Tests for search operations."""

    def test_search_similar_returns_results(self, store: SqliteVecStore) -> None:
        """search_similar should find inserted vectors."""
        emb1 = _make_embedding(1.0)
        emb2 = _make_embedding(2.0)
        store.insert_embedding(embedding=emb1, file_path="a.md", chunk_index=0, chunk_text="first")
        store.insert_embedding(embedding=emb2, file_path="b.md", chunk_index=0, chunk_text="second")

        results = store.search_similar(emb1, k=2)
        assert len(results) == 2
        # First result should be closest to emb1
        assert results[0].distance <= results[1].distance

    def test_search_interface(self, store: SqliteVecStore) -> None:
        """ISqliteVecStore.search() should return VectorSearchResult objects."""
        emb = _make_embedding(1.0)
        store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="test content")

        results = store.search(query_embedding=emb, top_k=1)
        assert len(results) == 1
        assert results[0].metadata["file_path"] == "test.md"
        assert results[0].metadata["chunk_text"] == "test content"

    def test_search_with_metadata_filter(self, store: SqliteVecStore) -> None:
        """search with filter_metadata should filter results."""
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb, file_path="a.md", chunk_index=0, chunk_text="chunk a", memory_type="semantic"
        )
        store.insert_embedding(
            embedding=emb, file_path="b.md", chunk_index=0, chunk_text="chunk b", memory_type="episodic"
        )

        results = store.search(query_embedding=emb, top_k=10, filter_metadata={"memory_type": "episodic"})
        assert len(results) == 1
        assert results[0].metadata["memory_type"] == "episodic"


@_skip_no_vec
class TestStoreUpdate:
    """Tests for update operations."""

    def test_update_embedding(self, store: SqliteVecStore) -> None:
        """update_embedding should replace the vector and update metadata."""
        emb1 = _make_embedding(1.0)
        emb2 = _make_embedding(3.0)
        rowid = store.insert_embedding(embedding=emb1, file_path="test.md", chunk_index=0, chunk_text="original")

        store.update_embedding(rowid, emb2, chunk_text="updated", hash_value="newhash")
        meta = store.get_meta(rowid)
        assert meta is not None
        assert meta["chunk_text"] == "updated"
        assert meta["content_hash"] == "newhash"
        assert store.count() == 1  # Still just one record

    def test_update_nonexistent_is_noop(self, store: SqliteVecStore) -> None:
        """update_embedding for non-existent row should be a no-op."""
        emb = _make_embedding(1.0)
        store.update_embedding(9999, emb)  # Should not raise
        assert store.count() == 0


@_skip_no_vec
class TestStoreHashes:
    """Tests for hash-related operations."""

    def test_get_hashes_for_file(self, store: SqliteVecStore) -> None:
        """get_hashes_for_file should return chunk_index -> content_hash mapping."""
        emb = _make_embedding(1.0)
        store.insert_embedding(
            embedding=emb, file_path="test.md", chunk_index=0, chunk_text="chunk 0", hash_value="hash0"
        )
        store.insert_embedding(
            embedding=emb, file_path="test.md", chunk_index=1, chunk_text="chunk 1", hash_value="hash1"
        )
        store.insert_embedding(
            embedding=emb, file_path="other.md", chunk_index=0, chunk_text="other", hash_value="hashx"
        )

        hashes = store.get_hashes_for_file("test.md")
        assert hashes == {0: "hash0", 1: "hash1"}

    def test_record_access(self, store: SqliteVecStore) -> None:
        """record_access should bump access_count and update last_accessed."""
        emb = _make_embedding(1.0)
        rowid = store.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="content")

        meta_before = store.get_meta(rowid)
        assert meta_before is not None
        assert meta_before["access_count"] == 0

        store.record_access(rowid)
        meta_after = store.get_meta(rowid)
        assert meta_after is not None
        assert meta_after["access_count"] == 1
        assert meta_after["last_accessed"] is not None


@_skip_no_vec
class TestStoreLifecycle:
    """Tests for connection lifecycle."""

    def test_close_and_reopen(self) -> None:
        """Closing and accessing connection should work (lazy reconnect for in-memory is a new db)."""
        s = SqliteVecStore(db_path=":memory:", dimension=_DIM)
        s.ensure_tables()
        emb = _make_embedding(1.0)
        s.insert_embedding(embedding=emb, file_path="test.md", chunk_index=0, chunk_text="content")
        assert s.count() == 1

        s.close()
        # After close, connection is None — accessing creates a new in-memory DB
        # (this is expected behavior for :memory: databases)


class TestContentHash:
    """Tests for the content_hash helper."""

    def test_deterministic(self) -> None:
        """Same input should produce same hash."""
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self) -> None:
        """Different inputs should produce different hashes."""
        assert content_hash("hello") != content_hash("world")

    def test_sha256_length(self) -> None:
        """Hash should be 64 hex characters (SHA-256)."""
        h = content_hash("test")
        assert len(h) == 64


class TestMemoryVecMeta:
    """Tests for the MemoryVecMeta dataclass."""

    def test_default_values(self) -> None:
        """MemoryVecMeta should have sensible defaults."""
        meta = MemoryVecMeta()
        assert meta.id == 0
        assert meta.file_path == ""
        assert meta.chunk_index == 0
        assert meta.importance == 0.5
        assert meta.access_count == 0
        assert meta.tags == []

    def test_custom_values(self) -> None:
        """MemoryVecMeta should accept custom values."""
        meta = MemoryVecMeta(
            id=1,
            file_path="test.md",
            chunk_index=2,
            content_hash="abc",
            chunk_text="hello",
            importance=0.9,
            memory_type="episodic",
            tags=["tag1"],
        )
        assert meta.id == 1
        assert meta.file_path == "test.md"
        assert meta.memory_type == "episodic"
        assert meta.tags == ["tag1"]
