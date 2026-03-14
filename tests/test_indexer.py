# pyright: reportUnusedVariable=false, reportArgumentType=false
"""
Tests for MemoryIndexer: chunking, frontmatter parsing, and indexing pipeline.

Uses FakeEmbedder and either FakeVecStore or real SqliteVecStore (if available).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from memory_vec.indexer import MemoryIndexer, chunk_text, parse_frontmatter
from memory_vec.store import SqliteVecStore, is_sqlite_vec_available

from .conftest import FakeEmbedder

# ============================================================================
# Chunking Tests (no dependencies needed)
# ============================================================================


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_empty_text_returns_empty(self) -> None:
        """Empty or whitespace-only text should return empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than chunk_size should be a single chunk."""
        result = chunk_text("Hello world", chunk_size=100)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        # Create text with many paragraphs
        paragraphs = [f"Paragraph {i} with some content." for i in range(50)]
        text = "\n\n".join(paragraphs)

        result = chunk_text(text, chunk_size=200, overlap_size=50)
        assert len(result) > 1

    def test_chunks_have_overlap(self) -> None:
        """Adjacent chunks should share some content (overlap)."""
        # Use short paragraphs so they fit within overlap_size
        paragraphs = [f"Para {i}: " + "a" * 30 for i in range(30)]
        text = "\n\n".join(paragraphs)

        result = chunk_text(text, chunk_size=300, overlap_size=200)
        assert len(result) >= 2

        # At least some adjacent chunks should share content
        overlap_found = False
        for i in range(len(result) - 1):
            paras_current = set(result[i].split("\n\n"))
            paras_next = set(result[i + 1].split("\n\n"))
            if paras_current & paras_next:
                overlap_found = True
                break
        assert overlap_found, "No overlap found between any adjacent chunks"

    def test_paragraph_boundary_splitting(self) -> None:
        """Chunks should split on paragraph boundaries (\\n\\n)."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunk_text(text, chunk_size=50, overlap_size=10)
        # Each chunk should not split mid-paragraph
        for chunk in result:
            # No truncated paragraph (all paragraphs should be complete)
            assert not chunk.startswith(" ")


# ============================================================================
# Frontmatter Parsing Tests
# ============================================================================


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_no_frontmatter(self) -> None:
        """Text without frontmatter should return empty dict."""
        meta, body = parse_frontmatter("# Hello\nWorld")
        assert meta == {}
        assert body == "# Hello\nWorld"

    def test_valid_frontmatter(self) -> None:
        """Valid YAML frontmatter should be parsed correctly."""
        text = "---\nimportance: 0.9\ntype: episodic\ntags:\n  - daily\n  - work\n---\n# Content\nBody text"
        meta, body = parse_frontmatter(text)
        assert meta["importance"] == 0.9
        assert meta["type"] == "episodic"
        assert meta["tags"] == ["daily", "work"]
        assert body.startswith("# Content")

    def test_invalid_yaml_returns_empty(self) -> None:
        """Invalid YAML in frontmatter should return empty dict."""
        text = "---\n: invalid: yaml: {{{\n---\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Body"

    def test_non_dict_frontmatter(self) -> None:
        """Non-dict YAML (e.g., a list) should return empty dict."""
        text = "---\n- item1\n- item2\n---\nBody"
        meta, _body = parse_frontmatter(text)
        assert meta == {}


# ============================================================================
# MemoryIndexer Tests (requires sqlite-vec)
# ============================================================================

pytestmark_vec = pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")

_DIM = 4


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Create a temporary memory directory with some .md files."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    (mem_dir / "file1.md").write_text("# File 1\n\nFirst paragraph of file 1.\n\nSecond paragraph.")
    (mem_dir / "file2.md").write_text(
        "---\nimportance: 0.9\ntype: episodic\n---\n# File 2\n\nContent with frontmatter."
    )

    sub_dir = mem_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "nested.md").write_text("# Nested\n\nNested file content.")

    return mem_dir


@pytest.fixture
def store_and_indexer(tmp_memory_dir: Path) -> tuple[SqliteVecStore, MemoryIndexer, Path]:
    """Create store + indexer for testing."""
    if not is_sqlite_vec_available():
        pytest.skip("sqlite-vec not installed")

    store = SqliteVecStore(db_path=":memory:", dimension=_DIM)
    store.ensure_tables()
    embedder = FakeEmbedder(dim=_DIM)
    indexer = MemoryIndexer(store, embedder, memory_root=tmp_memory_dir)  # type: ignore[arg-type]
    return store, indexer, tmp_memory_dir


@pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")
class TestMemoryIndexerFile:
    """Tests for indexing individual files."""

    def test_index_single_file(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """index_file should create chunks in the store."""
        store, indexer, mem_dir = store_and_indexer
        count = indexer.index_file(mem_dir / "file1.md")
        assert count > 0
        assert store.count() > 0

    def test_index_file_uses_relative_path(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """File paths stored should be relative to memory_root."""
        store, indexer, mem_dir = store_and_indexer
        indexer.index_file(mem_dir / "file1.md")

        # Check stored file_path is relative
        hashes = store.get_hashes_for_file("file1.md")
        assert len(hashes) > 0

    def test_index_file_dedup(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """Indexing the same file twice should not create duplicate chunks."""
        store, indexer, mem_dir = store_and_indexer
        _count1 = indexer.index_file(mem_dir / "file1.md")
        count_after_first = store.count()

        count2 = indexer.index_file(mem_dir / "file1.md")
        assert count2 == 0  # No new chunks
        assert store.count() == count_after_first

    def test_index_file_updates_changed_content(
        self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]
    ) -> None:
        """Changed file content should be re-indexed."""
        _store, indexer, mem_dir = store_and_indexer
        indexer.index_file(mem_dir / "file1.md")

        # Modify the file
        (mem_dir / "file1.md").write_text("# File 1 Updated\n\nCompletely new content.")
        count = indexer.index_file(mem_dir / "file1.md")
        assert count > 0  # Should re-embed changed chunks

    def test_index_nonexistent_file(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """Indexing a non-existent file should return 0."""
        _, indexer, mem_dir = store_and_indexer
        count = indexer.index_file(mem_dir / "nonexistent.md")
        assert count == 0

    def test_index_file_with_frontmatter(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """Files with frontmatter should use metadata for importance/type."""
        store, indexer, mem_dir = store_and_indexer
        indexer.index_file(mem_dir / "file2.md")

        # Check that the stored metadata reflects frontmatter
        meta = store.get_meta(1)
        assert meta is not None
        assert meta["importance"] == 0.9
        assert meta["memory_type"] == "episodic"


@pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")
class TestMemoryIndexerDirectory:
    """Tests for directory-level indexing."""

    def test_index_directory(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """index_directory should index all .md files recursively."""
        store, indexer, mem_dir = store_and_indexer
        total = indexer.index_directory(mem_dir)
        assert total > 0
        # Should have indexed file1.md, file2.md, and sub/nested.md
        assert store.count() >= 3

    def test_index_directory_nonexistent(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """index_directory on non-existent dir should return 0."""
        _, indexer, _ = store_and_indexer
        assert indexer.index_directory("/nonexistent/path") == 0

    def test_reindex_all(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """reindex_all should clear and rebuild."""
        store, indexer, mem_dir = store_and_indexer

        # First index
        indexer.index_directory(mem_dir)
        count_before = store.count()
        assert count_before > 0

        # Reindex should clear and rebuild
        total = indexer.reindex_all(mem_dir)
        assert total > 0
        assert store.count() == total


@pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec not installed")
class TestMemoryIndexerRemove:
    """Tests for removing files from the index."""

    def test_remove_file(self, store_and_indexer: tuple[SqliteVecStore, MemoryIndexer, Path]) -> None:
        """remove_file should delete all chunks for a file."""
        store, indexer, mem_dir = store_and_indexer
        indexer.index_file(mem_dir / "file1.md")
        indexer.index_file(mem_dir / "file2.md")
        count_before = store.count()
        assert count_before > 0

        deleted = indexer.remove_file(mem_dir / "file1.md")
        assert deleted > 0
        assert store.count() < count_before
