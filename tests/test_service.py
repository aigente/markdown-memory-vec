# pyright: reportPrivateUsage=false, reportGeneralTypeIssues=false
"""
Tests for MemoryVectorService: rebuild, incremental, search, stats.

Uses mocked components to avoid downloading real models.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from memory_vec.service import MemoryVectorService


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary memory directory with .md files.

    Returns the memory directory itself (not a workspace root), matching
    the ``MemoryVectorService(memory_dir=...)`` contract.
    """
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True)

    (mem_dir / "Memory.md").write_text("# Shared Memory\n\nSome shared content.")
    personal_dir = mem_dir / "personal"
    personal_dir.mkdir()
    (personal_dir / "Memory.md").write_text("# Personal Memory\n\nSome personal content.")

    return mem_dir


class TestMemoryVectorServiceInit:
    """Tests for service initialization."""

    def test_memory_dir_is_used_directly(self, workspace: Path) -> None:
        """memory_dir should be used as-is — no subdirectory appended."""
        svc = MemoryVectorService(workspace)
        assert svc.memory_root == workspace
        assert svc.db_path == workspace / "vector_index.db"

    def test_custom_memory_dir(self, tmp_path: Path) -> None:
        """An arbitrary directory can be used as memory_dir."""
        custom_dir = tmp_path / "docs" / "memory"
        custom_dir.mkdir(parents=True)
        svc = MemoryVectorService(custom_dir)
        assert svc.memory_root == custom_dir

    def test_is_available_when_deps_missing(self) -> None:
        """is_available should return False when dependencies are missing."""
        svc = MemoryVectorService("/tmp/nonexistent")
        svc._available = False
        assert not svc.is_available


class TestMemoryVectorServiceOperations:
    """Tests for service operations with mocked dependencies."""

    def test_rebuild_returns_zero_when_not_initialized(self, workspace: Path) -> None:
        """rebuild_index should return 0 when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.rebuild_index()
            assert result == 0

    def test_incremental_returns_zero_when_not_initialized(self, workspace: Path) -> None:
        """incremental_index should return 0 when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.incremental_index()
            assert result == 0

    def test_search_returns_empty_when_not_initialized(self, workspace: Path) -> None:
        """search should return empty list when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.search("test query")
            assert result == []

    def test_index_file_returns_zero_when_not_initialized(self, workspace: Path) -> None:
        """index_file should return 0 when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.index_file("test.md")
            assert result == 0

    def test_remove_file_returns_zero_when_not_initialized(self, workspace: Path) -> None:
        """remove_file should return 0 when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.remove_file("test.md")
            assert result == 0

    def test_stats_returns_error_when_not_initialized(self, workspace: Path) -> None:
        """stats should indicate unavailable when initialization fails."""
        svc = MemoryVectorService(workspace)
        with patch.object(svc, "_ensure_initialized", return_value=False):
            result = svc.stats()
            assert result["available"] is False

    def test_close_is_safe_when_not_initialized(self, workspace: Path) -> None:
        """close should not error when store is None."""
        svc = MemoryVectorService(workspace)
        svc.close()  # Should not raise

    def test_rebuild_delegates_to_indexer(self, workspace: Path) -> None:
        """rebuild_index should call indexer.reindex_all."""
        svc = MemoryVectorService(workspace)
        mock_indexer = MagicMock()
        mock_indexer.reindex_all.return_value = 42
        svc._indexer = mock_indexer
        svc._store = MagicMock()  # Prevent _ensure_initialized from running

        with patch.object(svc, "_ensure_initialized", return_value=True):
            result = svc.rebuild_index()
            assert result == 42
            mock_indexer.reindex_all.assert_called_once()

    def test_search_delegates_to_search_service(self, workspace: Path) -> None:
        """search should call search_service.search and format results."""
        svc = MemoryVectorService(workspace)
        mock_search = MagicMock()

        # Create a mock SearchResult
        mock_result = MagicMock()
        mock_result.file_path = "test.md"
        mock_result.chunk_text = "Test content"
        mock_result.chunk_index = 0
        mock_result.hybrid_score = 0.85
        mock_result.semantic_score = 0.9
        mock_result.importance = 0.7
        mock_result.temporal_decay = 0.8
        mock_result.memory_type = "semantic"
        mock_result.tags = ["tag1"]

        mock_search.search.return_value = [mock_result]
        svc._search_service = mock_search
        svc._store = MagicMock()  # Prevent _ensure_initialized from running

        with patch.object(svc, "_ensure_initialized", return_value=True):
            results = svc.search("test query", top_k=5)
            assert len(results) == 1
            assert results[0]["file_path"] == "test.md"
            assert results[0]["hybrid_score"] == 0.85
            assert results[0]["tags"] == ["tag1"]


class TestMemoryVectorServiceEnsureInit:
    """Tests for the lazy initialization logic."""

    def test_ensure_initialized_returns_false_without_memory_dir(self, tmp_path: Path) -> None:
        """_ensure_initialized should return False when memory dir doesn't exist."""
        svc = MemoryVectorService(tmp_path / "nonexistent")
        # Mock is_available to True so we test the directory check
        svc._available = True
        result = svc._ensure_initialized()
        assert result is False

    def test_ensure_initialized_caches_result(self, workspace: Path) -> None:
        """Second call to _ensure_initialized should return True immediately (cached)."""
        svc = MemoryVectorService(workspace)
        svc._store = MagicMock()  # Simulate already initialized
        assert svc._ensure_initialized() is True
