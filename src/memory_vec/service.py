"""
Memory vector index maintenance service.

This is the **integration glue** that connects the vector infrastructure
(store, embedder, indexer, search) to the actual memory system.

Key responsibilities:
1. Build / rebuild the vector index from Markdown memory files
2. Incremental re-index when memory files change
3. Search memories by semantic query
4. Provide a CLI-friendly entry point for cron tasks

Usage from Python::

    from memory_vec import MemoryVectorService

    svc = MemoryVectorService("/path/to/project/.claude/memory")
    svc.rebuild_index()  # Full rebuild
    svc.incremental_index()  # Only changed files
    results = svc.search("query")  # Hybrid search
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default DB filename inside the memory directory
_DEFAULT_DB_NAME = "vector_index.db"


class MemoryVectorService:
    """High-level service that wires up all vector components for a directory.

    This is the single entry point for all vector-related operations on a
    directory of Markdown files.

    Parameters
    ----------
    memory_dir:
        The directory containing Markdown files to index and search.
        The caller is responsible for resolving the full path — this class
        does **not** append any subdirectory.
    db_name:
        SQLite database filename (created inside *memory_dir*).
    model_name:
        Sentence-transformer model name for embeddings.
    """

    def __init__(
        self,
        memory_dir: str | Path,
        db_name: str = _DEFAULT_DB_NAME,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        file_extensions: Optional[list[str]] = None,
    ) -> None:
        self._memory_root = Path(memory_dir)
        self._db_path = self._memory_root / db_name
        self._model_name = model_name
        self._file_extensions = file_extensions  # None = .md only (backward compat)

        # Lazy-initialized components
        self._store: Optional[object] = None
        self._embedder: Optional[object] = None
        self._indexer: Optional[object] = None
        self._search_service: Optional[object] = None
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if vector dependencies are installed.

        Returns ``True`` if sqlite-vec is available AND at least one embedding
        backend (ONNX or sentence-transformers) is available.
        """
        if self._available is None:
            try:
                from .embedder import is_any_backend_available
                from .store import is_sqlite_vec_available

                self._available = is_sqlite_vec_available() and is_any_backend_available()
            except ImportError:
                self._available = False
        return self._available

    @property
    def memory_root(self) -> Path:
        """Return the memory root directory."""
        return self._memory_root

    @property
    def db_path(self) -> Path:
        """Return the path to the vector index database."""
        return self._db_path

    def _ensure_initialized(self) -> bool:
        """Lazy-initialize all vector components.

        Returns True if initialization succeeded.
        """
        if self._store is not None:
            return True

        if not self.is_available:
            logger.warning(
                "Vector dependencies not available. Install with:\n"
                "  pip install 'markdown-memory-vec[onnx]'   # lightweight\n"
                "  pip install 'markdown-memory-vec[vector]'  # full"
            )
            return False

        if not self._memory_root.is_dir():
            logger.warning("Memory directory does not exist: %s", self._memory_root)
            return False

        try:
            from .embedder import create_embedder
            from .indexer import MemoryIndexer
            from .search import HybridSearchService
            from .store import SqliteVecStore

            self._embedder = create_embedder(model_name=self._model_name)

            # Dynamic dimension: ask the embedder for the actual output dimension
            dimension = self._embedder.dimension

            self._store = SqliteVecStore(db_path=self._db_path, dimension=dimension)
            self._store.ensure_tables()  # type: ignore[union-attr]
            self._indexer = MemoryIndexer(  # type: ignore[arg-type]
                store=self._store,
                embedder=self._embedder,
                memory_root=self._memory_root,
            )
            self._search_service = HybridSearchService(
                vec_store=self._store,
                embedder=self._embedder,
            )
            return True
        except Exception:
            logger.error("Failed to initialize vector components", exc_info=True)
            return False

    # =========================================================================
    # Indexing operations
    # =========================================================================

    def rebuild_index(self) -> int:
        """Drop and rebuild the entire vector index from all indexed files.

        Returns the total number of chunks indexed.

        Note: ``reindex_all`` already recursively indexes the entire
        memory root (including ``personal/``), so there is no need to
        index ``personal/`` separately.
        """
        if not self._ensure_initialized():
            return 0

        from .indexer import MemoryIndexer

        indexer: MemoryIndexer = self._indexer  # type: ignore[assignment]

        total = indexer.reindex_all(self._memory_root, file_extensions=self._file_extensions)
        self._refresh_file_states()
        logger.info("Full rebuild complete: %d chunks indexed", total)
        return total

    def incremental_index(self) -> int:
        """Re-index only changed files (based on SHA-256 hash comparison).

        This is much faster than rebuild_index() for daily maintenance.
        Returns the number of new/updated chunks.
        """
        if not self._ensure_initialized():
            return 0

        from .indexer import MemoryIndexer
        from .store import SqliteVecStore

        indexer: MemoryIndexer = self._indexer  # type: ignore[assignment]
        store: SqliteVecStore = self._store  # type: ignore[assignment]

        # ── Migration: clean up stale absolute-path entries ──
        # Older versions stored file_path as absolute paths.  Now we store
        # relative paths (relative to memory_root).  Delete any entries whose
        # file_path starts with '/' to avoid duplicates during incremental runs.
        stale_rows = store.connection.execute(
            "SELECT DISTINCT file_path FROM memory_vec_meta WHERE file_path LIKE '/%'"
        ).fetchall()
        if stale_rows:
            stale_count = 0
            for (fp,) in stale_rows:
                stale_count += store.delete_by_file(fp)
            if stale_count:
                logger.info(
                    "Cleaned up %d stale entries with absolute paths (%d files)",
                    stale_count,
                    len(stale_rows),
                )

        total = 0
        extensions = self._file_extensions or [".md"]
        for f in sorted(self._memory_root.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensions:
                total += indexer.index_file(f)
        self._refresh_file_states()

        if total:
            logger.info("Incremental index: %d chunks updated", total)
        else:
            logger.info("Incremental index: all chunks up to date")
        return total

    def sync(self, max_changed_files: Optional[int] = None) -> dict[str, object]:
        """Fast freshness sync — index changed files, drop deleted ones.

        Unlike :meth:`incremental_index` (which reads and hashes *every*
        file), this compares cheap ``(mtime_ns, size)`` fingerprints stored
        in the ``memory_file_state`` table against the filesystem, so an
        unchanged memory directory costs only a ``rglob`` + one SQL query.

        Designed to be called at the start of every search so results never
        come from a stale index.

        Parameters
        ----------
        max_changed_files:
            Safety valve for very large knowledge bases.  When more files
            changed than this limit, only the first *max_changed_files*
            (most recently modified) are re-indexed; the rest are left for
            a full :meth:`incremental_index` run.  ``None`` = no limit.

        Returns a summary dict::

            {"indexed_files": 2, "updated_chunks": 7, "removed_files": 1, "skipped_files": 0, "elapsed_seconds": 0.31}
        """
        t0 = time.monotonic()
        empty: dict[str, object] = {
            "indexed_files": 0,
            "updated_chunks": 0,
            "removed_files": 0,
            "skipped_files": 0,
            "elapsed_seconds": 0.0,
        }
        if not self._ensure_initialized():
            return empty

        from .indexer import MemoryIndexer
        from .store import SqliteVecStore

        indexer: MemoryIndexer = self._indexer  # type: ignore[assignment]
        store: SqliteVecStore = self._store  # type: ignore[assignment]

        extensions = self._file_extensions or [".md"]

        # ── 1. Scan disk (stat only — no file reads) ──
        disk: dict[str, tuple[Path, int, int]] = {}
        for f in self._memory_root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            if f.name.startswith("vector_index.db"):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            disk[str(f.relative_to(self._memory_root))] = (f, st.st_mtime_ns, st.st_size)

        states = store.get_file_states()

        # ── 2. Detect changed / new files ──
        changed = [(rel, meta) for rel, meta in disk.items() if states.get(rel) != (meta[1], meta[2])]
        skipped = 0
        if max_changed_files is not None and len(changed) > max_changed_files:
            changed.sort(key=lambda item: item[1][1], reverse=True)  # newest mtime first
            skipped = len(changed) - max_changed_files
            changed = changed[:max_changed_files]
            logger.warning(
                "sync: %d files changed, only re-indexing the %d most recent "
                "(run incremental_index() for a full pass)",
                skipped + max_changed_files,
                max_changed_files,
            )

        updated_chunks = 0
        for rel, (path, mtime_ns, size) in changed:
            try:
                updated_chunks += indexer.index_file(path)
                store.set_file_state(rel, mtime_ns, size)
            except Exception:
                logger.warning("sync: failed to index %s", rel, exc_info=True)

        # ── 3. Drop files that no longer exist on disk ──
        # (also purges legacy absolute-path entries, which never match a
        # relative disk key)
        known = set(states) | set(store.list_indexed_files())
        removed = 0
        for rel in known - set(disk):
            store.delete_by_file(rel)
            removed += 1

        elapsed = round(time.monotonic() - t0, 3)
        if updated_chunks or removed:
            logger.info(
                "sync: %d files re-indexed (%d chunks), %d removed, %.2fs",
                len(changed),
                updated_chunks,
                removed,
                elapsed,
            )
        return {
            "indexed_files": len(changed),
            "updated_chunks": updated_chunks,
            "removed_files": removed,
            "skipped_files": skipped,
            "elapsed_seconds": elapsed,
        }

    def _refresh_file_states(self) -> None:
        """Record current ``(mtime_ns, size)`` fingerprints for all files.

        Called after a full/incremental index pass so that a subsequent
        :meth:`sync` sees an up-to-date baseline and does no extra work.
        """
        from .store import SqliteVecStore

        store: SqliteVecStore = self._store  # type: ignore[assignment]
        extensions = self._file_extensions or [".md"]
        states: dict[str, tuple[int, int]] = {}
        for f in self._memory_root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            if f.name.startswith("vector_index.db"):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            states[str(f.relative_to(self._memory_root))] = (st.st_mtime_ns, st.st_size)
        store.set_file_states(states)

    def index_file(self, file_path: str | Path) -> int:
        """Index a single file. Returns the number of new/updated chunks."""
        if not self._ensure_initialized():
            return 0

        from .indexer import MemoryIndexer
        from .store import SqliteVecStore

        indexer: MemoryIndexer = self._indexer  # type: ignore[assignment]
        store: SqliteVecStore = self._store  # type: ignore[assignment]
        count = indexer.index_file(file_path)

        # Keep the fingerprint in sync so a later sync() skips this file.
        path = Path(file_path).resolve()
        try:
            if path.is_file() and path.is_relative_to(self._memory_root.resolve()):
                st = path.stat()
                store.set_file_state(str(path.relative_to(self._memory_root.resolve())), st.st_mtime_ns, st.st_size)
        except OSError:
            pass
        return count

    def remove_file(self, file_path: str | Path) -> int:
        """Remove a file from the index. Returns rows deleted."""
        if not self._ensure_initialized():
            return 0

        from .indexer import MemoryIndexer

        indexer: MemoryIndexer = self._indexer  # type: ignore[assignment]
        return indexer.remove_file(file_path)

    # =========================================================================
    # Search operations
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_score: float = 0.0,
        mode: Optional[str] = None,
    ) -> list[dict[str, object]]:
        """Search memories using hybrid retrieval.

        Parameters
        ----------
        query:
            Natural language search query.
        top_k:
            Maximum number of results.
        memory_type:
            Optional filter (e.g. ``"semantic"``, ``"episodic"``).
        min_score:
            Minimum hybrid score threshold.
        mode:
            Search mode: ``"hybrid"`` (default), ``"vector_only"``, or
            ``"fts_only"``.  ``None`` maps to ``"hybrid"``.

        Returns a list of dicts with keys: file_path, chunk_text,
        hybrid_score, semantic_score, fts_score, importance, temporal_decay, etc.
        """
        if not self._ensure_initialized():
            return []

        from .search import HybridSearchService, SearchMode

        search_svc: HybridSearchService = self._search_service  # type: ignore[assignment]

        # Resolve mode
        search_mode = SearchMode.HYBRID
        if mode:
            try:
                search_mode = SearchMode(mode)
            except ValueError:
                logger.warning("Unknown search mode %r, defaulting to hybrid", mode)

        results = search_svc.search(
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            min_score=min_score,
            mode=search_mode,
        )
        return [
            {
                "file_path": r.file_path,
                "chunk_text": r.chunk_text,
                "chunk_index": r.chunk_index,
                "score": round(r.hybrid_score, 4),
                "hybrid_score": round(r.hybrid_score, 4),
                "semantic_score": round(r.semantic_score, 4),
                "fts_score": round(r.fts_score, 4),
                "importance": r.importance,
                "temporal_decay": round(r.temporal_decay, 4),
                "memory_type": r.memory_type,
                "tags": r.tags,
            }
            for r in results
        ]

    # =========================================================================
    # File listing
    # =========================================================================

    def list_files(self) -> list[str]:
        """Return relative paths of all indexable files under the memory root.

        The returned paths are relative to :attr:`memory_root`.
        """
        extensions = self._file_extensions or [".md"]
        result: list[str] = []
        for f in sorted(self._memory_root.rglob("*")):
            if not f.is_file():
                continue
            if f.name in ("vector_index.db", "vector_index.db-journal", "vector_index.db-wal"):
                continue
            if f.suffix.lower() in extensions:
                result.append(str(f.relative_to(self._memory_root)))
        return result

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> dict[str, object]:
        """Return statistics about the vector index."""
        if not self._ensure_initialized():
            return {"available": False, "error": "Vector dependencies not installed"}

        from .store import SqliteVecStore

        store: SqliteVecStore = self._store  # type: ignore[assignment]

        total_chunks = store.count()
        fts_chunks = store.fts_count()

        # Count indexed files
        rows = store.connection.execute("SELECT COUNT(DISTINCT file_path) FROM memory_vec_meta").fetchone()
        total_files = rows[0] if rows else 0

        # DB file size
        db_size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0

        # Count target files in memory directory
        extensions = self._file_extensions or [".md"]
        target_files = [f for f in self._memory_root.rglob("*") if f.is_file() and f.suffix.lower() in extensions]

        return {
            "available": True,
            "db_path": str(self._db_path),
            "db_size_kb": round(db_size_bytes / 1024, 1),
            "total_chunks": total_chunks,
            "fts_chunks": fts_chunks,
            "indexed_files": total_files,
            "total_md_files": len(target_files),
            "file_extensions": extensions,
            "model": self._model_name,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the vector store connection."""
        if self._store is not None:
            from .store import SqliteVecStore

            store: SqliteVecStore = self._store  # type: ignore[assignment]
            store.close()
            self._store = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
