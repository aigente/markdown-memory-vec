"""
sqlite-vec based vector store for memory embeddings.

Uses sqlite-vec extension to create vec0 virtual tables for KNN search.
All sqlite-vec imports are lazy and guarded with try/except for graceful
degradation when the extension is not installed.

Note: vec0 virtual tables require synchronous SQLite connections (not aiosqlite).
This module deliberately uses the stdlib ``sqlite3`` driver.

The public class :class:`SqliteVecStore` implements the :class:`ISqliteVecStore`
interface defined in ``interfaces.py`` so that it can be plugged directly into
:class:`HybridSearchService`.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .interfaces import ISqliteVecStore, VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability flag
# ---------------------------------------------------------------------------
_sqlite_vec_available = False
try:
    import sqlite_vec  # type: ignore[import-untyped]

    _sqlite_vec_available = True
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]


def is_sqlite_vec_available() -> bool:
    """Return ``True`` if the sqlite-vec extension is importable."""
    return _sqlite_vec_available


# ---------------------------------------------------------------------------
# Metadata dataclass (pure Python — no SQLAlchemy dependency)
# ---------------------------------------------------------------------------
@dataclass
class MemoryVecMeta:
    """Metadata row that mirrors each embedding stored in the vec0 table.

    This is a pure dataclass (no ORM).  All database operations are performed
    via raw SQL in :class:`SqliteVecStore`.
    """

    id: int = 0
    file_path: str = ""
    chunk_index: int = 0
    content_hash: str = ""
    chunk_text: str = ""
    importance: float = 0.5
    memory_type: Optional[str] = None
    last_accessed: Optional[str] = None
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: content hash
# ---------------------------------------------------------------------------
def content_hash(text: str) -> str:
    """Return SHA-256 hex digest for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------
_VEC_DIMENSION = 384  # paraphrase-multilingual-MiniLM-L12-v2 default


class SqliteVecStore(ISqliteVecStore):
    """Concrete :class:`ISqliteVecStore` implementation using sqlite-vec.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for tests.
    dimension:
        Vector dimension.  Defaults to 384 (paraphrase-multilingual-MiniLM-L12-v2).
    table_name:
        Name of the vec0 virtual table.  Defaults to ``"memory_vec"``.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        dimension: int = _VEC_DIMENSION,
        table_name: str = "memory_vec",
    ) -> None:
        if not _sqlite_vec_available:
            raise RuntimeError(
                "sqlite-vec is not installed. " "Install it with: pip install 'markdown-memory-vec[vector]'"
            )
        self._db_path = str(db_path)
        self._dimension = dimension
        self._table_name = table_name
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection management -----------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """Return (and lazily create) the underlying SQLite connection."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)  # type: ignore[union-attr]
        conn.enable_load_extension(False)
        return conn

    # -- schema --------------------------------------------------------------

    def ensure_tables(self) -> None:
        """Create the vec0 virtual table and the metadata table if they don't exist."""
        conn = self.connection

        # vec0 virtual table (raw SQL required)
        # Use cosine distance so similarity = 1 - distance (range [0, 2])
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table_name} USING vec0(
                embedding float[{self._dimension}] distance_metric=cosine,
                +file_path TEXT,
                +chunk_index INTEGER,
                +content_hash TEXT,
                +created_at TEXT,
                +updated_at TEXT
            )
            """
        )

        # Metadata table (plain SQL)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_vec_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                content_hash VARCHAR(64) NOT NULL,
                chunk_text TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                memory_type VARCHAR(20),
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                tags JSON DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_memory_vec_meta_file_path ON memory_vec_meta (file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_memory_vec_meta_content_hash ON memory_vec_meta (content_hash)")
        conn.commit()

    # =====================================================================
    # ISqliteVecStore interface implementation
    # =====================================================================

    def add(self, records: Sequence[VectorRecord]) -> None:
        """Add records to the vector store (ISqliteVecStore interface)."""
        for record in records:
            meta = record.metadata
            self.insert_embedding(
                embedding=record.embedding,
                file_path=meta.get("file_path", ""),
                chunk_index=meta.get("chunk_index", 0),
                chunk_text=meta.get("chunk_text", ""),
                hash_value=meta.get("content_hash", ""),
                importance=float(meta.get("importance", 0.5)),
                memory_type=str(meta.get("memory_type", "semantic")),
                tags=meta.get("tags"),
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Perform KNN search (ISqliteVecStore interface).

        Note: sqlite-vec does not natively support metadata filters in the
        KNN query.  We fetch extra candidates and filter in Python.
        """
        # Over-fetch to allow post-filtering
        fetch_k = top_k * 3 if filter_metadata else top_k
        raw_results = self.search_similar(query_embedding, k=fetch_k)

        results: List[VectorSearchResult] = []
        for raw in raw_results:
            meta = self.get_meta(raw.rowid)
            if meta is None:
                continue

            # Apply metadata filter
            if filter_metadata:
                skip = False
                for fk, fv in filter_metadata.items():
                    if meta.get(fk) != fv:
                        skip = True
                        break
                if skip:
                    continue

            results.append(
                VectorSearchResult(
                    id=str(raw.rowid),
                    distance=raw.distance,
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records by IDs (ISqliteVecStore interface)."""
        for id_str in ids:
            self.delete_embedding(int(id_str))

    def clear(self) -> None:
        """Delete all records from the store (ISqliteVecStore interface)."""
        conn = self.connection
        conn.execute(f"DELETE FROM {self._table_name}")
        conn.execute("DELETE FROM memory_vec_meta")
        conn.commit()

    # =====================================================================
    # Extended CRUD (richer API used by MemoryIndexer)
    # =====================================================================

    def insert_embedding(
        self,
        embedding: Sequence[float],
        file_path: str,
        chunk_index: int,
        chunk_text: str,
        hash_value: str = "",
        importance: float = 0.5,
        memory_type: str = "semantic",
        tags: Optional[list[str]] = None,
    ) -> int:
        """Insert a new embedding and its metadata.  Returns the rowid."""
        now = datetime.now(timezone.utc).isoformat()
        if not hash_value:
            hash_value = content_hash(chunk_text)

        conn = self.connection
        vec_blob = struct.pack(f"{len(embedding)}f", *embedding)
        cursor = conn.execute(
            f"INSERT INTO {self._table_name}(embedding, file_path, chunk_index, content_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (vec_blob, file_path, chunk_index, hash_value, now, now),
        )
        rowid = cursor.lastrowid or 0

        conn.execute(
            "INSERT INTO memory_vec_meta "
            "(id, file_path, chunk_index, content_hash, chunk_text, importance, memory_type, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rowid,
                file_path,
                chunk_index,
                hash_value,
                chunk_text,
                importance,
                memory_type,
                _json.dumps(tags or []),
                now,
                now,
            ),
        )
        conn.commit()
        return rowid

    def search_similar(
        self,
        query_embedding: Sequence[float],
        k: int = 10,
    ) -> list[_VecRawResult]:
        """Return the *k* nearest neighbours from the vec0 table."""
        vec_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        rows = self.connection.execute(
            f"SELECT rowid, distance, file_path, chunk_index, content_hash, created_at, updated_at "
            f"FROM {self._table_name} "
            f"WHERE embedding MATCH ? AND k = ?",
            (vec_blob, k),
        ).fetchall()
        return [
            _VecRawResult(
                rowid=r[0],
                distance=r[1],
                file_path=r[2],
                chunk_index=r[3],
                content_hash=r[4],
                created_at=r[5],
                updated_at=r[6],
            )
            for r in rows
        ]

    def delete_embedding(self, rowid: int) -> None:
        """Delete an embedding (and its metadata) by rowid."""
        conn = self.connection
        conn.execute(f"DELETE FROM {self._table_name} WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM memory_vec_meta WHERE id = ?", (rowid,))
        conn.commit()

    def delete_by_file(self, file_path: str) -> int:
        """Delete all embeddings for a given file.  Returns the number of rows deleted."""
        conn = self.connection
        rows = conn.execute("SELECT id FROM memory_vec_meta WHERE file_path = ?", (file_path,)).fetchall()
        if not rows:
            return 0
        rowids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(rowids))
        conn.execute(f"DELETE FROM {self._table_name} WHERE rowid IN ({placeholders})", rowids)
        conn.execute(f"DELETE FROM memory_vec_meta WHERE id IN ({placeholders})", rowids)
        conn.commit()
        return len(rowids)

    def update_embedding(
        self,
        rowid: int,
        embedding: Sequence[float],
        chunk_text: str = "",
        hash_value: str = "",
    ) -> None:
        """Replace the embedding vector for an existing row."""
        now = datetime.now(timezone.utc).isoformat()
        vec_blob = struct.pack(f"{len(embedding)}f", *embedding)
        conn = self.connection
        # vec0 update: delete + re-insert with same rowid
        conn.execute(f"DELETE FROM {self._table_name} WHERE rowid = ?", (rowid,))

        meta_row = conn.execute(
            "SELECT file_path, chunk_index, content_hash, created_at FROM memory_vec_meta WHERE id = ?",
            (rowid,),
        ).fetchone()
        if meta_row is None:
            conn.commit()
            return

        file_path, chunk_index, old_hash, created_at = meta_row
        new_hash = hash_value or (content_hash(chunk_text) if chunk_text else old_hash)

        conn.execute(
            f"INSERT INTO {self._table_name}(rowid, embedding, file_path, chunk_index, content_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rowid, vec_blob, file_path, chunk_index, new_hash, created_at, now),
        )

        update_fields: dict[str, str] = {"updated_at": now, "content_hash": new_hash}
        if chunk_text:
            update_fields["chunk_text"] = chunk_text
        set_clause = ", ".join(f"{k} = ?" for k in update_fields)
        conn.execute(
            f"UPDATE memory_vec_meta SET {set_clause} WHERE id = ?",
            (*update_fields.values(), rowid),
        )
        conn.commit()

    def get_meta(self, rowid: int) -> Optional[Dict[str, Any]]:
        """Return the metadata dict for a given rowid, or ``None``."""
        row = self.connection.execute(
            "SELECT id, file_path, chunk_index, content_hash, chunk_text, "
            "importance, memory_type, last_accessed, access_count, tags, "
            "created_at, updated_at "
            "FROM memory_vec_meta WHERE id = ?",
            (rowid,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "file_path": row[1],
            "chunk_index": row[2],
            "content_hash": row[3],
            "chunk_text": row[4],
            "importance": row[5],
            "memory_type": row[6],
            "last_accessed": row[7],
            "access_count": row[8],
            "tags": _json.loads(row[9]) if isinstance(row[9], str) else (row[9] or []),
            "created_at": row[10],
            "updated_at": row[11],
        }

    def get_hashes_for_file(self, file_path: str) -> Dict[int, str]:
        """Return ``{chunk_index: content_hash}`` for all chunks of *file_path*."""
        rows = self.connection.execute(
            "SELECT chunk_index, content_hash FROM memory_vec_meta WHERE file_path = ?",
            (file_path,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def record_access(self, rowid: int) -> None:
        """Bump access count and update ``last_accessed`` for a result row."""
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "UPDATE memory_vec_meta SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, rowid),
        )
        self.connection.commit()

    def count(self) -> int:
        """Return the total number of stored embeddings."""
        row = self.connection.execute("SELECT COUNT(*) FROM memory_vec_meta").fetchone()
        return row[0] if row else 0

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal raw result (lighter than VectorSearchResult, used internally)
# ---------------------------------------------------------------------------
class _VecRawResult:
    """A raw KNN result row from the vec0 table."""

    __slots__ = ("rowid", "distance", "file_path", "chunk_index", "content_hash", "created_at", "updated_at")

    def __init__(
        self,
        rowid: int,
        distance: float,
        file_path: str = "",
        chunk_index: int = 0,
        content_hash: str = "",
        created_at: str = "",
        updated_at: str = "",
    ) -> None:
        self.rowid = rowid
        self.distance = distance
        self.file_path = file_path
        self.chunk_index = chunk_index
        self.content_hash = content_hash
        self.created_at = created_at
        self.updated_at = updated_at
