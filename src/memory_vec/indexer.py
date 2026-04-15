"""
Markdown-to-vector indexing pipeline.

Reads Markdown files, splits them into overlapping chunks, computes
SHA-256 hashes for deduplication, embeds the text, and stores the
resulting vectors in a :class:`SqliteVecStore`.

Key design principles (following OpenClaw memsearch):
- Markdown files remain the **source of truth**; the vector index is a
  derived acceleration structure.
- SHA-256 content hashing ensures we never re-embed unchanged chunks.
- YAML frontmatter is parsed for metadata (importance, type, tags).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .interfaces import IEmbedder
from .store import SqliteVecStore, content_hash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants (reference: OpenClaw ~400 tokens, 80 overlap)
# ---------------------------------------------------------------------------
_APPROX_CHARS_PER_TOKEN = 4  # rough heuristic for English text
_DEFAULT_CHUNK_TOKENS = 400
_DEFAULT_OVERLAP_TOKENS = 80
_CHUNK_SIZE = _DEFAULT_CHUNK_TOKENS * _APPROX_CHARS_PER_TOKEN  # ~1600 chars
_OVERLAP_SIZE = _DEFAULT_OVERLAP_TOKENS * _APPROX_CHARS_PER_TOKEN  # ~320 chars

# Regex for YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------
def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter (if any) and return ``(metadata, body)``.

    If the file has no frontmatter, returns empty metadata and the
    original text.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    body = text[match.end() :]
    try:
        meta = yaml.safe_load(raw_yaml)
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    return meta, body


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap_size: int = _OVERLAP_SIZE,
) -> list[str]:
    """Split *text* into overlapping chunks.

    Strategy:
    1. Split on ``\\n\\n`` (paragraph boundaries) first.
    2. Accumulate paragraphs until the chunk exceeds *chunk_size*.
    3. Adjacent chunks share *overlap_size* characters of trailing context.
    4. Files shorter than *chunk_size* are returned as a single chunk.
    """
    if not text.strip():
        return []

    # Short text — no need to split
    if len(text) <= chunk_size:
        return [text.strip()]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        if current_len + para_len > chunk_size and current:
            # Flush current chunk
            chunk_text_val = "\n\n".join(current).strip()
            if chunk_text_val:
                chunks.append(chunk_text_val)

            # Build overlap from the tail of current
            overlap_buf: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) > overlap_size:
                    break
                overlap_buf.insert(0, p)
                overlap_len += len(p)
            current = overlap_buf
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    # Final chunk
    if current:
        chunk_text_val = "\n\n".join(current).strip()
        if chunk_text_val:
            chunks.append(chunk_text_val)

    return chunks


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
class MemoryIndexer:
    """Build and maintain the vector index for Markdown memory files.

    Parameters
    ----------
    store:
        The :class:`SqliteVecStore` to write embeddings to.
    embedder:
        An :class:`IEmbedder` implementation used for text → vector.
    memory_root:
        Root directory of the memory files.  When provided, all stored
        ``file_path`` values are converted to paths **relative** to this
        root so that the index is portable and free of duplicates caused
        by mixing absolute / relative paths.
    chunk_size:
        Target chunk size in characters (default ~1600 ≈ 400 tokens).
    overlap_size:
        Overlap between adjacent chunks in characters (default ~320 ≈ 80 tokens).
    """

    def __init__(
        self,
        store: SqliteVecStore,
        embedder: IEmbedder,
        memory_root: Optional[str | Path] = None,
        chunk_size: int = _CHUNK_SIZE,
        overlap_size: int = _OVERLAP_SIZE,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._memory_root: Optional[Path] = Path(memory_root).resolve() if memory_root else None
        self._chunk_size = chunk_size
        self._overlap_size = overlap_size

    # -- public API ----------------------------------------------------------

    def index_file(self, file_path: str | Path) -> int:
        """Index a single Markdown file.

        Returns the number of *new or updated* chunks that were embedded.
        Chunks whose SHA-256 hash has not changed are skipped.
        """
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            logger.warning("index_file: %s does not exist or is not a file", path)
            return 0

        raw_text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw_text)

        importance = float(meta.get("importance", 0.5))
        memory_type = str(meta.get("type", "semantic"))
        tags: list[str] = meta.get("tags", []) or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        chunks = _chunk_text(body, self._chunk_size, self._overlap_size)
        if not chunks:
            return 0

        # Use relative path (relative to memory_root) as the canonical key
        # to avoid duplicates from absolute vs relative path differences.
        if self._memory_root and path.is_relative_to(self._memory_root):
            file_key = str(path.relative_to(self._memory_root))
        else:
            file_key = str(path)
        existing_hashes = self._store.get_hashes_for_file(file_key)

        new_or_updated = 0
        # Track which chunk indexes we process this round
        current_indexes: set[int] = set()

        for idx, chunk in enumerate(chunks):
            current_indexes.add(idx)
            chunk_hash = content_hash(chunk)

            old_hash = existing_hashes.get(idx)
            if old_hash == chunk_hash:
                # Unchanged — skip re-embedding
                continue

            if old_hash is not None:
                # Hash changed — delete old then re-insert
                self._delete_chunk(file_key, idx)

            embedding = self._embedder.embed(chunk)
            self._store.insert_embedding(
                embedding=embedding,
                file_path=file_key,
                chunk_index=idx,
                chunk_text=chunk,
                hash_value=chunk_hash,
                importance=importance,
                memory_type=memory_type,
                tags=tags,
            )
            new_or_updated += 1

        # Remove stale chunks (old chunks beyond new chunk count)
        for old_idx in set(existing_hashes.keys()) - current_indexes:
            self._delete_chunk(file_key, old_idx)

        if new_or_updated:
            logger.info("Indexed %s: %d chunks embedded (%d total)", path.name, new_or_updated, len(chunks))
        return new_or_updated

    def index_directory(
        self,
        dir_path: str | Path,
        file_extensions: Optional[list[str]] = None,
    ) -> int:
        """Recursively index text files under *dir_path*.

        Parameters
        ----------
        dir_path:
            Directory to scan.
        file_extensions:
            List of file extensions (with dot) to index, e.g.
            ``[".md", ".txt", ".py"]``.  When ``None`` (default), only
            ``.md`` files are indexed for backward compatibility.

        Returns the total number of new/updated chunks.
        """
        root = Path(dir_path)
        if not root.is_dir():
            logger.warning("index_directory: %s is not a directory", root)
            return 0

        if file_extensions is None:
            file_extensions = [".md"]

        total = 0
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix.lower() in file_extensions:
                total += self.index_file(f)
        return total

    def index_directory_textfiles(self, dir_path: str | Path) -> int:
        """Recursively index all readable text files under *dir_path*.

        Unlike :meth:`index_directory`, this attempts to index **any**
        file that can be read as UTF-8 text.  Binary files are silently
        skipped.

        Returns the total number of new/updated chunks.
        """
        root = Path(dir_path)
        if not root.is_dir():
            logger.warning("index_directory_textfiles: %s is not a directory", root)
            return 0

        total = 0
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            # Skip known binary/index files
            if f.name in ("vector_index.db", "vector_index.db-journal", "vector_index.db-wal"):
                continue
            # Try to read as text
            try:
                f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            total += self.index_file(f)
        return total

    def reindex_all(
        self,
        memory_root: str | Path,
        file_extensions: Optional[list[str]] = None,
    ) -> int:
        """Drop all existing index data and rebuild from scratch.

        Parameters
        ----------
        memory_root:
            Root directory containing memory files.
        file_extensions:
            Passed through to :meth:`index_directory`.  Defaults to
            ``[".md"]`` for backward compatibility.

        Returns the total number of chunks indexed.
        """
        root = Path(memory_root)
        # Clear everything via the store's public clear() method
        self._store.clear()

        return self.index_directory(root, file_extensions=file_extensions)

    def remove_file(self, file_path: str | Path) -> int:
        """Remove all indexed chunks for *file_path*.

        Returns the number of rows deleted.
        """
        path = Path(file_path).resolve()
        if self._memory_root and path.is_relative_to(self._memory_root):
            file_key = str(path.relative_to(self._memory_root))
        else:
            file_key = str(path)
        return self._store.delete_by_file(file_key)

    # -- internal helpers ----------------------------------------------------

    def _delete_chunk(self, file_path: str, chunk_index: int) -> None:
        """Delete a specific chunk (by file_path + chunk_index) from both tables."""
        conn = self._store.connection
        rows = conn.execute(
            "SELECT id FROM memory_vec_meta WHERE file_path = ? AND chunk_index = ?",
            (file_path, chunk_index),
        ).fetchall()
        for (rowid,) in rows:
            self._store.delete_embedding(rowid)


# ---------------------------------------------------------------------------
# Module-level helpers exposed for testing / scripting
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap_size: int = _OVERLAP_SIZE,
) -> list[str]:
    """Public wrapper around the internal chunking function."""
    return _chunk_text(text, chunk_size, overlap_size)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Public wrapper around YAML frontmatter parsing."""
    return _parse_frontmatter(text)
