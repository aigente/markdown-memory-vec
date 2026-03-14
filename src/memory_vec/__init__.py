"""
Vector search infrastructure for Markdown-based memory systems.

This package provides:

- **Interfaces** (``interfaces.py``): ``IEmbedder``, ``ISqliteVecStore``
- **Storage** (``store.py``): ``SqliteVecStore`` — concrete sqlite-vec
  backed vector store implementing ``ISqliteVecStore``
- **Embedder** (``embedder.py``): ``SentenceTransformerEmbedder`` — concrete
  embedder using sentence-transformers, implementing ``IEmbedder``
- **Indexer** (``indexer.py``): ``MemoryIndexer`` — Markdown-to-vector indexing
  pipeline with chunking, SHA-256 dedup, and YAML frontmatter parsing
- **Search** (``search.py``): ``HybridSearchService`` — hybrid retrieval
  combining semantic similarity, importance, and temporal decay

All heavy dependencies (``sqlite-vec``, ``sentence-transformers``) are optional.
Use the ``is_*_available()`` helpers to check at runtime.

Quick start::

    from memory_vec import (
        SqliteVecStore,
        SentenceTransformerEmbedder,
        MemoryIndexer,
        HybridSearchService,
    )

    store = SqliteVecStore("memory.db")
    store.ensure_tables()

    embedder = SentenceTransformerEmbedder()
    indexer = MemoryIndexer(store, embedder)
    indexer.index_directory("path/to/markdown/files")

    search = HybridSearchService(vec_store=store, embedder=embedder)
    results = search.search("how to deploy")
"""

# Interfaces
# Concrete implementations
from .embedder import SentenceTransformerEmbedder, is_sentence_transformers_available
from .indexer import MemoryIndexer, chunk_text, parse_frontmatter
from .interfaces import IEmbedder, ISqliteVecStore, VectorRecord, VectorSearchResult
from .search import HybridSearchService, SearchResult
from .service import MemoryVectorService
from .store import MemoryVecMeta, SqliteVecStore, content_hash, is_sqlite_vec_available

__all__ = [
    # Interfaces
    "IEmbedder",
    "ISqliteVecStore",
    "VectorRecord",
    "VectorSearchResult",
    # Store
    "SqliteVecStore",
    "MemoryVecMeta",
    "content_hash",
    "is_sqlite_vec_available",
    # Embedder
    "SentenceTransformerEmbedder",
    "is_sentence_transformers_available",
    # Indexer
    "MemoryIndexer",
    "chunk_text",
    "parse_frontmatter",
    # Search
    "HybridSearchService",
    "SearchResult",
    # High-level service
    "MemoryVectorService",
]

__version__ = "0.1.0"
