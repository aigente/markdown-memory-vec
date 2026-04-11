"""
Vector search infrastructure for Markdown-based memory systems.

This package provides:

- **Interfaces** (``interfaces.py``): ``IEmbedder``, ``ISqliteVecStore``
- **Storage** (``store.py``): ``SqliteVecStore`` — concrete sqlite-vec
  backed vector store implementing ``ISqliteVecStore``
- **Embedder** (``embedder.py``): Dual-backend embedding with auto-detection:
  - ``OnnxEmbedder`` — lightweight (~55 MB), uses onnxruntime + tokenizers
  - ``SentenceTransformerEmbedder`` — full-featured (~1.5 GB), uses torch
  - ``create_embedder()`` — factory that picks the best available backend
- **Indexer** (``indexer.py``): ``MemoryIndexer`` — Markdown-to-vector indexing
  pipeline with chunking, SHA-256 dedup, and YAML frontmatter parsing
- **Search** (``search.py``): ``HybridSearchService`` — hybrid retrieval
  combining semantic similarity, FTS5 keyword search (BM25), importance,
  and temporal decay.  Supports three modes: ``vector_only``, ``fts_only``,
  ``hybrid`` (default, uses Reciprocal Rank Fusion)

All heavy dependencies are optional:
  - ``pip install 'markdown-memory-vec[onnx]'``   → onnxruntime + tokenizers (~55 MB)
  - ``pip install 'markdown-memory-vec[vector]'``  → sentence-transformers + torch (~1.5 GB)

Quick start::

    from memory_vec import create_embedder, SqliteVecStore, MemoryIndexer, HybridSearchService

    store = SqliteVecStore("memory.db")
    store.ensure_tables()

    embedder = create_embedder()  # auto-detects best backend
    indexer = MemoryIndexer(store, embedder)
    indexer.index_directory("path/to/markdown/files")

    search = HybridSearchService(vec_store=store, embedder=embedder)
    results = search.search("how to deploy")
"""

# Interfaces
# Concrete implementations
from .embedder import (
    OnnxEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    download_onnx_model,
    is_any_backend_available,
    is_onnx_available,
    is_sentence_transformers_available,
)
from .indexer import MemoryIndexer, chunk_text, parse_frontmatter
from .interfaces import IEmbedder, ISqliteVecStore, VectorRecord, VectorSearchResult
from .search import HybridSearchService, SearchMode, SearchResult, reciprocal_rank_fusion
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
    "OnnxEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "download_onnx_model",
    "is_onnx_available",
    "is_sentence_transformers_available",
    "is_any_backend_available",
    # Indexer
    "MemoryIndexer",
    "chunk_text",
    "parse_frontmatter",
    # Search
    "HybridSearchService",
    "SearchMode",
    "SearchResult",
    "reciprocal_rank_fusion",
    # High-level service
    "MemoryVectorService",
]

__version__ = "0.1.0"
