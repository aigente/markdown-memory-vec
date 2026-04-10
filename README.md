# markdown-memory-vec

Lightweight vector search for Markdown-based memory systems. Chunk, embed, index, and hybrid-search your `.md` knowledge base with sqlite-vec.

<!-- Badges -->
<!-- [![PyPI version](https://badge.fury.io/py/markdown-memory-vec.svg)](https://pypi.org/project/markdown-memory-vec/) -->
<!-- [![Python](https://img.shields.io/pypi/pyversions/markdown-memory-vec.svg)](https://pypi.org/project/markdown-memory-vec/) -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

## Features

- **Markdown-native**: YAML frontmatter parsing for metadata (importance, type, tags)
- **Smart chunking**: Paragraph-aware splitting with configurable overlap (~400 tokens, 80 overlap)
- **SHA-256 dedup**: Never re-embed unchanged content — incremental indexing is fast
- **Hybrid search**: Combines semantic similarity (α), importance weighting (β), and temporal decay (γ)
- **Zero-copy storage**: sqlite-vec KNN search with cosine distance in a single `.db` file

## Quick Start

### Installation

```bash
# Core only (YAML parsing, chunking, interfaces)
pip install markdown-memory-vec

# With vector search support (sqlite-vec + sentence-transformers)
pip install 'markdown-memory-vec[vector]'
```

### Python API

```python
from memory_vec import MemoryVectorService

svc = MemoryVectorService("/path/to/project/.claude/memory")
svc.rebuild_index()                        # Full index build
results = svc.search("how to deploy")      # Hybrid search
svc.close()
```

### Low-level API

```python
from memory_vec import (
    SqliteVecStore,
    SentenceTransformerEmbedder,
    MemoryIndexer,
    HybridSearchService,
)

store = SqliteVecStore("memory.db")
store.ensure_tables()

embedder = SentenceTransformerEmbedder()
indexer = MemoryIndexer(store, embedder, memory_root="/path/to/memory")
indexer.index_directory("/path/to/memory")

search = HybridSearchService(vec_store=store, embedder=embedder)
results = search.search("how to deploy")
for r in results:
    print(f"{r.file_path} (score={r.hybrid_score:.3f}): {r.chunk_text[:80]}...")
```

## CLI Usage

```bash
# Full rebuild (pass the memory directory directly)
memory-vec /path/to/project/.claude/memory --rebuild

# Incremental update (only changed files)
memory-vec /path/to/project/.claude/memory --incremental

# Search
memory-vec /path/to/project/.claude/memory --search "how to deploy" --top-k 5

# Statistics
memory-vec /path/to/project/.claude/memory --stats

# Or use --memory-subdir to compose the path from workspace root
memory-vec /path/to/project --memory-subdir .claude/memory --rebuild

# Verbose logging
memory-vec /path/to/project/.claude/memory --rebuild -v
```

## API Reference

### High-level

| Class | Description |
|-------|-------------|
| `MemoryVectorService` | All-in-one service: rebuild, incremental index, search, stats |

### Components

| Class | Description |
|-------|-------------|
| `SqliteVecStore` | sqlite-vec backed vector store with KNN search |
| `SentenceTransformerEmbedder` | Lazy-loading sentence-transformers embedder |
| `MemoryIndexer` | Markdown → chunks → embeddings → store pipeline |
| `HybridSearchService` | Hybrid scoring: `α×semantic + β×importance + γ×temporal` |

### Interfaces

| Interface | Description |
|-----------|-------------|
| `IEmbedder` | Abstract embedder (`embed`, `embed_batch`, `dimension`) |
| `ISqliteVecStore` | Abstract vector store (`add`, `search`, `delete`, `clear`, `count`) |

### Data Types

| Type | Description |
|------|-------------|
| `VectorRecord` | Record for insertion (id, embedding, metadata) |
| `VectorSearchResult` | Raw KNN result (id, distance, metadata) |
| `SearchResult` | Hybrid search result with all score components |
| `MemoryVecMeta` | Metadata dataclass for stored embeddings |

### Utilities

| Function | Description |
|----------|-------------|
| `chunk_text(text, chunk_size, overlap_size)` | Split text into overlapping chunks |
| `parse_frontmatter(text)` | Extract YAML frontmatter from Markdown |
| `content_hash(text)` | SHA-256 hex digest |
| `is_sqlite_vec_available()` | Check sqlite-vec availability |
| `is_sentence_transformers_available()` | Check sentence-transformers availability |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                MemoryVectorService                  │
│          (high-level orchestration layer)            │
├──────────┬──────────┬──────────────┬────────────────┤
│          │          │              │                 │
│  Indexer  │  Search  │   Embedder   │     Store      │
│          │          │              │                 │
│ .md file │  hybrid  │  sentence-   │  sqlite-vec    │
│ → chunks │  scoring │  transformers│  KNN + meta    │
│ → embed  │  α+β+γ   │  (lazy load) │  (cosine)      │
│ → store  │          │              │                 │
└──────────┴──────────┴──────────────┴────────────────┘
     ▲                                      │
     │         YAML frontmatter             │
     │         importance/type/tags          ▼
   ┌─┴─────────────────────┐     ┌─────────────────────┐
   │    Markdown Files     │     │   vector_index.db    │
   │  (caller-specified)   │     │   (single file)      │
   └───────────────────────┘     └─────────────────────┘
```

## Configuration

### HuggingFace Model

By default, uses `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages).

For users in China or regions with slow HuggingFace access:

```bash
# Use a mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or use offline mode (model must be pre-cached)
export HF_HUB_OFFLINE=1
```

### Hybrid Search Weights

Default: `α=0.6, β=0.2, γ=0.2, λ=0.05`

```python
search = HybridSearchService(
    vec_store=store,
    embedder=embedder,
    alpha=0.8,    # Semantic weight
    beta=0.1,     # Importance weight
    gamma=0.1,    # Temporal decay weight
    decay_lambda=0.03,  # Slower decay
)
```

## License

MIT
