# markdown-memory-vec

Lightweight vector search for Markdown-based memory systems. Chunk, embed, index, and hybrid-search your `.md` knowledge base with sqlite-vec + FTS5.

## Features

- **Hybrid search**: Three modes — `vector_only` (semantic KNN), `fts_only` (BM25 keywords), `hybrid` (RRF fusion, default)
- **CJK support**: rjieba-powered Chinese word segmentation for FTS5, English untouched
- **Dual embedding backend**: ONNX Runtime (~55 MB) or sentence-transformers (~1.5 GB), auto-detected
- **Markdown-native**: YAML frontmatter parsing for metadata (importance, type, tags)
- **Smart chunking**: Paragraph-aware splitting with configurable overlap (~400 tokens, 80 overlap)
- **SHA-256 dedup**: Never re-embed unchanged content — incremental indexing is fast
- **Single-file storage**: sqlite-vec KNN + FTS5 full-text + metadata in one `.db` file

## Quick Start

### Installation

```bash
# Core (YAML parsing, chunking, interfaces, CJK segmentation)
pip install markdown-memory-vec

# With ONNX embedding (lightweight, recommended)
pip install 'markdown-memory-vec[onnx]'

# With sentence-transformers embedding (full-featured)
pip install 'markdown-memory-vec[vector]'
```

### Python API

```python
from memory_vec import MemoryVectorService

svc = MemoryVectorService("/path/to/memory")
svc.rebuild_index()

# Default: hybrid mode (vector + FTS5 fused via RRF)
results = svc.search("how to deploy")

# Keyword-only (fast, no embedding needed for query)
results = svc.search("部署 kubernetes", mode="fts_only")

# Semantic-only (original behavior)
results = svc.search("deployment strategies", mode="vector_only")

svc.close()
```

### Low-level API

```python
from memory_vec import (
    SqliteVecStore,
    create_embedder,
    MemoryIndexer,
    HybridSearchService,
    SearchMode,
)

store = SqliteVecStore("memory.db")
store.ensure_tables()

embedder = create_embedder()  # auto-detects ONNX or sentence-transformers
indexer = MemoryIndexer(store, embedder, memory_root="/path/to/memory")
indexer.index_directory("/path/to/memory")

search = HybridSearchService(vec_store=store, embedder=embedder)

# Hybrid (default) — fuses vector KNN + FTS5 BM25 via Reciprocal Rank Fusion
results = search.search("how to deploy", mode=SearchMode.HYBRID)

for r in results:
    print(f"{r.file_path} (score={r.hybrid_score:.3f}): {r.chunk_text[:80]}...")
```

## CLI

```bash
# Full rebuild
memory-vec /path/to/memory --rebuild

# Incremental update (only changed files)
memory-vec /path/to/memory --incremental

# Search (default: hybrid mode)
memory-vec /path/to/memory --search "how to deploy" --top-k 5

# Search with specific mode
memory-vec /path/to/memory --search "桌面端构建" --mode fts_only

# Statistics
memory-vec /path/to/memory --stats

# Verbose logging
memory-vec /path/to/memory --rebuild -v
```

## Search Modes

| Mode | Relevance Signal | Best For |
|------|-----------------|----------|
| `hybrid` (default) | RRF(vector, FTS5) | General use — combines semantic understanding with keyword precision |
| `vector_only` | Cosine similarity | Conceptual/semantic queries where exact words don't matter |
| `fts_only` | BM25 | Exact keyword/term lookup, fastest (~2ms) |

All modes apply the same final scoring:

```
score = α × relevance + β × importance + γ × temporal_decay
```

Where `relevance` is cosine similarity, normalized BM25, or normalized RRF score depending on mode.

## API Reference

### High-level

| Class | Description |
|-------|-------------|
| `MemoryVectorService` | All-in-one: rebuild, incremental index, search, stats |

### Components

| Class | Description |
|-------|-------------|
| `SqliteVecStore` | sqlite-vec KNN + FTS5 full-text + metadata storage |
| `OnnxEmbedder` | Lightweight ONNX Runtime embedder (~55 MB) |
| `SentenceTransformerEmbedder` | Full sentence-transformers embedder (~1.5 GB) |
| `create_embedder()` | Factory — picks best available backend (ONNX preferred) |
| `MemoryIndexer` | Markdown → chunks → embeddings → store pipeline |
| `HybridSearchService` | Multi-mode search with RRF fusion and hybrid scoring |

### Interfaces

| Interface | Description |
|-----------|-------------|
| `IEmbedder` | Abstract embedder (`embed`, `embed_batch`, `dimension`) |
| `ISqliteVecStore` | Abstract vector store (`add`, `search`, `delete`, `clear`, `count`) |

### Data Types

| Type | Description |
|------|-------------|
| `SearchMode` | Enum: `HYBRID`, `VECTOR_ONLY`, `FTS_ONLY` |
| `SearchResult` | Result with `hybrid_score`, `semantic_score`, `fts_score`, `rrf_score` |
| `VectorRecord` | Record for insertion (id, embedding, metadata) |
| `VectorSearchResult` | Raw KNN result (id, distance, metadata) |

### Utilities

| Function | Description |
|----------|-------------|
| `reciprocal_rank_fusion(ranked_lists, k)` | RRF score fusion from multiple ranked lists |
| `chunk_text(text, chunk_size, overlap_size)` | Split text into overlapping chunks |
| `parse_frontmatter(text)` | Extract YAML frontmatter from Markdown |
| `content_hash(text)` | SHA-256 hex digest |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  MemoryVectorService                     │
│            (high-level orchestration layer)               │
├──────────┬───────────┬──────────────┬────────────────────┤
│          │           │              │                     │
│ Indexer  │  Search   │  Embedder    │      Store          │
│          │           │              │                     │
│ .md file │ 3 modes:  │ ONNX or      │ sqlite-vec KNN     │
│ → chunk  │  hybrid   │ sentence-    │ + FTS5 full-text   │
│ → embed  │  vector   │ transformers │ + metadata table   │
│ → store  │  fts      │ (auto-detect)│ (single .db file)  │
│          │ + RRF     │              │                     │
└──────────┴───────────┴──────────────┴────────────────────┘
     ▲          │                              │
     │   rjieba CJK segmentation              │
     │   (Chinese word splitting)              ▼
   ┌─┴────────────────────────┐    ┌──────────────────────┐
   │     Markdown Files       │    │   vector_index.db    │
   │   (caller-specified)     │    │   (vec0 + FTS5 +     │
   └──────────────────────────┘    │    meta + version)   │
                                   └──────────────────────┘
```

## Configuration

### Embedding Model

Default: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages).

```bash
# For regions with slow HuggingFace access
export HF_ENDPOINT=https://hf-mirror.com

# Offline mode (model must be pre-cached)
export HF_HUB_OFFLINE=1
```

### Search Weights

Default: `α=0.6, β=0.2, γ=0.2, λ=0.05`

```python
search = HybridSearchService(
    vec_store=store,
    embedder=embedder,
    alpha=0.8,         # Relevance weight
    beta=0.1,          # Importance weight
    gamma=0.1,         # Temporal decay weight
    decay_lambda=0.03, # Slower decay
    rrf_k=60,          # RRF constant (higher = less top-rank bias)
)
```

## License

MIT
