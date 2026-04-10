"""
CLI entry point: ``python -m memory_vec`` or ``memory-vec``.

Provides commands for building, maintaining, and querying the vector index.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .service import MemoryVectorService


def main() -> None:
    """Command-line interface for memory vector maintenance."""
    parser = argparse.ArgumentParser(
        prog="memory-vec",
        description="Markdown memory vector index — build, maintain, and search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  memory-vec /path/to/memory --rebuild\n"
            "  memory-vec /path/to/memory --incremental\n"
            "  memory-vec /path/to/memory --search 'how to deploy'\n"
            "  memory-vec /path/to/memory --stats\n"
            "\n"
            "  # Convenience: pass a workspace root and a --memory-subdir\n"
            "  memory-vec /path/to/project --memory-subdir .claude/memory --rebuild\n"
            "\n"
            "Environment variables:\n"
            "  HF_ENDPOINT      HuggingFace mirror URL (e.g. https://hf-mirror.com)\n"
            "  HF_HUB_OFFLINE   Set to '1' to disable network access\n"
        ),
    )
    parser.add_argument("workspace", help="Memory directory (or project root when combined with --memory-subdir)")
    parser.add_argument("--rebuild", action="store_true", help="Full rebuild of vector index")
    parser.add_argument("--incremental", action="store_true", help="Incremental re-index (only changed files)")
    parser.add_argument("--search", type=str, help="Search memories by query")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results (default: 5)")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--memory-subdir",
        type=str,
        default="",
        help="Optional subdirectory relative to workspace (e.g. '.claude/memory'). "
        "When set, the indexed directory is workspace/memory-subdir.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    memory_dir = os.path.join(args.workspace, args.memory_subdir) if args.memory_subdir else args.workspace
    svc = MemoryVectorService(
        memory_dir,
        model_name=args.model,
    )

    if not svc.is_available:
        print("ERROR: Vector dependencies not installed.", file=sys.stderr)  # noqa: T201
        print("Install with: pip install 'markdown-memory-vec[vector]'", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    try:
        if args.rebuild:
            total = svc.rebuild_index()
            print(f"Full rebuild complete: {total} chunks indexed")  # noqa: T201

        elif args.incremental:
            total = svc.incremental_index()
            print(f"Incremental index: {total} chunks updated")  # noqa: T201

        elif args.search:
            results = svc.search(args.search, top_k=args.top_k)
            if not results:
                print("No results found.")  # noqa: T201
            else:
                for i, r in enumerate(results, 1):
                    print(f"\n--- Result {i} (score: {r['hybrid_score']}) ---")  # noqa: T201
                    print(f"File: {r['file_path']}")  # noqa: T201
                    print(f"Type: {r['memory_type']}  Importance: {r['importance']}")  # noqa: T201
                    text = str(r["chunk_text"])
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"Text: {text}")  # noqa: T201

        elif args.stats:
            s = svc.stats()
            print("Vector Index Statistics:")  # noqa: T201
            for k, v in s.items():
                print(f"  {k}: {v}")  # noqa: T201

        else:
            parser.print_help()

    finally:
        svc.close()


if __name__ == "__main__":
    main()
