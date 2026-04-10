#!/usr/bin/env python3
"""
Export a sentence-transformers model to ONNX format for lightweight inference.

This script converts a HuggingFace sentence-transformers model into:
  - model.onnx     — the ONNX model graph
  - tokenizer.json — the fast tokenizer (HuggingFace tokenizers library format)
  - config.json    — model config (copied from source)

Output directory: ~/.cache/aigente/models/onnx/{model_name}/

Requirements (export only — not needed at inference time):
  pip install sentence-transformers optimum[onnxruntime]

Usage:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --model paraphrase-multilingual-MiniLM-L12-v2
  python scripts/export_onnx.py --output /custom/path
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT_BASE = Path.home() / ".cache" / "aigente" / "models" / "onnx"


def find_hf_snapshot(model_name: str) -> Path | None:
    """Find the local HuggingFace snapshot for a sentence-transformers model."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--sentence-transformers--{model_name}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def export_via_optimum(model_name: str, output_dir: Path) -> None:
    """Export using HuggingFace Optimum (recommended, cleanest output)."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "optimum[onnxruntime] not installed. Install with:\n"
            "  pip install optimum[onnxruntime]"
        )

    hf_model_id = f"sentence-transformers/{model_name}"
    logger.info("Exporting '%s' via Optimum → %s", hf_model_id, output_dir)

    model = ORTModelForFeatureExtraction.from_pretrained(
        hf_model_id, export=True
    )
    model.save_pretrained(output_dir)
    logger.info("Optimum export complete")


def export_via_torch(model_name: str, output_dir: Path) -> None:
    """Export using torch.onnx.export (fallback if optimum not installed)."""
    import numpy as np
    import torch

    logger.info("Optimum not available, using torch.onnx.export fallback")

    # Load from local HF snapshot if available
    snapshot = find_hf_snapshot(model_name)
    if snapshot:
        model_path = str(snapshot)
        logger.info("Using local snapshot: %s", model_path)
    else:
        model_path = f"sentence-transformers/{model_name}"
        logger.info("Downloading model: %s", model_path)

    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Dummy input for tracing
    dummy_text = "This is a test sentence."
    inputs = tokenizer(
        dummy_text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=128,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("Exporting ONNX model to %s", onnx_path)
    # Use legacy TorchScript exporter (dynamo=False) for maximum compatibility
    # with execution providers (CoreML, DirectML, etc.)
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        dynamo=False,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info("torch.onnx.export complete")

    # Verify ONNX output matches torch output
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
            "token_type_ids": inputs["token_type_ids"].numpy(),
        }
        ort_out = sess.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_out = model(**inputs).last_hidden_state.numpy()

        max_diff = float(np.max(np.abs(ort_out - torch_out)))
        logger.info("Max difference between ONNX and PyTorch outputs: %.6e", max_diff)
        if max_diff > 1e-4:
            logger.warning("Large difference detected! Embeddings may not match.")
        else:
            logger.info("Verification passed — outputs match within tolerance")
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")


def ensure_tokenizer_json(output_dir: Path, model_name: str) -> None:
    """Ensure tokenizer.json exists in the output directory."""
    tokenizer_json = output_dir / "tokenizer.json"
    if tokenizer_json.exists():
        logger.info("tokenizer.json already exists")
        return

    # Try copying from HF snapshot
    snapshot = find_hf_snapshot(model_name)
    if snapshot and (snapshot / "tokenizer.json").exists():
        shutil.copy2(snapshot / "tokenizer.json", tokenizer_json)
        logger.info("Copied tokenizer.json from HF snapshot")
        return

    # Generate from transformers
    logger.info("Generating tokenizer.json from transformers")
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tok = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
    tok.save_pretrained(output_dir)


def copy_config(output_dir: Path, model_name: str) -> None:
    """Copy config.json from HF snapshot if available."""
    config_out = output_dir / "config.json"
    if config_out.exists():
        return

    snapshot = find_hf_snapshot(model_name)
    if snapshot and (snapshot / "config.json").exists():
        shutil.copy2(snapshot / "config.json", config_out)
        logger.info("Copied config.json from HF snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sentence-transformers model to ONNX")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_BASE}/{{model}})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing export",
    )
    args = parser.parse_args()

    model_name: str = args.model
    output_dir: Path = args.output or (DEFAULT_OUTPUT_BASE / model_name)

    if (output_dir / "model.onnx").exists() and not args.force:
        logger.info("ONNX model already exists at %s (use --force to overwrite)", output_dir)
        # Still ensure tokenizer.json is present
        ensure_tokenizer_json(output_dir, model_name)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try optimum first, fall back to torch.onnx.export
    try:
        export_via_optimum(model_name, output_dir)
    except Exception as e:
        logger.warning("Optimum export failed (%s), trying torch fallback", e)
        export_via_torch(model_name, output_dir)

    ensure_tokenizer_json(output_dir, model_name)
    copy_config(output_dir, model_name)

    # Print summary
    model_onnx = output_dir / "model.onnx"
    size_mb = model_onnx.stat().st_size / (1024 * 1024) if model_onnx.exists() else 0
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info("  Model: %s", model_name)
    logger.info("  Output: %s", output_dir)
    logger.info("  model.onnx: %.1f MB", size_mb)
    logger.info("  tokenizer.json: %s", "✓" if (output_dir / "tokenizer.json").exists() else "✗")
    logger.info("  config.json: %s", "✓" if (output_dir / "config.json").exists() else "✗")


if __name__ == "__main__":
    main()
