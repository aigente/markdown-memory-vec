"""
Embedding model abstraction layer — dual backend.

Provides two concrete implementations of the :class:`IEmbedder` interface:

1. :class:`OnnxEmbedder` — lightweight (~55 MB deps), uses ``onnxruntime``
   + ``tokenizers``.  Preferred for desktop / resource-constrained envs.
2. :class:`SentenceTransformerEmbedder` — full-featured (~1.5 GB deps), uses
   ``sentence-transformers`` (torch).  Preferred for dev / cloud.

The factory :func:`create_embedder` auto-detects the best available backend
(ONNX first, then sentence-transformers).  All callers should use this factory
or import the specific class they need.

The heavy ML imports are deferred until the first call so that importing this
module is always cheap.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List

from .interfaces import IEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_DIMENSION = 384

# ONNX model cache directory
_ONNX_CACHE_DIR = Path.home() / ".cache" / "aigente" / "models" / "onnx"

# ---------------------------------------------------------------------------
# Availability checks (lazy — avoid heavy imports at module level)
# ---------------------------------------------------------------------------
_sentence_transformers_available: bool | None = None
_onnx_available: bool | None = None


def is_sentence_transformers_available() -> bool:
    """Return ``True`` if the sentence-transformers package is importable.

    The check is deferred to first call to avoid a ~7s import of torch/transformers
    at module load time.
    """
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            import sentence_transformers  # noqa: F401  # type: ignore[import-untyped]

            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
    return _sentence_transformers_available


def is_onnx_available() -> bool:
    """Return ``True`` if onnxruntime + tokenizers are importable."""
    global _onnx_available
    if _onnx_available is None:
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401

            _onnx_available = True
        except ImportError:
            _onnx_available = False
    return _onnx_available


def is_any_backend_available() -> bool:
    """Return ``True`` if at least one embedding backend is available."""
    return is_onnx_available() or is_sentence_transformers_available()


# ---------------------------------------------------------------------------
# HuggingFace cache helpers (shared by both backends)
# ---------------------------------------------------------------------------
def _resolve_hf_snapshot_path(model_name: str) -> str | None:
    """Resolve the local snapshot path for a cached HuggingFace model.

    Returns the absolute path to the latest snapshot directory if the model
    is cached locally, or ``None`` if not found.
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--sentence-transformers--{model_name}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    # Pick the latest snapshot (by modification time)
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        return None

    snapshot_path = str(snapshots[0])
    logger.info("Model '%s' resolved to local HF snapshot: %s", model_name, snapshot_path)
    return snapshot_path


def _ensure_hf_env(model_name: str) -> str:
    """Prepare HuggingFace environment for model loading.

    Returns the model identifier to pass to ``SentenceTransformer()``:
    - Local snapshot path if cached (zero network, bypasses HF Hub entirely).
    - Original model name otherwise (will download via mirror if needed).
    """
    local_path = _resolve_hf_snapshot_path(model_name)
    if local_path:
        return local_path

    # Model not cached — ensure mirror is set for download
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("HF_ENDPOINT not set, using hf-mirror.com for model download")

    return model_name


# ---------------------------------------------------------------------------
# ONNX model path resolution
# ---------------------------------------------------------------------------
def _resolve_onnx_model_dir(model_name: str) -> Path | None:
    """Find the ONNX model directory.

    Search order:
    1. ``~/.cache/aigente/models/onnx/{model_name}/`` (exported or downloaded)
    2. HuggingFace snapshot with ``onnx/model.onnx`` (some models ship ONNX)
    3. HuggingFace snapshot with ``model.onnx`` at root

    Returns the directory containing ``model.onnx`` + ``tokenizer.json``,
    or ``None`` if not found.
    """
    # 1. Our dedicated ONNX cache
    onnx_dir = _ONNX_CACHE_DIR / model_name
    if (onnx_dir / "model.onnx").exists() and (onnx_dir / "tokenizer.json").exists():
        logger.info("ONNX model found in aigente cache: %s", onnx_dir)
        return onnx_dir

    # 2 & 3. Check HF snapshot
    hf_snapshot = _resolve_hf_snapshot_path(model_name)
    if hf_snapshot:
        snap = Path(hf_snapshot)
        # Check onnx/ subfolder first
        onnx_sub = snap / "onnx"
        if (onnx_sub / "model.onnx").exists():
            tokenizer = snap / "tokenizer.json"
            if tokenizer.exists():
                logger.info("ONNX model found in HF snapshot onnx/ subfolder: %s", onnx_sub)
                return onnx_sub  # caller must look for tokenizer.json in parent
        # Check root of snapshot
        if (snap / "model.onnx").exists() and (snap / "tokenizer.json").exists():
            logger.info("ONNX model found in HF snapshot root: %s", snap)
            return snap

    return None


# ---------------------------------------------------------------------------
# ONNX model download from OSS (zip archive)
# ---------------------------------------------------------------------------
_OSS_MODEL_ZIP_DEFAULT = (
    "https://aigente-studio.oss-cn-shanghai.aliyuncs.com" "/models/onnx/paraphrase-multilingual-MiniLM-L12-v2.zip"
)


def download_onnx_model(model_name: str = _DEFAULT_MODEL_NAME) -> Path | None:
    """Download ONNX model zip from OSS and extract to local cache.

    Downloads a single ``.zip`` archive containing ``model.onnx``,
    ``tokenizer.json``, etc. into
    ``~/.cache/aigente/models/onnx/{model_name}/``.

    The zip URL is configurable via the ``ONNX_MODEL_OSS_URL`` env var
    (should point to the full URL of the zip file).

    Returns the cache directory on success, or ``None`` on failure.
    """
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    onnx_dir = _ONNX_CACHE_DIR / model_name
    zip_url = os.environ.get("ONNX_MODEL_OSS_URL", _OSS_MODEL_ZIP_DEFAULT)

    logger.info("Downloading ONNX model from %s ...", zip_url)

    tmp_zip = None
    try:
        # Download to a temp file with progress reporting
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_zip = Path(tmp.name)

        req = urllib.request.urlopen(zip_url)
        total = int(req.headers.get("Content-Length", 0))
        total_mb = total / (1024 * 1024) if total else 0

        downloaded = 0
        last_pct = -1
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(tmp_zip, "wb") as f:
            while True:
                chunk = req.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    # Log every 10%
                    if pct >= last_pct + 10:
                        last_pct = pct
                        logger.info(
                            "Downloading ONNX model: %d%% (%.0f/%.0f MB)",
                            pct,
                            downloaded / (1024 * 1024),
                            total_mb,
                        )

        dl_mb = downloaded / (1024 * 1024)
        logger.info("Download complete (%.1f MB), extracting...", dl_mb)

        # Extract zip to cache dir
        onnx_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(onnx_dir)

        # Verify required files
        if (onnx_dir / "model.onnx").exists() and (onnx_dir / "tokenizer.json").exists():
            logger.info("ONNX model ready at %s", onnx_dir)
            return onnx_dir
        else:
            logger.warning("Zip extracted but required files missing in %s", onnx_dir)
            shutil.rmtree(onnx_dir, ignore_errors=True)
            return None

    except Exception as e:
        logger.warning("ONNX model download failed: %s", e)
        # Clean up partial state
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir, ignore_errors=True)
        return None
    finally:
        if tmp_zip and tmp_zip.exists():
            tmp_zip.unlink()


# ---------------------------------------------------------------------------
# ONNX Embedder
# ---------------------------------------------------------------------------
class OnnxEmbedder(IEmbedder):
    """Embedder backed by ``onnxruntime`` + ``tokenizers``.

    Lightweight alternative (~55 MB) to sentence-transformers (~1.5 GB).
    Produces embeddings identical to the original model (cosine sim > 0.99).

    The ONNX session and tokenizer are loaded lazily on first use and cached
    at class level.

    Parameters
    ----------
    model_name:
        Model identifier. Used to locate the ONNX files.
    model_dir:
        Explicit path to the directory containing ``model.onnx`` and
        ``tokenizer.json``.  If ``None``, auto-resolved via
        :func:`_resolve_onnx_model_dir`.
    """

    # Class-level cache: model_name -> (session, tokenizer)
    _cache: dict[str, tuple[Any, Any]] = {}

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        model_dir: str | Path | None = None,
    ) -> None:
        if not is_onnx_available():
            raise RuntimeError(
                "onnxruntime and/or tokenizers not installed. " "Install with: pip install 'markdown-memory-vec[onnx]'"
            )
        self._model_name = model_name
        self._model_dir = Path(model_dir) if model_dir else None
        self._dimension_override: int | None = None

    def _resolve_dir(self) -> Path:
        """Resolve the model directory, raising if not found."""
        if self._model_dir:
            return self._model_dir
        resolved = _resolve_onnx_model_dir(self._model_name)
        if resolved is None:
            raise FileNotFoundError(
                f"ONNX model files not found for '{self._model_name}'. "
                f"Export with: python -m memory_vec.scripts.export_onnx"
            )
        return resolved

    @property
    def _session_and_tokenizer(self) -> tuple[Any, Any]:
        """Return cached (InferenceSession, Tokenizer) tuple."""
        if self._model_name not in self._cache:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            model_dir = self._resolve_dir()

            # Find model.onnx
            model_path = model_dir / "model.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"model.onnx not found in {model_dir}")

            # Find tokenizer.json — may be in parent dir (HF onnx/ subfolder case)
            tokenizer_path = model_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                tokenizer_path = model_dir.parent / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"tokenizer.json not found in {model_dir} or {model_dir.parent}")

            logger.info("Loading ONNX model: %s", model_path)
            logger.info("Loading tokenizer: %s", tokenizer_path)

            # Configure ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Use CPU provider — Apple Silicon's Accelerate/NEON is efficient enough
            # for a 384-dim embedding model (~4ms per query).
            # Note: CoreMLExecutionProvider doesn't support dynamic-shape transformer
            # models (session.run() crashes on batch>1 due to unbounded dimensions).
            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX session using CPU provider")

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            # Set max length consistent with the model (128 tokens for MiniLM)
            tokenizer.enable_truncation(max_length=128)
            tokenizer.enable_padding(length=128)

            self._cache[self._model_name] = (session, tokenizer)

            # Infer dimension from model output shape
            outputs = session.get_outputs()
            # token_embeddings output shape: [batch, seq_len, dim]
            if outputs and len(outputs[0].shape) == 3:
                self._dimension_override = outputs[0].shape[2]

        return self._cache[self._model_name]

    def _encode(self, texts: List[str]) -> Any:
        """Tokenize, run ONNX inference, mean-pool, and L2-normalize."""
        import numpy as np

        session, tokenizer = self._session_and_tokenizer

        # Tokenize
        encodings = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Build feed dict based on what the model actually expects
        input_names = {inp.name for inp in session.get_inputs()}
        feeds: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = token_type_ids

        # Run inference
        outputs = session.run(None, feeds)
        # outputs[0] = token_embeddings: [batch, seq_len, hidden_dim]
        token_embeddings = outputs[0]

        # Mean pooling (only over non-padding tokens)
        # Matches sentence-transformers default: mean pooling WITHOUT L2 normalization
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled

    # -- IEmbedder interface implementation ----------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        result = self._encode([text])
        return result[0].tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into vectors."""
        if not texts:
            return []
        result = self._encode(texts)
        return [vec.tolist() for vec in result]

    @property
    def dimension(self) -> int:
        """Return the vector dimension (triggers model load if unknown)."""
        if self._dimension_override is not None:
            return self._dimension_override
        _ = self._session_and_tokenizer
        return self._dimension_override or _DEFAULT_DIMENSION


# ---------------------------------------------------------------------------
# Sentence-transformers Embedder (original implementation)
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder(IEmbedder):
    """Embedder backed by the ``sentence-transformers`` library.

    The underlying ``SentenceTransformer`` model is loaded lazily on the first
    call to :meth:`embed` or :meth:`embed_batch`, and is then cached as a
    class-level variable so it is shared across all instances using the same
    model name.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to
        ``"paraphrase-multilingual-MiniLM-L12-v2"`` (384-dim, 50+ languages).
    """

    # Class-level cache: model_name -> SentenceTransformer instance
    _model_cache: dict[str, Any] = {}

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        if not is_sentence_transformers_available():
            raise RuntimeError(
                "sentence-transformers is not installed. " "Install it with: pip install 'markdown-memory-vec[vector]'"
            )
        self._model_name = model_name
        self._dimension_override: int | None = None

    # -- lazy model loading --------------------------------------------------

    @property
    def _model(self) -> Any:
        """Return the cached ``SentenceTransformer`` instance."""
        if self._model_name not in self._model_cache:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            model_path = _ensure_hf_env(self._model_name)
            logger.info("Loading sentence-transformer model: %s", model_path)
            model = SentenceTransformer(model_path)
            self._model_cache[self._model_name] = model
            # Infer dimension from the model
            dim: int = model.get_sentence_embedding_dimension()  # type: ignore[assignment]
            self._dimension_override = dim
        return self._model_cache[self._model_name]

    # -- IEmbedder interface implementation ----------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        result = self._model.encode([text], show_progress_bar=False)
        return result[0].tolist()  # type: ignore[union-attr, no-any-return]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into vectors."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]  # type: ignore[union-attr]

    @property
    def dimension(self) -> int:
        """Return the vector dimension (triggers model load if unknown)."""
        if self._dimension_override is not None:
            return self._dimension_override
        # Trigger model load to discover dimension
        _ = self._model
        return self._dimension_override or _DEFAULT_DIMENSION


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def create_embedder(
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    prefer_onnx: bool = True,
) -> IEmbedder:
    """Create the best available embedder.

    Selection order (when *prefer_onnx* is ``True``, the default):

    1. **ONNX** — if ``onnxruntime`` + ``tokenizers`` installed AND the ONNX
       model files exist locally (or can be downloaded from OSS).
    2. **sentence-transformers** — fallback if ONNX deps or files are missing.

    Raises :class:`RuntimeError` if neither backend is available.
    """
    if prefer_onnx and is_onnx_available():
        onnx_dir = _resolve_onnx_model_dir(model_name)

        # Model files not found locally — try downloading from OSS
        if onnx_dir is None:
            logger.info(
                "ONNX model files not found locally for '%s', " "attempting download from OSS...",
                model_name,
            )
            try:
                onnx_dir = download_onnx_model(model_name)
            except Exception as e:
                logger.warning("ONNX model download failed: %s", e)
                onnx_dir = None

        if onnx_dir is not None:
            logger.info("Using ONNX embedding backend (lightweight)")
            return OnnxEmbedder(model_name=model_name)
        else:
            logger.info(
                "ONNX model not available for '%s', " "falling back to sentence-transformers",
                model_name,
            )

    if is_sentence_transformers_available():
        logger.info("Using sentence-transformers embedding backend")
        return SentenceTransformerEmbedder(model_name=model_name)

    # ONNX available but no model files, and no sentence-transformers
    if is_onnx_available():
        raise RuntimeError(
            f"ONNX model files not found for '{model_name}' and sentence-transformers "
            f"is not installed. Either:\n"
            f"  1. Ensure network access to download from OSS\n"
            f"  2. Export manually: python -m memory_vec.scripts.export_onnx\n"
            f"  3. Install sentence-transformers: pip install 'markdown-memory-vec[vector]'"
        )

    raise RuntimeError(
        "No embedding backend available. Install one of:\n"
        "  pip install 'markdown-memory-vec[onnx]'   # lightweight (~55MB)\n"
        "  pip install 'markdown-memory-vec[vector]'  # full (~1.5GB)"
    )
