"""
Embedding model abstraction layer.

Provides a concrete implementation :class:`SentenceTransformerEmbedder` that
wraps the ``sentence-transformers`` library and conforms to the
:class:`IEmbedder` interface defined in ``interfaces.py``.

The heavy ML import is deferred until the first call so that importing this
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
# Availability check
# ---------------------------------------------------------------------------
_sentence_transformers_available = False
try:
    import sentence_transformers  # noqa: F401  # type: ignore[import-untyped]

    _sentence_transformers_available = True
except ImportError:
    pass


def is_sentence_transformers_available() -> bool:
    """Return ``True`` if the sentence-transformers package is importable."""
    return _sentence_transformers_available


# ---------------------------------------------------------------------------
# Sentence-transformers implementation
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_DIMENSION = 384


def _ensure_hf_env(model_name: str) -> None:
    """Set HuggingFace environment variables for reliable model loading.

    Strategy:
    - If model is already cached locally → set ``HF_HUB_OFFLINE=1`` (no network).
    - Otherwise respect user-set ``HF_ENDPOINT`` (do **not** override it).
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return  # User explicitly requested offline mode

    # Check HuggingFace hub cache for the model
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--sentence-transformers--{model_name}"
    if model_dir.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("Model '%s' found in local cache, using offline mode", model_name)
        return

    # Model not cached — log a hint if no endpoint is set
    if not os.environ.get("HF_ENDPOINT"):
        logger.info(
            "HF_ENDPOINT not set. If downloading is slow, set HF_ENDPOINT to a mirror "
            "(e.g. https://hf-mirror.com for users in China)."
        )


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
        if not _sentence_transformers_available:
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

            _ensure_hf_env(self._model_name)
            logger.info("Loading sentence-transformer model: %s", self._model_name)
            model = SentenceTransformer(self._model_name)
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
