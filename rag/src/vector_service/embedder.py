"""
src/vector_service/embedder.py
================================

External Libraries & Models:
1. Sentence-Transformers (https://www.sbert.net/)
   - Library : sentence-transformers
   - Purpose : High-level interface for text embedding models

2. HuggingFace Model Hub (https://huggingface.co/)
   - Model   : Configurable via settings.EMBEDDING_MODEL_ID
   - Example : BAAI/bge-small-en-v1.5
   - Purpose : Generate dense vector embeddings for retrieval

3. PyTorch (https://pytorch.org/)
   - Library : torch
   - Purpose : Backend for model inference + device detection

Local embedding using sentence-transformers (Hugging Face).

No OpenAI key required.  Model is downloaded once and cached by
the HF hub (~80 MB for bge-small-en-v1.5).

Device detection order:  CUDA  →  MPS (Apple Silicon)  →  CPU
"""

from __future__ import annotations

import logging
from typing import List

from config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            log.info("Embedder: using CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            log.info("Embedder: using MPS (Apple Silicon)")
            return "mps"
    except ImportError:
        pass
    log.info("Embedder: using CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class Embedder:
    """
    Wraps a SentenceTransformer model with batched encoding.

    Usage
    -----
    embedder = Embedder()                       # loads default model from config
    vecs = embedder.encode(["text1", "text2"])  # returns list[list[float]]
    """

    def __init__(self, model_id: str | None = None):
        from sentence_transformers import SentenceTransformer

        model_id  = model_id or settings.EMBEDDING_MODEL_ID
        device    = _detect_device()

        log.info("Loading embedding model '%s' on %s …", model_id, device)
        self._model  = SentenceTransformer(model_id, device=device)
        self._device = device
        log.info("Embedding model loaded. Dimension: %d", self.dimension)

    # ---- public interface --------------------------------------------------

    @property
    def dimension(self) -> int:
        """Output vector dimensionality."""
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        """
        Encode a list of strings into embedding vectors.

        Parameters
        ----------
        texts      : strings to embed
        batch_size : override the default from config

        Returns
        -------
        list of float vectors (one per input text)
        """
        if not texts:
            return []

        bs = batch_size or settings.EMBEDDING_BATCH_SIZE

        # BGE models benefit from a query prefix when encoding queries.
        # We expose a separate method for that, so here we encode as-is.
        vectors = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # unit-norm → cosine sim == dot product
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query string.
        Adds BGE-style instruction prefix if the model is a BGE variant.
        """
        model_id = settings.EMBEDDING_MODEL_ID.lower()
        if "bge" in model_id:
            # BGE models recommend a retrieval instruction for queries
            prefixed = f"Represent this sentence for searching relevant passages: {query}"
        else:
            prefixed = query

        vec = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return (or create) the module-level Embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
