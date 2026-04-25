"""
src/rag/sentiment.py
=====================

External Libraries & Models:
1. HuggingFace Transformers (https://huggingface.co/docs/transformers)
   - Library : transformers
   - Purpose : Load pretrained NLP models and tokenizers

2. FinBERT Model (https://huggingface.co/ProsusAI/finbert)
   - Model ID : ProsusAI/finbert
   - Type     : Pretrained financial sentiment classification model
   - Labels   : positive / negative / neutral
   - Usage    : _load_finbert(), _score_texts()

3. PyTorch (https://pytorch.org/)
   - Library : torch
   - Purpose : Model inference (forward pass, softmax)

FinBERT sentiment scoring on the top reranked chunks.

Output
------
{
  "final_label":        "positive" | "negative" | "neutral",
  "probabilities":      {"positive": float, "negative": float, "neutral": float},
  "chunk_level_scores": [{"positive": ..., "negative": ..., "neutral": ...}, ...]
}

Sentiment is weighted-averaged by each chunk's Cohere rerank score,
so higher-relevance chunks contribute more to the final label.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

_FINBERT_MODEL_ID = "ProsusAI/finbert"

# Module-level singletons — loaded once, reused across calls
_tokenizer = None
_model     = None


def _load_finbert():
    global _tokenizer, _model
    if _tokenizer is not None:
        return
    log.info("Loading FinBERT (%s) …", _FINBERT_MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_ID)
    _model     = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_ID)
    _model.eval()
    log.info("FinBERT loaded.")


def _score_texts(texts: List[str]) -> List[Dict[str, float]]:
    """Return per-text FinBERT probability dicts."""
    _load_finbert()
    results = []
    for text in texts:
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            probs = torch.softmax(_model(**inputs).logits, dim=1).numpy()[0]

        label_map = _model.config.id2label
        results.append({label_map[i].lower(): float(probs[i]) for i in range(len(probs))})
    return results


def score_sentiment(top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run FinBERT on the top reranked chunks and aggregate into a single sentiment.

    Parameters
    ----------
    top_chunks : output of reranker.rerank() — each chunk must have "rerank_score"

    Returns
    -------
    Sentiment dict (see module docstring).
    """
    texts          = [c["text"] for c in top_chunks]
    chunk_scores   = _score_texts(texts)

    # Weight by rerank score so more relevant chunks dominate
    weights = np.array([c.get("rerank_score", 1.0) for c in top_chunks])
    weights = weights / weights.sum()

    agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for w, s in zip(weights, chunk_scores):
        for label in agg:
            agg[label] += w * s.get(label, 0.0)

    final_label = max(agg, key=agg.get)
    log.info(
        "FinBERT sentiment: %s  (pos=%.3f  neg=%.3f  neu=%.3f)",
        final_label.upper(), agg["positive"], agg["negative"], agg["neutral"],
    )

    return {
        "final_label":        final_label,
        "probabilities":      agg,
        "chunk_level_scores": chunk_scores,
    }
