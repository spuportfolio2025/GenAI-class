"""
src/rag/reranker.py
====================
Re-rank retrieved chunks for higher relevance precision.

Strategy
--------
* If USE_COHERE_RERANK=True (default) → Cohere cross-encoder rerank API
* Otherwise                           → return top-N by raw vector distance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cohere rerank
# ---------------------------------------------------------------------------

def _cohere_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    try:
        import cohere
    except ImportError:
        log.warning("cohere package not installed – falling back to vector ranking.")
        return _vector_rerank(candidates, top_n)

    api_key = settings.COHERE_API_KEY
    if not api_key or api_key == "YOUR_COHERE_KEY":
        log.warning("COHERE_API_KEY not set – falling back to vector ranking.")
        return _vector_rerank(candidates, top_n)

    co   = cohere.Client(api_key)
    docs = [c["text"] for c in candidates]

    try:
        response = co.rerank(
            model     = settings.COHERE_RERANK_MODEL,
            query     = query,
            documents = docs,
            top_n     = min(top_n, len(docs)),
        )
    except Exception as exc:
        log.error("Cohere rerank failed: %s – falling back to vector ranking.", exc)
        return _vector_rerank(candidates, top_n)

    reranked = []
    for result in response.results:
        chunk = dict(candidates[result.index])       # shallow copy
        chunk["rerank_score"] = float(result.relevance_score)
        reranked.append(chunk)

    log.info("Cohere rerank selected %d / %d chunks", len(reranked), len(candidates))
    return reranked


# ---------------------------------------------------------------------------
# Fallback: sort by cosine distance (no external API)
# ---------------------------------------------------------------------------

def _vector_rerank(
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    """Return top_n candidates sorted by ascending cosine distance."""
    sorted_candidates = sorted(candidates, key=lambda c: c.get("distance", 1.0))
    top = sorted_candidates[:top_n]
    # Add a synthetic rerank_score for downstream consistency
    for c in top:
        c.setdefault("rerank_score", 1.0 - c.get("distance", 0.0))
    log.info("Vector-fallback rerank selected %d / %d chunks", len(top), len(candidates))
    return top


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = settings.TOP_N_RERANK,
) -> List[Dict[str, Any]]:
    """
    Re-rank candidate chunks and return the best top_n.

    Parameters
    ----------
    query      : original user query
    candidates : output of chroma_manager.similarity_search()
    top_n      : number of chunks to return

    Returns
    -------
    List of chunk dicts, each with an added "rerank_score" key.
    """
    if not candidates:
        return []

    return _cohere_rerank(query, candidates, top_n)
