"""
src/vector_service/retriever.py
================================
Pure in-memory similarity search using numpy cosine similarity.

Usage
-----
    candidates = retrieve(query, chunks, top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from config import settings
from src.vector_service.embedder import get_embedder

log = logging.getLogger(__name__)


def retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = settings.TOP_K_RETRIEVE,
) -> List[Dict[str, Any]]:
    """
    Embed the query and all chunks, return top_k by cosine similarity.

    Parameters
    ----------
    query  : user question
    chunks : output of cleaner.clean_and_chunk()
    top_k  : number of candidates to return

    Returns
    -------
    List of chunk dicts with an added "distance" key (lower = more similar).
    """
    if not chunks:
        log.warning("retrieve() called with empty chunk list.")
        return []

    embedder = get_embedder()

    # Embed all chunk texts in one batched call
    texts    = [c["text"] for c in chunks]
    doc_vecs = np.array(embedder.encode(texts))        # (N, dim), unit-norm
    q_vec    = np.array(embedder.encode_query(query))  # (dim,),   unit-norm

    # Cosine similarity = dot product for unit-norm vectors
    scores      = doc_vecs @ q_vec                     # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = dict(chunks[idx])
        c["distance"] = float(1.0 - scores[idx])      # keep API consistent with chroma
        results.append(c)

    log.info("In-memory retrieve: %d chunks → top %d returned", len(chunks), len(results))
    return results
