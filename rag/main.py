"""
main.py
========
Top-level RAG pipeline.

Flow
----
  1. Collect   — JIT news fetch (NewsAPI + RSS + scraping)
  2. Clean     — boilerplate removal + sentence chunking
  3. Retrieve  — in-memory cosine similarity → top 20
  4. Rerank    — Cohere rerank → top 5
  5. Generate  — OpenAI GPT answer
  6. Sentiment — FinBERT weighted sentiment over top 5 chunks

Usage
-----
    python main.py --query "What is happening with Meta lately?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("main")

from src.data_engine.collector       import collect
from src.data_engine.cleaner         import clean_and_chunk
from src.vector_service.retriever    import retrieve
from src.rag.reranker                import rerank
from src.rag.generator               import generate_answer
from src.rag.sentiment               import score_sentiment
from config                          import settings


def run_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a single query.
    All intermediate data lives in memory and is discarded after the call.
    """
    # ------------------------------------------------------------------
    # Step 1: Collect
    # ------------------------------------------------------------------
    log.info("=== Step 1: Collecting news ===")
    articles = collect(query)
    if not articles:
        log.warning("No articles collected — check API keys or try a different query.")
        return {"query": query, "answer": "No articles found.", "sources": [], "sentiment": None}

    # ------------------------------------------------------------------
    # Step 2: Clean & chunk
    # ------------------------------------------------------------------
    log.info("=== Step 2: Cleaning & chunking ===")
    chunks = clean_and_chunk(articles)
    log.info("%d chunks ready", len(chunks))

    # ------------------------------------------------------------------
    # Step 3: In-memory similarity search
    # ------------------------------------------------------------------
    log.info("=== Step 3: Similarity search (top %d) ===", settings.TOP_K_RETRIEVE)
    candidates = retrieve(query, chunks, top_k=settings.TOP_K_RETRIEVE)

    if not candidates:
        return {"query": query, "answer": "No relevant chunks found.", "sources": [], "sentiment": None}

    # ------------------------------------------------------------------
    # Step 4: Rerank → top 5
    # ------------------------------------------------------------------
    log.info("=== Step 4: Reranking → top %d ===", settings.TOP_N_RERANK)
    top_chunks = rerank(query, candidates, top_n=settings.TOP_N_RERANK)
    _log_rerank_table(top_chunks)

    # ------------------------------------------------------------------
    # Step 5: Generate answer
    # ------------------------------------------------------------------
    log.info("=== Step 5: Generating answer ===")
    answer = generate_answer(query, top_chunks)

    # ------------------------------------------------------------------
    # Step 6: FinBERT sentiment
    # ------------------------------------------------------------------
    log.info("=== Step 6: FinBERT sentiment ===")
    sentiment = score_sentiment(top_chunks)

    return {
        "query":  query,
        "answer": answer,
        "sources": [
            {
                "title":        c.get("title",        ""),
                "source":       c.get("source",       ""),
                "published_at": c.get("published_at", "")[:10],
                "url":          c.get("url",          ""),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "snippet":      c["text"][:200],
            }
            for c in top_chunks
        ],
        "sentiment": sentiment,
    }


def _log_rerank_table(top_chunks):
    log.info("  %-5s %-8s  %-50s  %s", "Rank", "Score", "Title", "Source")
    log.info("  %s", "-" * 80)
    for i, c in enumerate(top_chunks, 1):
        log.info("  %-5d %-8.4f  %-50s  %s",
                 i,
                 c.get("rerank_score", 0.0),
                 c.get("title",  "")[:48],
                 c.get("source", "")[:20])


def _print_result(result: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"QUERY:  {result['query']}")
    print("=" * 70)
    print(f"\nANSWER:\n{result['answer']}")

    print("\nSOURCES:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  {i}. [{s['rerank_score']:.4f}]  {s['title']}  |  {s['source']}  |  {s['published_at']}")

    if result.get("sentiment"):
        sent  = result["sentiment"]
        probs = sent["probabilities"]
        print(
            f"\nSENTIMENT: {sent['final_label'].upper()}"
            f"  (pos={probs['positive']:.3f}"
            f"  neg={probs['negative']:.3f}"
            f"  neu={probs['neutral']:.3f})"
        )
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Financial News RAG")
    parser.add_argument("--query", type=str, required=True,
                        help='e.g. "What is happening with Meta lately?"')
    parser.add_argument("--json", action="store_true",
                        help="Print result as JSON")
    args = parser.parse_args()

    result = run_pipeline(args.query)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
