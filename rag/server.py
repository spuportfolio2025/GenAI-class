"""
server.py
=========

External Libraries & APIs:
1. FastAPI (https://fastapi.tiangolo.com/)
   - Library : fastapi
   - Purpose : Build high-performance async REST API

2. Uvicorn (https://www.uvicorn.org/)
   - Library : uvicorn
   - Purpose : ASGI server for running FastAPI applications

3. Pydantic (https://docs.pydantic.dev/)
   - Library : pydantic
   - Purpose : Request validation and data modeling

4. Server-Sent Events (SSE) (https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
   - Standard : HTTP streaming protocol
   - Purpose  : Real-time streaming of pipeline progress to frontend

FastAPI server that wraps the RAG pipeline and streams
each step as Server-Sent Events (SSE) to the frontend.

Run:
    pip install fastapi uvicorn
    uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import logging
import sys
import time
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("server")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Financial News RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


# ── SSE helper ────────────────────────────────────────────────────────────
def sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Streaming pipeline ─────────────────────────────────────────────────────
async def stream_pipeline(query: str) -> AsyncGenerator[str, None]:
    """
    Run the RAG pipeline step-by-step and yield SSE events so the
    frontend can render progress in real time.
    """
    loop = asyncio.get_event_loop()

    # ── Step 1: Collect ──────────────────────────────────────────────────
    yield sse("step_start", {"step": 1, "label": "Collecting news", "icon": "satellite"})
    t0 = time.perf_counter()

    from src.data_engine.collector import collect, detect_company, detect_days_back
    company   = detect_company(query)
    days_back = detect_days_back(query)
    yield sse("step_detail", {"text": f"Detected company: {company or '(generic)'} · lookback: {days_back} days"})

    articles = await loop.run_in_executor(None, collect, query)
    elapsed  = time.perf_counter() - t0

    if not articles:
        yield sse("error", {"message": "No articles found. Check your API keys or try a different query."})
        return

    yield sse("step_done", {
        "step": 1,
        "label": "Collecting news",
        "stats": f"{len(articles)} articles scraped in {elapsed:.1f}s",
        "items": [{"title": a.title, "source": a.source,
                   "date": a.published_at.strftime("%b %d") if a.published_at else "",
                   "url": a.url}
                  for a in articles[:8]],  # show first 8 in UI
    })

    # ── Step 2: Clean & chunk ────────────────────────────────────────────
    yield sse("step_start", {"step": 2, "label": "Cleaning & chunking", "icon": "scissors"})
    t0 = time.perf_counter()

    from src.data_engine.cleaner import clean_and_chunk
    chunks  = await loop.run_in_executor(None, clean_and_chunk, articles)
    elapsed = time.perf_counter() - t0

    avg_tok = int(sum(c.get("token_count", 0) for c in chunks) / max(len(chunks), 1))
    yield sse("step_done", {
        "step": 2,
        "label": "Cleaning & chunking",
        "stats": f"{len(chunks)} chunks · avg {avg_tok} tokens · {elapsed:.2f}s",
    })

    # ── Step 3: Embed + retrieve ─────────────────────────────────────────
    yield sse("step_start", {"step": 3, "label": "Embedding & retrieval", "icon": "search"})
    t0 = time.perf_counter()

    from src.vector_service.retriever import retrieve
    from config import settings
    candidates = await loop.run_in_executor(None, retrieve, query, chunks, settings.TOP_K_RETRIEVE)
    elapsed    = time.perf_counter() - t0

    yield sse("step_done", {
        "step": 3,
        "label": "Embedding & retrieval",
        "stats": f"top {len(candidates)} candidates · cosine sim · {elapsed:.2f}s",
    })

    # ── Step 4: Rerank ───────────────────────────────────────────────────
    yield sse("step_start", {"step": 4, "label": "Reranking", "icon": "sort"})
    t0 = time.perf_counter()

    from src.rag.reranker import rerank
    top_chunks = await loop.run_in_executor(None, rerank, query, candidates, settings.TOP_N_RERANK)
    elapsed    = time.perf_counter() - t0

    yield sse("step_done", {
        "step": 4,
        "label": "Reranking",
        "stats": f"Cohere rerank → top {len(top_chunks)} · {elapsed:.2f}s",
        "chunks": [
            {
                "rank":   i + 1,
                "score":  round(c.get("rerank_score", 0), 4),
                "title":  c.get("title", "")[:70],
                "source": c.get("source", ""),
                "date":   c.get("published_at", "")[:10],
                "snippet": c.get("text", "")[:180] + "…",
            }
            for i, c in enumerate(top_chunks)
        ],
    })

    # ── Step 5: Generate ─────────────────────────────────────────────────
    yield sse("step_start", {"step": 5, "label": "Generating answer", "icon": "cpu"})
    t0 = time.perf_counter()

    from src.rag.generator import generate_answer
    answer  = await loop.run_in_executor(None, generate_answer, query, top_chunks)
    elapsed = time.perf_counter() - t0

    yield sse("step_done", {
        "step": 5,
        "label": "Generating answer",
        "stats": f"gpt-4o-mini · {elapsed:.2f}s",
    })

    # ── Step 6: Sentiment ────────────────────────────────────────────────
    yield sse("step_start", {"step": 6, "label": "FinBERT sentiment", "icon": "bar-chart"})
    t0 = time.perf_counter()

    from src.rag.sentiment import score_sentiment
    sentiment = await loop.run_in_executor(None, score_sentiment, top_chunks)
    elapsed   = time.perf_counter() - t0

    probs = sentiment["probabilities"]
    yield sse("step_done", {
        "step": 6,
        "label": "FinBERT sentiment",
        "stats": f"{sentiment['final_label'].upper()} · {elapsed:.2f}s",
        "sentiment": {
            "label":    sentiment["final_label"],
            "positive": round(probs.get("positive", 0), 3),
            "negative": round(probs.get("negative", 0), 3),
            "neutral":  round(probs.get("neutral",  0), 3),
        },
    })

    # ── Final result ─────────────────────────────────────────────────────
    yield sse("result", {
        "answer":    answer,
        "query":     query,
        "sentiment": sentiment["final_label"],
        "sources": [
            {
                "title":  c.get("title",        ""),
                "source": c.get("source",       ""),
                "date":   c.get("published_at", "")[:10],
                "url":    c.get("url",          ""),
                "score":  round(c.get("rerank_score", 0), 4),
            }
            for c in top_chunks
        ],
    })


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/query")
async def query_endpoint(req: QueryRequest):
    return StreamingResponse(
        stream_pipeline(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("frontend/index.html", encoding="utf-8") as f:
        return f.read()


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
