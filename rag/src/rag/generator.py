"""
src/rag/generator.py
=====================
LLM answer generation using OpenAI (gpt-4o-mini).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import openai

from config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analyst assistant with access to recent news articles.

Guidelines:
- Answer the user's question using ONLY the provided news context below.
- If the context lacks sufficient information, explicitly say so.
- Cite source titles when referencing specific facts (e.g., "According to Reuters, …").
- Be concise, factual, and neutral in tone.
- Do NOT fabricate any numbers, dates, or quotes.
"""


def _build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        title  = chunk.get("title",        "Unknown")
        source = chunk.get("source",       "Unknown")
        date   = chunk.get("published_at", "")[:10]
        parts.append(f"[Source {i} | {source} | {title} | {date}]\n{chunk['text']}")

    context_block = "\n\n".join(parts)
    return f"=== NEWS CONTEXT ===\n{context_block}\n\n=== QUESTION ===\n{query}"


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int    = 512,
) -> str:
    """
    Generate a grounded answer from the top reranked chunks.

    Parameters
    ----------
    query          : original user question
    context_chunks : top-N reranked chunks from reranker.rerank()

    Returns
    -------
    Answer string from GPT.
    """
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = _build_prompt(query, context_chunks)

    log.info("Sending prompt to OpenAI (%s) …", settings.OPENAI_MODEL)
    try:
        response = client.chat.completions.create(
            model       = settings.OPENAI_MODEL,
            temperature = temperature,
            max_tokens  = max_tokens,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.error("OpenAI call failed: %s", exc)
        return f"[LLM error: {exc}]"
