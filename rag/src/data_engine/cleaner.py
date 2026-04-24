"""
src/data_engine/cleaner.py
===========================
Text cleaning and sentence-based chunking.

Input : list[CandidateArticle]   (raw_text populated)
Output: list[dict]               (clean chunks ready for vector store)

Each chunk dict schema:
  {
    "id":           str,    # "{article_idx}_chunk_{chunk_idx}"
    "text":         str,    # cleaned chunk text
    "title":        str,
    "source":       str,
    "published_at": str,    # ISO-8601
    "url":          str,
    "chunk_index":  int,
    "chunk_total":  int,
    "token_count":  int,
    "char_count":   int,
  }
"""

from __future__ import annotations

import logging
import re
from typing import List

from config import settings
from src.data_engine.collector import CandidateArticle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise patterns (from original notebook)
# ---------------------------------------------------------------------------

_BOILERPLATE = re.compile(
    "|".join([
        r"subscribe to our newsletter",
        r"sign up for",
        r"enable javascript",
        r"cookies? to improve",
        r"please enable",
        r"advertisement",
        r"click here to",
        r"read the full article",
        r"all rights reserved",
        r"©\s*\d{4}",
    ]),
    re.IGNORECASE,
)

_HEADER_NOISE = re.compile(
    r"^(share|read more|advertisement|skip to|sign in|subscribe|follow us|newsletter)$",
    re.IGNORECASE | re.MULTILINE,
)

_EARNINGS_BOILERPLATE = re.compile(
    "|".join([
        r"(?i)forward.looking statements.{0,2000}",
        r"(?i)safe harbor.{0,1000}",
        r"(?i)private securities litigation.{0,500}",
    ]),
    re.DOTALL,
)

_WHITESPACE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Sentence tokenisation (NLTK with plain fallback)
# ---------------------------------------------------------------------------

_sent_tokenize = None

def _get_sent_tokenize():
    global _sent_tokenize
    if _sent_tokenize is not None:
        return _sent_tokenize
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize as _st
        _sent_tokenize = _st
    except Exception:
        # Fallback: split on ". " boundaries
        def _st(text):
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        _sent_tokenize = _st
    return _sent_tokenize


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Remove boilerplate, noise lines, and normalise whitespace."""
    # Strip earnings disclaimer blocks
    text = _EARNINGS_BOILERPLATE.sub("", text)
    # Strip header noise single-line tokens
    text = _HEADER_NOISE.sub("", text)
    # Remove lines that match boilerplate patterns
    lines = []
    for line in text.splitlines():
        if not _BOILERPLATE.search(line):
            lines.append(line)
    text = "\n".join(lines)
    # Normalise whitespace
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Sentence-based chunking with overlap
# ---------------------------------------------------------------------------

def _chunk_sentences(
    text: str,
    max_tokens: int = settings.CHUNK_MAX_TOKENS,
    overlap_sentences: int = settings.CHUNK_OVERLAP_SENTENCES,
    min_chars: int = settings.CHUNK_MIN_CHARS,
    avg_chars: int = settings.AVG_CHARS_PER_TOKEN,
) -> List[str]:
    """
    Split cleaned text into overlapping sentence-window chunks.
    Token budget is approximated as len(text) / avg_chars.
    """
    sent_tok = _get_sent_tokenize()
    sentences = [s for s in sent_tok(text) if len(s) > 10]

    if not sentences:
        return [text] if len(text) >= min_chars else []

    max_chars = max_tokens * avg_chars
    chunks: List[str] = []
    i = 0

    while i < len(sentences):
        current_sentences = []
        current_chars = 0

        while i < len(sentences):
            s_chars = len(sentences[i])
            if current_chars + s_chars > max_chars and current_sentences:
                break
            current_sentences.append(sentences[i])
            current_chars += s_chars
            i += 1

        chunk_text = " ".join(current_sentences).strip()
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)

        # Overlap: step back by overlap_sentences
        if overlap_sentences > 0 and i < len(sentences):
            i = max(i - overlap_sentences, i - len(current_sentences) + 1)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_and_chunk(articles: List[CandidateArticle]) -> List[dict]:
    """
    Clean article text and produce chunk dicts ready for vector indexing.

    Parameters
    ----------
    articles : output of collector.collect()

    Returns
    -------
    List of chunk dicts (see module docstring for schema).
    """
    all_chunks: List[dict] = []

    for art_idx, article in enumerate(articles):
        text = _clean_text(article.raw_text)
        if len(text) < settings.MIN_TEXT_LEN:
            log.debug("Article '%s' too short after cleaning – skipped", article.title[:60])
            continue

        chunks = _chunk_sentences(text)
        if not chunks:
            continue

        pub_str = article.published_at.isoformat() if article.published_at else ""

        for chunk_idx, chunk_text in enumerate(chunks):
            token_est = max(1, len(chunk_text) // settings.AVG_CHARS_PER_TOKEN)
            all_chunks.append({
                "id":           f"art{art_idx}_chunk{chunk_idx}",
                "text":         chunk_text,
                "title":        article.title,
                "source":       article.source,
                "published_at": pub_str,
                "url":          article.url,
                "chunk_index":  chunk_idx,
                "chunk_total":  len(chunks),
                "token_count":  token_est,
                "char_count":   len(chunk_text),
            })

        log.debug("  '%s' → %d chunks", article.title[:60], len(chunks))

    log.info("clean_and_chunk: %d articles → %d chunks", len(articles), len(all_chunks))
    return all_chunks
