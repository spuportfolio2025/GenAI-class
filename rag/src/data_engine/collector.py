"""
src/data_engine/collector.py
=============================
JIT news collection:
  1. Parse user query → company + days_back
  2. Fetch article metadata from NewsAPI and RSS feeds (concurrently)
  3. Scrape full article text via newspaper3k (ThreadPoolExecutor)
  4. Return list[CandidateArticle]
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import feedparser
import requests

log = logging.getLogger(__name__)

# Optional heavy deps
try:
    from newspaper import Article as NewspaperArticle
    _HAS_NEWSPAPER = True
except ImportError:
    _HAS_NEWSPAPER = False
    log.warning("newspaper3k not installed. Full-text scraping disabled.")

try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

from config import settings


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidateArticle:
    title: str
    source: str
    published_at: datetime
    url: str
    raw_text: str = ""
    drop_reason: str = ""


# ---------------------------------------------------------------------------
# Query parsing helpers
# ---------------------------------------------------------------------------

_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        if not _HAS_SPACY:
            raise ImportError("spacy not installed: pip install spacy && python -m spacy download en_core_web_sm")
        import spacy as _spacy
        _NLP = _spacy.load("en_core_web_sm")
    return _NLP


def detect_company(question: str) -> str:
    """Extract primary company/entity name from a free-text question."""
    try:
        nlp = _get_nlp()
        doc = nlp(question)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if orgs:
            return orgs[0]
    except Exception:
        pass  # fall through to heuristic

    skip = {"what", "who", "how", "when", "where", "why", "is", "are",
            "the", "a", "an", "about", "happening", "latest", "news"}
    for word in question.split():
        clean = word.strip("?.,!")
        if clean and clean[0].isupper() and clean.lower() not in skip:
            return clean
    return ""


def detect_days_back(question: str) -> int:
    """Map temporal expressions in the question to an integer day count."""
    q = question.lower()
    patterns = [
        (r"\btoday\b|\bright now\b|\bcurrently\b", 1),
        (r"\bthis week\b|\blately\b|\brecently\b", 7),
        (r"\blast week\b", 7),
        (r"\bthis month\b|\blast month\b", 30),
        (r"\blast (\d+) days?\b", "days"),
        (r"\blast (\d+) weeks?\b", "weeks"),
        (r"\blast (\d+) months?\b", "months"),
    ]
    for pattern, unit in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        if isinstance(unit, int):
            return min(unit, settings.MAX_DAYS_BACK)
        n = int(m.group(1))
        days = n * (7 if unit == "weeks" else 30 if unit == "months" else 1)
        return min(days, settings.MAX_DAYS_BACK)
    return min(settings.DEFAULT_DAYS_BACK, settings.MAX_DAYS_BACK)


def get_company_aliases(company: str) -> set:
    c = company.lower()
    aliases = {c}
    aliases.update({a.lower() for a in settings.ALIAS_MAP.get(c, set())})
    return aliases


# ---------------------------------------------------------------------------
# NewsAPI fetch
# ---------------------------------------------------------------------------

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def _fetch_newsapi(company: str, days_back: int) -> List[CandidateArticle]:
    api_key = settings.NEWSAPI_KEY
    if not api_key or api_key == "YOUR_NEWSAPI_KEY":
        log.warning("NEWSAPI_KEY not set – skipping NewsAPI.")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    seen_urls: set = set()
    candidates: List[CandidateArticle] = []

    query_variants = [
        {"q": f'"{company}"', "domains": settings.NEWSAPI_DOMAINS,
         "pageSize": settings.PAGE_SIZE},
    ]

    for params_extra in query_variants:
        params = {
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "apiKey": api_key,
            **params_extra,
        }
        try:
            resp = requests.get(_NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("NewsAPI request failed: %s", exc)
            continue

        for art in data.get("articles", []):
            url = art.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                pub = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            candidates.append(CandidateArticle(
                title=art.get("title") or "",
                source=art.get("source", {}).get("name") or "",
                published_at=pub,
                url=url,
            ))

    log.info("NewsAPI returned %d candidates for '%s'", len(candidates), company)
    return candidates


# ---------------------------------------------------------------------------
# RSS fetch
# ---------------------------------------------------------------------------

def _fetch_rss(company: str, days_back: int) -> List[CandidateArticle]:
    aliases = get_company_aliases(company)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    candidates: List[CandidateArticle] = []

    def _parse_feed(name_url):
        name, url = name_url
        try:
            feed = feedparser.parse(url)
            results = []
            for entry in feed.entries:
                title = entry.get("title", "")
                link  = entry.get("link", "")
                if not link:
                    continue
                # Filter by company mention in title
                title_lower = title.lower()
                if not any(alias in title_lower for alias in aliases):
                    continue
                # Parse date
                pub = datetime.now(timezone.utc)
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        from time import mktime
                        pub = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
                    except Exception:
                        pass
                if pub < cutoff:
                    continue
                results.append(CandidateArticle(
                    title=title, source=name, published_at=pub, url=link
                ))
            return results
        except Exception as exc:
            log.debug("RSS feed '%s' failed: %s", name, exc)
            return []

    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as pool:
        futures = {pool.submit(_parse_feed, nf): nf[0] for nf in settings.RSS_FEEDS}
        for fut in as_completed(futures):
            candidates.extend(fut.result())

    log.info("RSS feeds returned %d candidates for '%s'", len(candidates), company)
    return candidates


# ---------------------------------------------------------------------------
# Full-text scraping
# ---------------------------------------------------------------------------

def _scrape_article(candidate: CandidateArticle) -> CandidateArticle:
    """Download and extract full article text. Returns the same object mutated."""
    if not _HAS_NEWSPAPER:
        # Fallback: plain requests + first 2000 chars of raw HTML (rough)
        try:
            r = requests.get(candidate.url, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            # Strip obvious HTML tags
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text).strip()
            candidate.raw_text = text[:4000]
        except Exception as exc:
            candidate.drop_reason = f"scrape_error: {exc}"
        return candidate

    try:
        art = NewspaperArticle(candidate.url)
        art.download()
        art.parse()
        candidate.raw_text = art.text or ""
        if not candidate.title and art.title:
            candidate.title = art.title
    except Exception as exc:
        candidate.drop_reason = f"newspaper_error: {exc}"
    return candidate


def _scrape_all(candidates: List[CandidateArticle]) -> List[CandidateArticle]:
    """Concurrently scrape full text for all candidates."""
    results: List[CandidateArticle] = []
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as pool:
        futures = {pool.submit(_scrape_article, c): c for c in candidates}
        for i, fut in enumerate(as_completed(futures), 1):
            art = fut.result()
            results.append(art)
            if i % 10 == 0:
                log.info("  Scraped %d / %d articles …", i, len(candidates))
    return results


# ---------------------------------------------------------------------------
# Source deduplication / capping
# ---------------------------------------------------------------------------

def _get_source_priority(source: str) -> int:
    src = source.lower()
    return max((v for k, v in settings.SOURCE_PRIORITY.items() if k in src), default=3)


def _deduplicate_and_cap(candidates: List[CandidateArticle]) -> List[CandidateArticle]:
    seen_urls: set = set()
    source_counts: dict = {}
    kept = []
    # Sort by priority descending
    for c in sorted(candidates, key=lambda x: _get_source_priority(x.source), reverse=True):
        if c.url in seen_urls:
            continue
        if source_counts.get(c.source, 0) >= settings.MAX_PER_SOURCE:
            continue
        seen_urls.add(c.url)
        source_counts[c.source] = source_counts.get(c.source, 0) + 1
        kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect(question: str) -> List[CandidateArticle]:
    """
    JIT pipeline entry point.

    Parameters
    ----------
    question : free-text user query (e.g. "What is happening with Meta lately?")

    Returns
    -------
    List of CandidateArticle with .raw_text populated (empty if scraping failed).
    """
    company   = detect_company(question)
    days_back = detect_days_back(question)

    if not company:
        log.warning("Could not detect a company name in: '%s'", question)
        return []

    log.info("Collecting news for company='%s', days_back=%d", company, days_back)

    # Fetch metadata from all sources
    candidates: List[CandidateArticle] = []
    candidates.extend(_fetch_newsapi(company, days_back))
    candidates.extend(_fetch_rss(company, days_back))

    # Dedup and cap per source before scraping (saves bandwidth)
    candidates = _deduplicate_and_cap(candidates)
    log.info("After dedup/cap: %d candidates to scrape", len(candidates))

    # Scrape full text
    candidates = _scrape_all(candidates)

    # Filter empty / very short articles
    valid = [c for c in candidates if len(c.raw_text) >= settings.MIN_TEXT_LEN and not c.drop_reason]
    log.info("Valid articles after scraping: %d", len(valid))
    return valid
