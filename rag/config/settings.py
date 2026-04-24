"""
config/settings.py
==================
Single source of truth for all configuration.
"""

import os

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY",    "YOUR_NEWSAPI_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YOUR_COHERE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")

# ---------------------------------------------------------------------------
# Embedding  (Hugging Face, runs locally, no API key needed)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_ID   = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Retrieval & Reranking
# ---------------------------------------------------------------------------
TOP_K_RETRIEVE      = 20   # candidates from in-memory similarity search
TOP_N_RERANK        = 5    # final chunks fed to LLM + FinBERT
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# ---------------------------------------------------------------------------
# LLM (OpenAI)
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_TOKENS        = 400
CHUNK_OVERLAP_SENTENCES = 1
CHUNK_MIN_CHARS         = 100
AVG_CHARS_PER_TOKEN     = 4

# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
MAX_WORKERS       = 10
MAX_PER_SOURCE    = 7
PAGE_SIZE         = 100
MIN_TEXT_LEN      = 200
DEFAULT_DAYS_BACK = 7
MAX_DAYS_BACK     = 25

# ---------------------------------------------------------------------------
# Source lists
# ---------------------------------------------------------------------------
NEWSAPI_DOMAINS = ",".join([
    "reuters.com", "apnews.com", "cnbc.com", "marketwatch.com",
    "finance.yahoo.com", "thestreet.com", "nasdaq.com", "fortune.com",
    "fool.com", "businesswire.com", "prnewswire.com", "techcrunch.com",
    "venturebeat.com", "wired.com", "arstechnica.com", "zdnet.com",
    "theguardian.com",
])

RSS_FEEDS = [
    ("Reuters Business",      "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Technology",    "https://feeds.reuters.com/reuters/technologyNews"),
    ("AP News Business",      "https://feeds.apnews.com/rss/apf-business"),
    ("MarketWatch",           "https://feeds.marketwatch.com/marketwatch/topstories"),
    ("TechCrunch",            "https://techcrunch.com/category/startups/feed/"),
    ("VentureBeat",           "https://venturebeat.com/feed/"),
    ("The Guardian Business", "https://www.theguardian.com/business/rss"),
    ("BBC Business",          "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Fortune",               "https://fortune.com/feed"),
]

SOURCE_PRIORITY = {
    "reuters": 10, "associated press": 10, "ap news": 10,
    "cnbc": 9, "marketwatch": 9, "yahoo finance": 8, "nasdaq": 8,
    "fortune": 8, "the street": 7, "business wire": 7, "pr newswire": 7,
    "techcrunch": 6, "venturebeat": 6, "wired": 6, "ars technica": 6,
    "zdnet": 5, "motley fool": 5,
}

ALIAS_MAP = {
    "meta":       {"meta platforms", "facebook", "META"},
    "apple":      {"aapl", "AAPL"},
    "google":     {"alphabet", "googl", "GOOGL", "GOOG"},
    "amazon":     {"amzn", "AMZN"},
    "tesla":      {"tsla", "TSLA"},
    "nvidia":     {"nvda", "NVDA"},
    "microsoft":  {"msft", "MSFT"},
    "netflix":    {"nflx", "NFLX"},
    "intel":      {"intc", "INTC"},
    "amd":        {"advanced micro devices", "AMD"},
}
