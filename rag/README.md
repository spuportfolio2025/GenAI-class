# Financial News RAG

Query-driven financial news RAG with in-memory retrieval and FinBERT sentiment.

## Pipeline (per query, fully in-memory)

```
User Query
   │
   ├─ 1. Collect    NewsAPI + RSS feeds → newspaper3k full-text scraping
   ├─ 2. Clean      Boilerplate removal + NLTK sentence chunking
   ├─ 3. Retrieve   SentenceTransformer embed → numpy cosine similarity → top 20
   ├─ 4. Rerank     Cohere rerank-english-v3.0 → top 5
   ├─ 5. Generate   OpenAI gpt-4o-mini answer
   └─ 6. Sentiment  FinBERT weighted by rerank score
```

## Project structure

```
fin_rag/
├── config/
│   └── settings.py                 ← all API keys + tuneable params
├── src/
│   ├── data_engine/
│   │   ├── collector.py            ← JIT fetch: query parsing, NewsAPI, RSS, scraping
│   │   └── cleaner.py              ← text cleaning + sentence chunking
│   ├── vector_service/
│   │   ├── embedder.py             ← SentenceTransformer (BAAI/bge-small-en-v1.5)
│   │   └── retriever.py            ← in-memory numpy cosine similarity
│   └── rag/
│       ├── reranker.py             ← Cohere rerank
│       ├── generator.py            ← OpenAI GPT answer
│       └── sentiment.py            ← FinBERT sentiment scoring
├── main.py
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export NEWSAPI_KEY="..."
export COHERE_API_KEY="..."
export OPENAI_API_KEY="..."

python main.py --query "What is happening with Meta lately?"
```
