# GenAI RAG Pipeline with Sentiment Analysis

## Project Overview
This project implements a query-driven financial news Retrieval-Augmented Generation (RAG) system. The goal is to provide concise, relevant, and sentiment-aware answers to user queries using real-time financial news data.

The system collects news from multiple sources, processes and retrieves the most relevant content, and generates natural language responses using a large language model. In addition, it applies domain-specific sentiment analysis (FinBERT) to quantify market sentiment. This allows users to better understand both the information and the underlying tone of financial news.

---

## Architecture Diagram

```mermaid
flowchart TD
    A[User Query]
    B[Data Collection: NewsAPI, RSS, Scraping]
    C[Data Cleaning: Boilerplate Removal, Chunking]
    D[Retrieval: SentenceTransformer, Cosine Similarity]
    E[Reranking: Cohere Reranker]
    F[Generation: GPT-4o-mini]
    G[Sentiment Analysis: FinBERT Weighted Score]
    H[Final Output: Answer and Sentiment]

    A --> B --> C --> D --> E --> F --> G --> H
```

---

## Pipeline Structure

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
rag/
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

---

## Setup and Execution Instructions

To run the project locally, users should first clone the repository from GitHub and navigate into the project directory. After that, all required dependencies can be installed using the provided requirements file. The system relies on several external APIs, so users need to configure their API keys before execution. Specifically, the OpenAI API key, NewsAPI key, and Cohere API key should be added to the `rag/config/settings.py` file or set as an environment variable. Once the environment is properly configured, the pipeline can be executed by running the main script, which will trigger the full workflow from data collection to final response generation.

## Quick start

```bash
cd rag
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export NEWSAPI_KEY="..."
export COHERE_API_KEY="..."
export OPENAI_API_KEY="..."

python main.py --query "What is happening with Meta lately?"
```

---

## Implemented Features vs. Planned Features

The current system successfully implements an end-to-end Retrieval-Augmented Generation pipeline. It integrates multiple data sources, including NewsAPI, RSS feeds, and web scraping, to collect financial news in real time. The collected data is cleaned and segmented into manageable text chunks before being transformed into embeddings using a SentenceTransformer model. These embeddings enable semantic retrieval through cosine similarity, followed by a reranking step using the Cohere model to improve relevance. The final response is generated using a large language model, and sentiment analysis is applied using FinBERT, with scores weighted by retrieval relevance. In addition, a logging system has been introduced to track the execution of the pipeline and improve transparency during debugging.

Despite these implemented components, several planned features remain under development. The system currently operates entirely in-memory, and future improvements include integrating a vector database such as FAISS or Pinecone to enhance scalability. Real-time streaming updates for financial news and a frontend interface for user interaction are also planned. Furthermore, the evaluation framework can be extended with more rigorous metrics such as precision, recall, and ranking quality. Improvements in sentiment aggregation and interpretability are also expected, along with potential support for multiple languages.

---

## Known Limitations and Technical Debt

The current implementation prioritizes functionality and rapid prototyping, which introduces several limitations. The in-memory design restricts scalability and may not perform well with large datasets. The sequential nature of API calls can lead to increased latency, especially when multiple external services are involved. In addition, the system relies heavily on third-party APIs, which introduces potential instability and dependency risks.

From a software engineering perspective, error handling and retry mechanisms are still limited, which may affect robustness in real-world scenarios. The sentiment aggregation approach is relatively simple and may not fully capture the complexity of financial language and context. Moreover, the system does not currently implement caching or persistence, which leads to redundant computations for repeated queries.

These limitations reflect deliberate trade-offs made during development, where the primary focus was on demonstrating the core technical workflow and validating the feasibility of the approach.
