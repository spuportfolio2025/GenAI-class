# GenAI RAG Pipeline with Sentiment Analysis

## Overview
This project builds a Retrieval-Augmented Generation (RAG) pipeline for financial news analysis.

The pipeline:
1. Collects and preprocesses financial news
2. Converts news into structured chunks
3. Indexes articles for retrieval
4. Performs RAG-based question answering
5. Adds sentiment analysis to enhance interpretation

---

## Pipeline Structure

### 1. Preprocessing (`clean_preprocessing.py`)
- Fetch news data (NewsAPI)
- Clean and filter articles
- Chunk text into smaller pieces
- Output:
  - `df_news`
  - `df_chunks`
  - metadata

---

### 2. RAG Pipeline (`rag_pipeline.py`)
- Convert dataframe → articles
- Index articles into vector store
- Retrieve relevant content
- Generate answer using LLM
- Perform sentiment analysis

---

### 3. Full Pipeline Example

```python
from clean_preprocessing import run_preprocessing
from rag_pipeline import (
    dataframe_to_articles,
    index_articles,
    rag_query_with_sentiment
)

# Step 1: preprocessing
outputs = run_preprocessing(
    question="What is happening with Meta lately?",
    ticker="META",
    include_edgar=False
)

df_news = outputs["df_news"]

# Step 2: convert
articles = dataframe_to_articles(df_news)

# Step 3: index
index_articles(articles)

# Step 4: query
result = rag_query_with_sentiment(
    "What is happening with Meta lately?"
)

print(result)
