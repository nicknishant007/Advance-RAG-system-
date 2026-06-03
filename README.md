# Advanced Hybrid RAG System

## Overview

Advanced Hybrid RAG System is a Retrieval-Augmented Generation (RAG) application that combines semantic search and keyword search to provide accurate, context-aware answers from a collection of documents.

The system processes PDFs and other documents through an ingestion pipeline, generates vector embeddings, stores them in Qdrant, builds a BM25 index for keyword retrieval, and uses a hybrid retrieval strategy to fetch the most relevant context before generating responses using an LLM.

---

## Features

* Document ingestion pipeline
* PDF text extraction and cleaning
* Metadata generation
* Recursive chunking
* Dense vector embeddings
* Qdrant vector database integration
* BM25 keyword retrieval
* Hybrid retrieval (Semantic + BM25)
* Maximum Marginal Relevance (MMR) reranking
* Cross-encoder reranking
* Streaming LLM responses
* File registry with SQLite
* Incremental ingestion using file hashing
* LangSmith tracing and observability

---

## Architecture

Document Upload
↓
Text Extraction
↓
Text Cleaning
↓
Metadata Generation
↓
Chunking
↓
Embedding Generation
↓
├── Qdrant (Dense Retrieval)
└── BM25 (Sparse Retrieval)
↓
Hybrid Retrieval
↓
MMR Reranking
↓
Cross-Encoder Reranking
↓
LLM Generation
↓
Final Answer

---

## Tech Stack

### Backend

* Python

### Retrieval

* Qdrant
* BM25 (rank-bm25)

### Embeddings

* Sentence Transformers

### LLM

* Mistral AI

### Frameworks

* LangChain
* LangSmith

### Database

* SQLite

---

## Retrieval Pipeline

The retrieval system uses a hybrid search strategy:

### Semantic Search

Dense embeddings are generated for both documents and user queries. Qdrant performs vector similarity search to retrieve semantically relevant chunks.

### Keyword Search

BM25 retrieves chunks containing exact keyword matches.

### Hybrid Fusion

Results from both retrieval methods are combined to improve recall.

### MMR Reranking

Maximum Marginal Relevance removes redundant chunks and increases context diversity.

### Cross-Encoder Reranking

A reranker scores retrieved chunks and selects the most relevant context for the LLM.

---

## Incremental Ingestion

The system avoids reprocessing previously ingested files.

Each file:

* Generates a SHA-256 hash
* Stores hash in SQLite registry
* Checks registry before ingestion

This significantly reduces processing time during repeated ingestion runs.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Ingestion

```bash
python main.py
```

Select:

```text
1. Run Ingestion
```

### Start Chat

```bash
python main.py
```

Select:

```text
2. Start RAG Chat
```

---

## Future Improvements

* Metadata filtering
* Query caching
* Parent-child retrieval
* Context compression
* Multi-vector retrieval
* GPU acceleration
* Hybrid score fusion
* Agentic RAG workflows
* Cloud deployment (Azure / AWS)

---

## Project Goal

This project demonstrates the implementation of a production-style Hybrid Retrieval-Augmented Generation system using modern retrieval techniques, vector databases, reranking strategies, and observability tooling.
