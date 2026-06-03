# 🚀 Advance RAG System

A production-style Retrieval-Augmented Generation (RAG) system built using LangChain,LangSmith, Qdrant,Semantic search, BM25, MMR, Cross-Encoder Reranking, and LLMs.

The system supports document ingestion, hybrid retrieval, intelligent reranking, and grounded response generation from user-provided knowledge bases.

---

# ✨ Features

- PDF Document Ingestion
- Multi-threaded Processing Pipeline
- Automatic Metadata Generation
- Recursive Text Chunking
- Dense Vector Embeddings
- Local Qdrant Vector Database
- BM25 Sparse Retrieval
- Hybrid Search (Dense + Sparse)
- MMR (Maximal Marginal Relevance) Diversification
- Cross-Encoder Reranking
- Duplicate Document Detection using File Hashing
- Streaming LLM Responses
- LangSmith Observability & Tracing
- Local-First Architecture

---

# 🏗️ System Architecture

```text
                 ┌──────────────────┐
                 │   PDF Documents   │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Text Extraction  │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Text Cleaning    │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Metadata Builder │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Recursive Chunk  │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ BGE Embeddings   │
                 └─────────┬────────┘
                           │
          ┌────────────────┴───────────────┐
          │                                │
          ▼                                ▼
 ┌─────────────────┐             ┌─────────────────┐
 │ Qdrant VectorDB │             │   BM25 Index    │
 └─────────────────┘             └─────────────────┘


                   USER QUERY
                         │
                         ▼
                Query Embedding
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Dense Retrieval              Sparse Retrieval
      Top 15                        Top 7
          │                             │
          ▼                             │
        MMR                             │
      Top 8                             │
          │                             │
          └──────────────┬──────────────┘
                         ▼
                 Merge & Deduplicate
                         │
                         ▼
                 Cross Encoder
                   Reranker
                    Top 5
                         │
                         ▼
                       LLM
                         │
                         ▼
                  Final Response
```

---

# ⚙️ Technology Stack

## Retrieval

- Qdrant
- BM25
- MMR

## Embeddings

- BAAI/bge-small-en-v1.5

## Reranking

- cross-encoder/ms-marco-MiniLM-L-6-v2

## Framework

- LangChain

## Monitoring

- LangSmith

## Database

- SQLite
- Qdrant

## LLMs

- Gemini 2.5 Flash
- Mistral Large (Supported)

---

# 📂 Project Structure

```text
app/
│
├── ingestion/
│   ├── extractor/
│   ├── chunking/
│   ├── embedding/
│   ├── retrieval/
│   ├── registry/
│   ├── vectordb/
│   └── pipeline/
│
├── generation/
│   ├── llm.py
│   ├── prompt_builder.py
│   ├── response_generator.py
│   └── rag_pipeline.py
│
storage/
│
├── qdrant_data/
├── bm25/
└── registry.db
│
main.py
```

---

# 🔄 Ingestion Pipeline

Each document follows the workflow below:

```text
PDF
 ↓
Extract Text
 ↓
Clean Text
 ↓
Generate Metadata
 ↓
Chunk Document
 ↓
Generate Embeddings
 ↓
Store in Qdrant
 ↓
Store in BM25
 ↓
Mark as Processed
```

---

# 🔍 Retrieval Pipeline

The retrieval system uses Hybrid Search.

## Dense Retrieval

Vector similarity search from Qdrant.

```text
Query
 ↓
Embedding
 ↓
Top 15 Dense Chunks
```

---

## MMR Diversification

Removes redundant dense chunks.

```text
15 Dense Chunks
 ↓
MMR
 ↓
8 Diverse Chunks
```

---

## Sparse Retrieval

Keyword-based BM25 retrieval.

```text
Query
 ↓
BM25
 ↓
Top 7 Sparse Chunks
```

---

## Cross Encoder Reranking

Final relevance optimization.

```text
8 MMR Chunks
 +
7 BM25 Chunks
 ↓
Cross Encoder
 ↓
Top 5 Chunks
```

---

# 🛡️ Duplicate Document Protection

To avoid reprocessing the same files:

```text
File
 ↓
SHA256 Hash
 ↓
Registry Check
 ↓
Already Processed?
 ↓
YES → Skip
NO  → Process
```

Benefits:

- Prevents duplicate embeddings
- Faster ingestion
- Reduced storage usage
- Consistent indexing

---

# 🚀 Performance Optimizations

Several optimizations were introduced during development.

## SQLite Concurrency Fix

Enabled:

```python
PRAGMA journal_mode=WAL;
```

Benefits:

- Eliminated database locking
- Improved multi-threaded ingestion

---

## Hybrid Retrieval Optimization

### Old Architecture

```text
Dense (15)
+
BM25 (15)
↓
Embed Again
↓
MMR
↓
Rerank
```

Problems:

- Duplicate embeddings
- High latency
- Shape mismatch issues
- Additional embedding overhead

---

### New Architecture

```text
Dense (15)
↓
MMR (8)

BM25 (7)

Merge
↓
Rerank (5)
```

Benefits:

- Lower latency
- Reduced embedding calls
- Cleaner retrieval flow
- Better retrieval diversity
- Simpler MMR implementation

---

# 📈 Latency Improvements

Measured using LangSmith tracing.

## Before Optimization

| Query | Latency |
|---------|---------|
| Run 1 | 48.52s |
| Run 2 | 45.37s |

Average:

```text
~47 Seconds
```

---

## After Optimization

| Query | Latency |
|---------|---------|
| Run 1 | 2.18s |
| Run 2 | 2.26s |
| Run 3 | 1.86s |

Average:

```text
~2 Seconds
```

### Improvement

```text
47s → 2s
≈ 95% Latency Reduction
```

---

# ▶️ Running the Project

## Clone Repository

```bash
git clone <your-repo-url>
cd Advance-RAG-system
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Advance-RAG-System
```

---

## Run Ingestion

```bash
python main.py
```

Choose:

```text
1. Run Ingestion
```

---

## Start Chat

```bash
python main.py
```

Choose:

```text
2. Start RAG Chat
```

---

# 📊 Observability

LangSmith is integrated for:

- Tracing
- Latency Monitoring
- Token Usage Tracking
- Retrieval Debugging
- Performance Evaluation
- Prompt Inspection

---

# 💡 Example Query Flow

```text
User Question
      ↓
Generate Query Embedding
      ↓
Dense Retrieval (15)
      ↓
MMR (8)
      ↓
BM25 Retrieval (7)
      ↓
Merge Results
      ↓
Cross Encoder Rerank
      ↓
Top 5 Context Chunks
      ↓
LLM Generation
      ↓
Final Answer
```

---

# 👨‍💻 Author

**Nishant Kumar**

Built as a deep dive into modern Retrieval-Augmented Generation systems, hybrid search, vector databases, reranking, observability, and production-grade AI retrieval pipelines.

---

## ⭐ If you found this project useful, consider starring the repository.
