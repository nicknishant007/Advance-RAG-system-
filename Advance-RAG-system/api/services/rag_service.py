"""
api/services/rag_service.py
----------------------------
Thin wrapper around YOUR existing pipeline classes.

YOUR CODE PATH (untouched):
    app/generation/rag_pipeline.py        →  RAGPipeline
    app/ingestion/retrieval/              →  RetrievalPipeline (called inside RAGPipeline)
    app/ingestion/embedding/embedder.py   →  Embedder
    app/ingestion/retrieval/reranker.py   →  Reranker

DATA FLOW this service calls:
    user question
        │
        ▼
    HybridRetriever.retrieve(query)          ← hybrid_retrieval.py
        ├── Embedder.embed([query])           ← embedder.py  (BGE small)
        ├── QdrantManager.semantic_search()   ← qdrant_client.py
        └── BM25Indexer.search()             ← bm25_index.py
        │
        ▼
    MMRRetriever.rerank(dense_results)       ← mmr.py
        │
        ▼
    merge dense (MMR) + sparse (BM25)
        │
        ▼
    Reranker.rerank(candidates)              ← reranker.py  (CrossEncoder)
        │
        ▼
    ResponseGenerator.stream_response()      ← response_generator.py
        └── PromptBuilder.build()            ← prompt_builder.py
        └── LLM.get_llm()  (Gemini)         ← llm.py
"""

from typing import Generator, List

# ── Lazy singletons (models load only once on first call) ────────────────────
_rag_pipeline = None
_retrieval_pipeline = None


def _get_rag_pipeline():
    """
    Returns the singleton RAGPipeline.
    First call imports + instantiates it (loads Gemini LLM config).
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        from app.generation.rag_pipeline import RAGPipeline   # YOUR file
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def _get_retrieval_pipeline():
    """
    Returns the singleton RetrievalPipeline.
    Used separately so we can call JUST retrieval without generation
    (for the /chat/retrieve endpoint that powers the Sources panel).
    """
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        from app.ingestion.retrieval.retrieval_pipeline import RetrievalPipeline  # YOUR file
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


class RAGService:

    @staticmethod
    def warmup():
        """
        Pre-load embedding model + CrossEncoder so the first user query
        doesn't have a cold-start delay.
        """
        try:
            from app.ingestion.embedding.embedder import Embedder   # YOUR file
            Embedder.load_model()
        except Exception as e:
            print(f"⚠️  Embedder warmup skipped: {e}")

        try:
            from app.ingestion.retrieval.reranker import Reranker   # YOUR file
            Reranker.load_model()
        except Exception as e:
            print(f"⚠️  Reranker warmup skipped: {e}")

    @staticmethod
    def stream_answer(question: str, top_k: int = 5) -> Generator[str, None, None]:
        """
        Call YOUR RAGPipeline.stream_query() and yield each token.

        Connected to: app/generation/rag_pipeline.py → stream_query()
        Which internally calls:
            1. RetrievalPipeline.retrieve()   (hybrid + MMR + rerank)
            2. ResponseGenerator.stream_response()  (Gemini streaming)
        """
        pipeline = _get_rag_pipeline()
        yield from pipeline.stream_query(user_query=question, top_k=top_k)

    @staticmethod
    def retrieve_sources(question: str, top_k: int = 5) -> List[dict]:
        """
        Call YOUR RetrievalPipeline.retrieve() and return chunk metadata
        so the frontend Sources panel can show exactly what was retrieved.

        Connected to:
            app/ingestion/retrieval/retrieval_pipeline.py → retrieve()
            Which returns List[langchain_core.documents.Document]
            Each Document has:
                .page_content  → the chunk text
                .metadata      → { source, file_name, file_size }  (from MetadataBuilder)
        """
        pipeline = _get_retrieval_pipeline()
        chunks = pipeline.retrieve(query=question, top_k=top_k)

        sources = []
        seen_texts = set()

        for chunk in chunks:
            # Deduplicate by first 80 chars of text
            key = chunk.page_content[:80]
            if key in seen_texts:
                continue
            seen_texts.add(key)

            sources.append({
                # From MetadataBuilder.build() → app/ingestion/extractor/metadata_builder.py
                "file_name": chunk.metadata.get("file_name", "Unknown"),
                "source":    chunk.metadata.get("source", ""),
                "file_size": chunk.metadata.get("file_size", 0),
                # Chunk text — first 350 chars as preview, full text for expand
                "preview":   chunk.page_content[:350] + (
                    "…" if len(chunk.page_content) > 350 else ""
                ),
                "full_text": chunk.page_content,
            })

        return sources