"""
api/routes/chat.py
-------------------
Two endpoints:

1. POST /chat/stream
   Streams the LLM answer token-by-token using Server-Sent Events (SSE).
   The frontend JS reads this with fetch() + ReadableStream.

2. POST /chat/retrieve
   Returns the retrieved + reranked chunks (no LLM call).
   Powers the Sources panel in the frontend.

HOW SSE WORKS:
    Client opens a fetch() stream to /chat/stream.
    Server keeps the connection open and sends lines like:
        data: {"type": "token", "content": "Hello"}
        data: {"type": "token", "content": " world"}
        data: {"type": "done"}
    Client reads chunks as they arrive and appends to the UI.
    This is how you get the "typing" effect in the frontend.
"""

import json
import traceback
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from api.services.rag_service import RAGService

router = APIRouter()


# ── Request schema ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5   # how many final chunks to pass to the LLM


# ── 1. Streaming chat ─────────────────────────────────────────────────────────
@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams the answer token-by-token as Server-Sent Events.

    Full flow:
        ChatRequest.question
            │
            ▼
        RAGService.stream_answer()              ← api/services/rag_service.py
            │
            ▼
        RAGPipeline.stream_query()              ← app/generation/rag_pipeline.py
            │
            ├── RetrievalPipeline.retrieve()    (hybrid BM25+dense, MMR, rerank)
            │
            └── ResponseGenerator.stream_response()
                    └── Gemini LLM (streaming=True)
                            │
                            ▼
                    yields token strings  ←──────── we catch each one here
                            │
                            ▼
                    SSE line: data: {"type": "token", "content": "..."}
    """
    question = req.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question cannot be empty."}
        )

    def event_stream():
        try:
            # RAGService.stream_answer() calls YOUR RAGPipeline.stream_query()
            # which is a generator — each iteration yields one token string
            for token in RAGService.stream_answer(
                question=question,
                top_k=req.top_k
            ):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"

            # Tell the client the stream is finished
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            err_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {err_payload}\n\n"
            print(traceback.format_exc())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if behind a proxy
        },
    )


# ── 2. Retrieval only (for Sources panel) ─────────────────────────────────────
@router.post("/retrieve")
async def chat_retrieve(req: ChatRequest):
    """
    Runs ONLY the retrieval stage (no LLM), returns the chunks.
    The frontend calls this in parallel with /stream to populate
    the Sources panel while the answer is streaming.

    Full flow:
        ChatRequest.question
            │
            ▼
        RAGService.retrieve_sources()           ← api/services/rag_service.py
            │
            ▼
        RetrievalPipeline.retrieve()            ← app/ingestion/retrieval/retrieval_pipeline.py
            │
            ├── HybridRetriever.retrieve()      (BM25 + Qdrant dense)
            ├── MMRRetriever.rerank()            (diversity reranking on dense)
            ├── merge dense + sparse
            └── Reranker.rerank()               (CrossEncoder final rerank)
            │
            ▼
        List[Document] → serialized to JSON for the frontend
    """
    question = req.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question cannot be empty."}
        )

    try:
        sources = RAGService.retrieve_sources(
            question=question,
            top_k=req.top_k
        )
        return JSONResponse({"sources": sources})

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )