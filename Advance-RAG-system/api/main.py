"""
api/main.py
-----------
FastAPI application entry point.

HOW TO RUN (from your project ROOT, not from inside api/):
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Your project root must have the 'app/' folder next to the 'api/' folder
so that Python can import your existing RAG code (app.generation.*, etc.)
"""

import sys
from pathlib import Path

# ── Make sure the project root is importable ──────────────────────────────────
# This lets Python find  app/generation/..., app/ingestion/...  etc.
ROOT = Path(__file__).resolve().parent.parent   # goes up from api/ → project root
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router
from api.services.rag_service import RAGService
from api.services.ingest_service import IngestService


# ── Create app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Advance RAG System",
    description=(
        "Hybrid RAG: BM25 sparse + Qdrant dense retrieval, "
        "MMR diversity reranking, CrossEncoder final reranking, "
        "Gemini streaming generation."
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the HTML frontend (opened from file:// or any localhost port) to call
# this API.  Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the frontend ────────────────────────────────────────────────────────
# Visit http://localhost:8000/  →  serves frontend/index.html automatically
app.mount(
    "/static",
    StaticFiles(directory=str(ROOT / "frontend"), html=True),
    name="frontend",
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(chat_router,   prefix="/chat",   tags=["Chat"])
app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])


# ── Startup: warm up heavy models once ───────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("🚀 Warming up RAG pipeline…")
    IngestService.initialize_registry()   # create SQLite table if not exists
    RAGService.warmup()                   # loads embedding + reranker models
    print("✅ API ready.")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "service": "Advance RAG System"}