"""
api/routes/ingest.py
---------------------
Two endpoints:

1. POST /ingest/upload
   Upload a single file → save → run Worker.process() on it immediately.
   This is the "Add Document" button in the frontend.

2. POST /ingest/run
   Trigger a full scan of data/incoming/ using IngestionPipeline.run()
   (uses your ThreadPoolExecutor — processes all unprocessed files in parallel).
"""

import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.services.ingest_service import IngestService

router = APIRouter()


# ── 1. Upload a single document ───────────────────────────────────────────────
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and immediately ingest it into the RAG knowledge base.

    Steps:
        1. Validate file extension
        2. Save to data/incoming/<filename>           ← IngestService.save_uploaded_file()
        3. Call Worker.process(file_path)             ← IngestService.process_single_file()
               └── ExtractorRouter.extract()          (PDF/DOCX/CSV/image)
               └── TextCleaner.clean()
               └── MetadataBuilder.build()
               └── RecursiveChunker.chunk()
               └── Embedder.embed()
               └── QdrantManager.store_documents()
               └── BM25Indexer.add_documents()
               └── RegistryManager.mark_processed()
        4. Return JSON with chunks_created count

    The frontend shows this count next to the file name in the sidebar.
    """
    # Validate extension before reading bytes
    if not IngestService.validate_extension(file.filename):
        allowed = ", ".join(IngestService.get_allowed_extensions())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed}"
        )

    # Read file bytes from the upload
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {e}")

    # Save to data/incoming/
    try:
        file_path = IngestService.save_uploaded_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process through YOUR Worker pipeline
    try:
        result = IngestService.process_single_file(file_path)
        return JSONResponse(content={
            "status":         "success" if not result["skipped"] else "skipped",
            "file":           result["file"],
            "chunks_created": result["chunks_created"],
            "skipped":        result["skipped"],
            "message":        result["message"],
        })
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ── 2. Run full pipeline scan ─────────────────────────────────────────────────
@router.post("/run")
async def run_full_pipeline():
    """
    Trigger a full scan of data/incoming/ directory using IngestionPipeline.
    Processes all files that haven't been hashed into the registry yet.
    Uses your ThreadPoolExecutor (parallel workers).

    Call this if you manually dropped files into data/incoming/
    and want to ingest them all at once without uploading through the UI.
    """
    try:
        result = IngestService.run_full_pipeline()
        return JSONResponse(content={
            "status":               "success",
            "total_chunks_created": result["total_chunks_created"],
            "message":              result["message"],
        })
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline run failed: {e}")