"""
api/services/ingest_service.py
-------------------------------
Wraps YOUR existing ingestion code for the upload + add-document feature.

YOUR CODE PATH (untouched):
    app/ingestion/queue/worker.py             →  Worker.process(file_path)
    app/ingestion/pipeline/ingestion_pipeline.py → IngestionPipeline.run()
    app/ingestion/registry/registry_manager.py   → RegistryManager
    app/ingestion/registry/hash.py               → generate_file_hash

FULL INGEST DATA FLOW (what happens when a file is uploaded):

    uploaded file saved to  data/incoming/<filename>
        │
        ▼
    Worker.process(file_path)
        │
        ├── generate_file_hash(file_path)          ← hash.py
        ├── RegistryManager.is_processed(hash)     ← registry_manager.py + db.py
        │       if already processed → SKIP
        │
        ├── ExtractorRouter.extract(file_path)     ← extractor_router.py
        │       .pdf   → PyMuPDFLoader             ← pdf_loader.py
        │       .docx  → python-docx               ← docx_loader.py
        │       .csv   → pandas                    ← csv_loader.py
        │       .jpg/.png → Tesseract OCR          ← tesseract_orc.py
        │
        ├── TextCleaner.clean(raw_text)            ← text_cleaner.py
        ├── MetadataBuilder.build(file_path)       ← metadata_builder.py
        ├── RecursiveChunker.chunk(text, metadata) ← recursive_chunker.py
        ├── Embedder.embed(texts)                  ← embedder.py  (BGE small)
        ├── QdrantManager.store_documents()        ← qdrant_client.py
        ├── BM25Indexer.add_documents()            ← bm25_index.py
        └── RegistryManager.mark_processed()      ← registry_manager.py
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # project root
INCOMING_DIR = ROOT / "data" / "incoming"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".jpg", ".jpeg", ".png"}


class IngestService:

    @staticmethod
    def initialize_registry():
        """
        Creates the SQLite file_registry table if it doesn't exist yet.
        Connected to: app/ingestion/registry/registry_manager.py → initialize()
        """
        try:
            from app.ingestion.registry.registry_manager import RegistryManager  # YOUR file
            RegistryManager.initialize()
            print("✅ Registry initialized.")
        except Exception as e:
            print(f"⚠️  Registry init warning: {e}")

    @staticmethod
    def save_uploaded_file(filename: str, content: bytes) -> Path:
        """
        Saves the uploaded bytes to data/incoming/<filename>.
        Returns the full Path so Worker.process() can read it.

        data/incoming/  is the same folder your LocalScanner watches,
        so running IngestionPipeline.run() later picks it up too.
        """
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        dest = INCOMING_DIR / filename
        with open(dest, "wb") as f:
            f.write(content)
        return dest

    @staticmethod
    def process_single_file(file_path: Path) -> dict:
        """
        Process ONE file through the full ingestion pipeline.

        Connected to: app/ingestion/queue/worker.py → Worker.process()

        Returns a dict:
            {
                "file": filename,
                "chunks_created": int,
                "skipped": bool,      ← True if hash already in registry
                "message": str
            }
        """
        from app.ingestion.queue.worker import Worker   # YOUR file

        chunks = Worker.process(str(file_path))

        skipped = (chunks == [])   # Worker returns [] when file is already processed

        return {
            "file":           file_path.name,
            "chunks_created": len(chunks),
            "skipped":        skipped,
            "message": (
                f"Skipped — '{file_path.name}' is already in the knowledge base."
                if skipped
                else f"Ingested {len(chunks)} chunks from '{file_path.name}'."
            ),
        }

    @staticmethod
    def run_full_pipeline() -> dict:
        """
        Scan data/incoming/ and process ALL unprocessed files.
        Uses your ThreadPoolExecutor-based IngestionPipeline.

        Connected to: app/ingestion/pipeline/ingestion_pipeline.py → run()
        Which internally uses:
            LocalScanner.scan()  →  yields file paths from data/incoming/
            ThreadPoolExecutor   →  parallel Worker.process() calls
        """
        from app.ingestion.pipeline.ingestion_pipeline import IngestionPipeline  # YOUR file

        all_chunks = IngestionPipeline.run()
        return {
            "total_chunks_created": len(all_chunks),
            "message": f"Full pipeline run complete. {len(all_chunks)} chunks indexed.",
        }

    @staticmethod
    def validate_extension(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    @staticmethod
    def get_allowed_extensions() -> list:
        return sorted(ALLOWED_EXTENSIONS)