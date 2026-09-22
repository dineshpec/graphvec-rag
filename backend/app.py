import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.auth import require_api_key
from backend.config import settings
from backend.logging_config import configure_logging, get_logger, new_request_id, request_id_ctx
from backend import document_store
from backend.ingestion import delete_from_faiss, delete_from_neo4j, ingest_pdf
from backend.rag import answer_query

configure_logging()
logger = get_logger(__name__)
document_store.init_store()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Hybrid RAG API (FAISS + Neo4j AuraDB)",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS only for explicitly configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

UPLOAD_DIR = Path(__file__).resolve().parent / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attaches a request id to every request for log correlation and timing."""
    req_id = new_request_id()
    token = request_id_ctx.set(req_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "request_completed",
        extra={
            "request_id": req_id,
            "path": request.url.path,
            "method": request.method,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Prevents raw stack traces / internal details from leaking to clients."""
    logger.exception("unhandled_exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    guardrail_status: str
    reason: str | None = None
    retrieval_metadata: dict | None = None


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunks_indexed: int
    graph_documents_extracted: int
    error: Optional[str] = None
    created_at: str
    updated_at: str


@app.get("/health")
def health_check():
    """Liveness probe: process is up and able to respond."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness probe: verifies downstream dependencies are reachable."""
    checks = {"neo4j": False, "openai_config": bool(settings.OPENAI_API_KEY)}
    try:
        from backend.config import graph as neo4j_graph
        neo4j_graph.query("RETURN 1 AS ok")
        checks["neo4j"] = True
    except Exception as e:
        logger.warning("readiness_neo4j_failed", extra={"error": str(e)})

    ready = all(checks.values())
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content={"ready": ready, "checks": checks})


@app.post("/api/v1/upload", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Receives a PDF, writes to disk, and executes dual ingestion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Enforce max upload size to avoid unbounded disk/memory usage.
    # Read settings.MAX_UPLOAD_MB dynamically (not cached at import time) so
    # runtime config changes take effect without an app restart.
    contents = await file.read()
    max_upload_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB upload limit.",
        )

    doc_id = document_store.create_document(file.filename)
    temp_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)

        result = ingest_pdf(str(temp_path), doc_id)
        document_store.mark_success(
            doc_id, result["chunks_indexed"], result["graph_documents_extracted"]
        )
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "success",
            "details": result
        }
    except Exception as e:
        document_store.mark_failed(doc_id, str(e))
        logger.exception("ingestion_failed", extra={"doc_id": doc_id})
        raise HTTPException(status_code=500, detail="Ingestion failed. Please check the file and try again.")
    finally:
        # Clean up temporary upload file to conserve disk space
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/v1/documents", response_model=list[DocumentResponse], dependencies=[Depends(require_api_key)])
def list_documents():
    """Lists all ingested documents and their status."""
    return document_store.list_documents()


@app.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(require_api_key)])
def delete_document(doc_id: str):
    """Deletes a document's chunks from FAISS, Neo4j, and the metadata store."""
    doc = document_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        removed_vectors = delete_from_faiss(doc_id)
        delete_from_neo4j(doc_id)
        document_store.delete_document_record(doc_id)
        return {"doc_id": doc_id, "status": "deleted", "vectors_removed": removed_vectors}
    except Exception:
        logger.exception("deletion_failed", extra={"doc_id": doc_id})
        raise HTTPException(status_code=500, detail="Failed to delete document.")


@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.RATE_LIMIT_QUERY)
def handle_query(request: Request, payload: QueryRequest):
    """Executes input guardrails, hybrid retrieval, and generation."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        result = answer_query(payload.query)
        return QueryResponse(
            answer=result["answer"],
            guardrail_status=result.get("guardrail_status", "UNKNOWN"),
            reason=result.get("reason"),
            retrieval_metadata=result.get("retrieval_metadata")
        )
    except Exception:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail="Failed to process query. Please try again later.")


# --- Backward-compatible unversioned aliases (deprecated) ---
@app.post("/upload", dependencies=[Depends(require_api_key)], include_in_schema=False)
async def upload_pdf_legacy(request: Request, file: UploadFile = File(...)):
    return await upload_pdf(request, file)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)], include_in_schema=False)
def handle_query_legacy(request: Request, payload: QueryRequest):
    return handle_query(request, payload)
