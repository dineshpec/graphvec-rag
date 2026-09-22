# syntax=docker/dockerfile:1
#
# Backend image for the Hybrid RAG FastAPI service.
#
# IMPORTANT: build this image with the *repository root* as the build
# context (not backend\), e.g. from C:\...\hybrid-rag:
#   docker build -f backend\Dockerfile -t hybrid-rag-backend .
# This is required because the app is imported as the package path
# `backend.app:app` (e.g. `from backend.config import settings`), so the
# `backend` package directory must exist under /app in the image.

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr, and
# keep pip from caching wheels (smaller image, faster/cleaner CI builds).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: curl is needed for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend package. The build context is the repo root so that the
# resulting layout is /app/backend/*, matching the `backend.app:app` import.
COPY backend/ ./backend/

# Ensure runtime-writable directories exist even if not mounted as volumes.
RUN mkdir -p /app/backend/faiss_index /app/backend/temp_uploads

# Create a non-root user to run the application.
RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Persist the FAISS index directory (and, inside it, the SQLite doc-tracking
# DB at faiss_index/doc_store.sqlite3) across container restarts/recreations.
# Both live under this single directory so one bind mount in
# docker-compose.yml covers both; see backend/config.py for why the sqlite
# file is nested here rather than mounted separately.
VOLUME ["/app/backend/faiss_index"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
