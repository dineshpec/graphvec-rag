# syntax=docker/dockerfile:1
#
# Frontend image for the Hybrid RAG Streamlit UI.
# Build from the repo root:
#   docker build -f frontend\Dockerfile -t hybrid-rag-frontend .

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is needed for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./

# Default backend location; override at runtime (e.g. via docker-compose)
# to point at the backend service, container, or a remote deployment.
ENV BACKEND_URL=http://127.0.0.1:8000

RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

# enableXsrfProtection=false: required when running behind a reverse proxy
# (Caddy/Nginx/Traefik). Streamlit's default XSRF protection relies on a
# same-origin cookie round-trip that reverse proxies can break, causing the
# file uploader to fail with "AxiosError: Request failed with status code
# 403". The backend API itself is still protected separately via the
# X-API-Key header (see backend/auth.py) and HTTPS is enforced by the proxy,
# so this only relaxes Streamlit's own (redundant, proxy-incompatible)
# upload-form protection, not the actual application's security boundary.
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
