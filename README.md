# Hybrid RAG — FAISS + Neo4j AuraDB

A production-hardened hybrid retrieval-augmented generation (RAG) application
combining a **FAISS** dense vector store with a **Neo4j AuraDB** knowledge
graph, served via a **FastAPI** backend and a **Streamlit** chat frontend.

- Dense retrieval (FAISS) + graph retrieval (Neo4j 1-hop relationships),
  combined into a single grounded generation step (OpenAI `gpt-4o-mini`).
- Security-first input guardrail (deterministic jailbreak/prompt-injection
  detection) and an output safety guardrail — see [Guardrails](#guardrails).
- Multi-document lifecycle: upload, list, and delete individual documents
  (tracked in a local SQLite metadata store, tagged into both FAISS and
  Neo4j so deletion is precise).
- API-key auth, per-route rate limiting, structured JSON logging, Prometheus
  metrics, liveness/readiness probes.
- Fully covered by an offline/mocked `pytest` suite and a GitHub Actions CI
  workflow.
- Runs either directly with Python/`venv` (no Docker required) or via
  Docker Compose.

---

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [1. Configure environment variables](#1-configure-environment-variables)
- [Running without Docker (local Python)](#running-without-docker-local-python)
- [Running with Docker](#running-with-docker)
- [Using the app](#using-the-app)
- [API reference](#api-reference)
- [Guardrails](#guardrails)
- [Running tests](#running-tests)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)

---

## Architecture

```
PDF Upload
    |
    v
PDF Loader
    |
    v
Text Chunking
    |
    +--------------------------+
    |                          |
    v                          v
OpenAI Embeddings       Entity / Relationship
    |                      Extraction
    v                          |
  FAISS                        v
                         Neo4j Graph
    |                          |
    +-------------+------------+
                  |
                  v
             User Question
                  |
                  v
           Input Guardrail
                  |
        +---------+---------+
        |                   |
        v                   v
   FAISS Search        Neo4j Search
        |                   |
        +---------+---------+
                  |
                  v
           Hybrid Context
                  |
                  v
          Context Guardrail
                  |
                  v
             OpenAI LLM
                  |
                  v
          Output Guardrail
                  |
                  v
                Answer
                  |
                  v
                 Evals
        +---------+---------+
        |         |         |
        v         v         v
    Retrieval  Faithful   Answer
    Relevance    ness     Relevance
```

- **Ingestion** (`POST /api/v1/upload`): a PDF is split into chunks, embedded
  and stored in FAISS, and passed through an LLM graph transformer to extract
  entities/relationships into Neo4j. Every chunk is tagged with a `doc_id` so
  it can later be looked up or deleted precisely from both stores.
- **Input Guardrail**: a fast, deterministic regex check for prompt-injection
  / jailbreak attempts, with a narrow LLM fallback for novel phrasing. It
  never judges topical scope — see [Guardrails](#guardrails).
- **Hybrid retrieval**: the query is matched against FAISS (top-k chunks) and
  Neo4j (1-hop entity relationships) in parallel, producing a combined
  "Hybrid Context".
- **Context Guardrail**: sanitizes the retrieved context before it reaches
  the LLM — strips any line matching a known injection pattern (defending
  against a poisoned/malicious ingested document trying to hijack the
  assistant) and flags when nothing relevant was retrieved. Never blocks the
  query; only removes offending lines.
- **Generation**: the sanitized hybrid context + question are sent to the
  LLM to produce a grounded answer.
- **Output Guardrail**: a final safety check limited to genuinely harmful or
  policy-violating content before the answer is returned.
- **Evals**: for every answer that passes both guardrails, an LLM-judge
  scores **Retrieval Relevance**, **Faithfulness**, and **Answer Relevance**
  (0.0–1.0 each). Purely advisory/observational — never blocks or alters the
  response — surfaced in `retrieval_metadata.evals` and visible in the
  frontend's "Show technical details" panel. Toggle with `ENABLE_EVALS`.

## Prerequisites

- **Python 3.11+** (a `venv` already exists in this repo at `venv\`)
- An **OpenAI API key**
- A **Neo4j AuraDB** instance (free tier works) — URI, username, password
- *(Optional, only if you want containers)* Docker Desktop with Compose v2

---

## 1. Configure environment variables

All configuration lives in a single `.env` file at the repo root, validated
at startup via `pydantic-settings` (the app **fails fast with a clear error**
if a required variable is missing, rather than failing later at query time).

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at minimum:

| Variable         | Required | Description                                             |
|------------------|:--------:|----------------------------------------------------------|
| `OPENAI_API_KEY` | ✅       | Your OpenAI API key                                       |
| `NEO4J_URI`      | ✅       | e.g. `neo4j+s://<instance-id>.databases.neo4j.io`         |
| `NEO4J_USERNAME` | ✅       | Neo4j AuraDB username                                     |
| `NEO4J_PASSWORD` | ✅       | Neo4j AuraDB password                                     |
| `API_KEY`        | recommended | Shared secret required as the `X-API-Key` header on every `/api/v1/*` route. Leave empty to disable auth (local dev only). |

Everything else (chunking, retrieval `k`, rate limits, CORS origins, log
level, upload size limit, etc.) has a working default — see
[Configuration reference](#configuration-reference) or `.env.example` for
the full annotated list.

> ⚠️ Never commit your real `.env`. Use a distinct, random value for `API_KEY`
> — do **not** reuse your `OPENAI_API_KEY` as the app's `API_KEY`.

---

## Running without Docker (local Python)

This is the fastest way to iterate locally. Two terminals, same `venv`.

### Terminal 1 — start the backend

```powershell
cd C:\Users\T0232GD\Desktop\GenAI\hybrid-rag
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

If `venv` doesn't yet have the backend dependencies installed:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Verify it's up:

```powershell
curl http://127.0.0.1:8000/health   # {"status": "healthy"}
curl http://127.0.0.1:8000/ready    # {"ready": true, "checks": {...}}  (confirms Neo4j is reachable)
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

> By default `uvicorn` does **not** auto-reload on code changes — restart the
> process after editing backend code. For local dev convenience only, you can
> add `--reload` to the command above (not recommended for production).

### Terminal 2 — start the frontend

```powershell
cd C:\Users\T0232GD\Desktop\GenAI\hybrid-rag
.\venv\Scripts\python.exe -m pip install -r frontend\requirements.txt   # first time only
.\venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

This opens the UI at `http://localhost:8501`.

By default the frontend looks for the backend at `http://127.0.0.1:8000`. If
you've set an `API_KEY` in `.env`, either paste it into the sidebar's
"Backend API Key" field once the UI loads, or pre-fill it by setting an env
var before launching Streamlit:

```powershell
$env:BACKEND_API_KEY = "<same value as API_KEY in .env>"
$env:BACKEND_URL      = "http://127.0.0.1:8000"   # only needed if backend isn't on the default host/port
.\venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

### Stopping

Press `Ctrl+C` in each terminal.

---

## Running with Docker

The repo ships with a `docker-compose.yml` that builds and runs both the
FastAPI backend and the Streamlit frontend.

### 1. Build and start the stack

```powershell
docker compose up --build
```

This builds:
- `backend` — FastAPI app on **http://localhost:8000**
- `frontend` — Streamlit UI on **http://localhost:8501**

The frontend automatically talks to the backend via the internal Docker
network (`BACKEND_URL=http://backend:8000`); the frontend container only
starts once the backend's healthcheck reports healthy.

### 2. Exposed ports

| Service  | Container port | Host port | URL                       |
|----------|-----------------|-----------|---------------------------|
| backend  | 8000            | 8000      | http://localhost:8000     |
| frontend | 8501            | 8501      | http://localhost:8501     |

### 3. Check health

```powershell
curl http://localhost:8000/health   # liveness: process is up
curl http://localhost:8000/ready    # readiness: verifies Neo4j connectivity
```

### Persistence

- The FAISS index directory (`backend/faiss_index`) is bind-mounted so
  ingested vectors persist across container restarts/rebuilds.
- The SQLite document-tracking DB lives at
  `backend/faiss_index/doc_store.sqlite3` — inside the same mounted
  directory — so it persists automatically without a separate mount. (A
  file-level bind mount was used previously but is fragile: if the file
  doesn't already exist on the host, Docker silently creates a directory at
  that path instead, which breaks SQLite. Nesting it inside the
  already-mounted directory avoids that pitfall entirely.)

### Stopping the stack

```powershell
docker compose down
```

---

## Using the app

1. Open the Streamlit UI (`http://localhost:8501`).
2. If auth is enabled, paste your `API_KEY` into the sidebar's "Backend API
   Key" field.
3. Go to the **📁 Documents** tab, upload a PDF, and click **Ingest
   Document**. This indexes it into both FAISS and Neo4j.
4. Switch to the **💬 Chat** tab and ask questions about the document.
   Each answer shows a small status badge (`✅ Verified`, `⚠️ Needs review`,
   `🚫 Blocked`); enable **"Show technical details"** in the sidebar to see
   the full guardrail reasoning and retrieval metadata per answer.
5. Back in **📁 Documents**, you can refresh the list or delete a document —
   deletion removes its chunks from FAISS, its nodes from Neo4j, and its
   tracking record.

---

## API reference

All `/api/v1/*` routes require the `X-API-Key` header when `API_KEY` is set
in `.env` (if `API_KEY` is empty, auth is disabled).

| Method | Path                          | Auth | Description                                      |
|--------|-------------------------------|:----:|---------------------------------------------------|
| GET    | `/health`                     | —    | Liveness probe                                     |
| GET    | `/ready`                      | —    | Readiness probe (checks Neo4j connectivity)        |
| GET    | `/metrics`                    | —    | Prometheus metrics                                 |
| POST   | `/api/v1/upload`              | ✅   | Upload + ingest a PDF (multipart, field `file`)    |
| GET    | `/api/v1/documents`           | ✅   | List ingested documents and their status           |
| DELETE | `/api/v1/documents/{doc_id}`  | ✅   | Delete a document from FAISS + Neo4j + metadata    |
| POST   | `/api/v1/query`               | ✅   | Ask a question (`{"query": "..."}`)                |

Legacy unversioned aliases (`/upload`, `/query`) remain available for
backward compatibility but are deprecated in favor of `/api/v1/*`.

Interactive Swagger docs: `http://127.0.0.1:8000/docs` (or `:8000` on
whichever host you're running the backend).

---

## Guardrails

The app deliberately keeps guardrails narrow and deterministic where
possible, to avoid blocking legitimate questions. There are three guardrail
stages plus an advisory evals layer, matching the pipeline in
[Architecture](#architecture):

- **Input guardrail** — blocks only genuine prompt-injection/jailbreak
  attempts, using a fast regex check first (e.g. "ignore previous
  instructions", "reveal your system prompt") with a narrowly-scoped LLM
  fallback for novel phrasing of the same attack. It never tries to judge
  whether a question is "in scope" of the uploaded documents — that
  determination happens naturally after retrieval.
- **Retrieval-first scope handling** — every non-malicious query always
  reaches FAISS + Neo4j retrieval. If nothing relevant is found, the
  generation prompt itself (grounded strictly in retrieved context) tells
  the user honestly that the documents don't contain the answer, instead of
  a blanket pre-emptive rejection.
- **Context guardrail** — sanitizes the retrieved (hybrid) context before it
  reaches the LLM. This is a deterministic, regex-based check (reusing the
  same jailbreak patterns as the input guardrail) that strips any line
  matching a known injection pattern — defending against *indirect prompt
  injection*, where a malicious/poisoned ingested document contains text
  designed to hijack the assistant. Only the offending line is removed, not
  the whole chunk, and the query is never blocked outright. It also flags
  (informationally, via `retrieval_metadata.context_guardrail_status`) when
  nothing relevant was retrieved at all (`OK` / `SANITIZED` / `EMPTY`).
- **Output guardrail** — a final safety check limited to genuinely harmful,
  dangerous, or policy-violating content; it does not re-litigate
  groundedness (already enforced by the generation prompt) to avoid
  discarding correct answers.
- **Evals (advisory, non-blocking)** — after a successful, guardrail-passed
  answer, an LLM judge scores three RAGAS-style metrics on a 0.0–1.0 scale
  and returns them in `retrieval_metadata.evals`:
  - `retrieval_relevance` — how relevant the retrieved context was to the question.
  - `faithfulness` — how well the answer's claims are supported by the retrieved context (no hallucination).
  - `answer_relevance` — how directly the answer addresses what was asked.

  These scores are purely observational (for monitoring answer quality over
  time) and never change what's returned to the user. Disable with
  `ENABLE_EVALS=false` in `.env` to save one LLM call per query.

---

## Running tests

```powershell
cd C:\Users\T0232GD\Desktop\GenAI\hybrid-rag
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\venv\Scripts\python.exe -m pytest tests\ -v
```

The suite (`tests/`) runs fully offline — OpenAI, Neo4j, and FAISS are all
mocked, so no API keys or network access are required to run it. The same
suite runs automatically in CI on every push/PR via
`.github/workflows/ci.yml`.

---

## Configuration reference

See `.env.example` for the full annotated list. Highlights:

| Variable              | Default                                   | Purpose                                    |
|------------------------|--------------------------------------------|---------------------------------------------|
| `OPENAI_MODEL`         | `gpt-4o-mini`                               | Chat/completion model                       |
| `EMBEDDING_MODEL`      | `text-embedding-3-small`                    | Embedding model for FAISS                   |
| `NEO4J_DATABASE`       | `neo4j`                                     | Logical database name                       |
| `API_KEY`              | *(empty = auth disabled)*                   | Required `X-API-Key` header value           |
| `ALLOWED_ORIGINS`      | `http://localhost:8501,http://127.0.0.1:8501` | CORS allow-list                           |
| `RATE_LIMIT_QUERY`     | `20/minute`                                  | Rate limit for `/api/v1/query`              |
| `RATE_LIMIT_UPLOAD`    | `5/minute`                                   | Rate limit for `/api/v1/upload`             |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200`                       | PDF text splitting                          |
| `RETRIEVAL_K`          | `3`                                          | Top-k vector chunks per query               |
| `GRAPH_RESULT_LIMIT`   | `20`                                         | Max graph triplets per query                |
| `MAX_UPLOAD_MB`        | `25`                                         | Max accepted PDF size                       |
| `ENABLE_EVALS`         | `true`                                       | Toggle the advisory evals LLM call (see [Guardrails](#guardrails)) |
| `LOG_LEVEL`            | `INFO`                                       | Python logging level (JSON structured logs) |

---

## Troubleshooting

- **"Missing or invalid API key" (401)** — the frontend's "Backend API Key"
  field (or `BACKEND_API_KEY` env var) must exactly match `API_KEY` in the
  backend's `.env`. Set `API_KEY=` (empty) in `.env` to disable auth for
  local dev.
- **A backend code change doesn't seem to take effect** — restart the
  `uvicorn` process; it does not auto-reload by default (see note above).
  Also double-check nothing else is already bound to port 8000
  (`Get-NetTCPConnection -LocalPort 8000` on Windows) serving stale code.
- **`/ready` returns `"neo4j": false`** — check `NEO4J_URI` /
  `NEO4J_USERNAME` / `NEO4J_PASSWORD` in `.env` and that your AuraDB instance
  is running (free-tier instances can pause after inactivity).
- **Answers say "the documents do not contain the answer"** — this is
  expected when nothing relevant was ingested yet, or the question isn't
  covered by the uploaded PDF(s); upload a relevant document first.
- **Upload fails with 413** — the PDF exceeds `MAX_UPLOAD_MB` (default 25MB);
  raise it in `.env` and restart the backend.

---

## Project structure

```
hybrid-rag/
├── backend/
│   ├── app.py              # FastAPI app: routes, auth, rate limiting, error handling
│   ├── config.py           # pydantic-settings validated configuration
│   ├── auth.py              # X-API-Key dependency
│   ├── logging_config.py   # Structured JSON logging
│   ├── document_store.py   # SQLite-backed document metadata store
│   ├── ingestion.py        # PDF -> chunks -> FAISS + Neo4j, with retries
│   ├── rag.py               # Guardrails, hybrid retrieval, generation
│   ├── faiss_index/        # Local FAISS index + doc_store.sqlite3 (persisted)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py    # Tabbed chat UI (Chat / Documents)
│   ├── requirements.txt
│   └── Dockerfile
├── tests/                  # Offline/mocked pytest suite
├── .github/workflows/ci.yml
├── docker-compose.yml
├── .env.example
└── README.md
```
