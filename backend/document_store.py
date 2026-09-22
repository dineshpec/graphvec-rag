"""SQLite-backed metadata store for ingested documents.

Enables multi-document management: listing what has been ingested, and
deleting a document's chunks from both FAISS and Neo4j by tagging every
chunk with a stable ``doc_id`` at ingestion time.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from backend.config import DOC_STORE_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    chunks_indexed INTEGER DEFAULT 0,
    graph_documents_extracted INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(DOC_STORE_DB))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_store() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def create_document(filename: str) -> str:
    """Registers a new document as 'processing' and returns its doc_id."""
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, status, created_at, updated_at) "
            "VALUES (?, ?, 'processing', ?, ?)",
            (doc_id, filename, now, now),
        )
    return doc_id


def mark_success(doc_id: str, chunks_indexed: int, graph_documents_extracted: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status='ready', chunks_indexed=?, "
            "graph_documents_extracted=?, updated_at=? WHERE doc_id=?",
            (chunks_indexed, graph_documents_extracted, now, doc_id),
        )


def mark_failed(doc_id: str, error: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status='failed', error=?, updated_at=? WHERE doc_id=?",
            (error, now, doc_id),
        )


def list_documents() -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_document_record(doc_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
