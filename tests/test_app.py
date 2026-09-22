"""TestClient-based tests for backend.app's HTTP surface.

All external dependencies (OpenAI, Neo4j, FAISS, the real RAG pipeline, real
ingestion, and the SQLite document store) are mocked or replaced so these
tests run fast, offline, and deterministically.
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------
def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_readiness_check_success(client, mocker):
    """/ready should report ready=True when the (mocked) graph query succeeds."""
    mock_graph = mocker.patch("backend.config.graph")
    mock_graph.query.return_value = [{"ok": 1}]

    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["checks"]["neo4j"] is True
    assert body["checks"]["openai_config"] is True


def test_readiness_check_neo4j_down(client, mocker):
    """/ready should report 503 when the (mocked) graph query raises."""
    mock_graph = mocker.patch("backend.config.graph")
    mock_graph.query.side_effect = RuntimeError("connection refused")

    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["checks"]["neo4j"] is False


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------
def test_query_requires_api_key(client):
    resp = client.post("/api/v1/query", json={"query": "What is in the doc?"})
    assert resp.status_code == 401


def test_query_with_wrong_api_key_rejected(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is in the doc?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_query_with_valid_api_key_accepted(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.answer_query",
        return_value={
            "answer": "The document says X.",
            "guardrail_status": "PASSED",
            "retrieval_metadata": {"vector_chunks_present": True, "graph_triplets": ""},
        },
    )
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is in the doc?"},
        headers=api_key_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The document says X."
    assert body["guardrail_status"] == "PASSED"


def test_documents_list_requires_api_key(client):
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 401


def test_documents_list_with_api_key(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.document_store.list_documents",
        return_value=[
            {
                "doc_id": "abc123",
                "filename": "sample.pdf",
                "status": "ready",
                "chunks_indexed": 5,
                "graph_documents_extracted": 2,
                "error": None,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        ],
    )
    resp = client.get("/api/v1/documents", headers=api_key_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["doc_id"] == "abc123"


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------
def test_query_empty_string_rejected(client, api_key_headers):
    resp = client.post(
        "/api/v1/query", json={"query": "   "}, headers=api_key_headers
    )
    assert resp.status_code == 400


def test_query_failure_returns_500_without_leaking_details(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.answer_query", side_effect=RuntimeError("boom, secret details")
    )
    resp = client.post(
        "/api/v1/query", json={"query": "hello"}, headers=api_key_headers
    )
    assert resp.status_code == 500
    assert "boom" not in resp.text
    assert resp.json()["detail"] == "Failed to process query. Please try again later."


# ---------------------------------------------------------------------------
# Document deletion
# ---------------------------------------------------------------------------
def test_delete_document_not_found(client, api_key_headers, mocker):
    mocker.patch("backend.app.document_store.get_document", return_value=None)
    resp = client.delete("/api/v1/documents/does-not-exist", headers=api_key_headers)
    assert resp.status_code == 404


def test_delete_document_success(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.document_store.get_document",
        return_value={"doc_id": "abc123", "filename": "sample.pdf", "status": "ready"},
    )
    mock_delete_faiss = mocker.patch(
        "backend.app.delete_from_faiss", return_value=3
    )
    mock_delete_neo4j = mocker.patch("backend.app.delete_from_neo4j")
    mock_delete_record = mocker.patch(
        "backend.app.document_store.delete_document_record"
    )

    resp = client.delete("/api/v1/documents/abc123", headers=api_key_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["vectors_removed"] == 3
    mock_delete_faiss.assert_called_once_with("abc123")
    mock_delete_neo4j.assert_called_once_with("abc123")
    mock_delete_record.assert_called_once_with("abc123")


def test_delete_document_failure_returns_500(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.document_store.get_document",
        return_value={"doc_id": "abc123", "filename": "sample.pdf", "status": "ready"},
    )
    mocker.patch(
        "backend.app.delete_from_faiss", side_effect=RuntimeError("faiss exploded")
    )
    resp = client.delete("/api/v1/documents/abc123", headers=api_key_headers)
    assert resp.status_code == 500
    assert "faiss exploded" not in resp.text


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_upload_rejects_non_pdf(client, api_key_headers):
    resp = client.post(
        "/api/v1/upload",
        headers=api_key_headers,
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_upload_rejects_oversized_file(client, api_key_headers, monkeypatch):
    monkeypatch.setattr(app_module.settings, "MAX_UPLOAD_MB", 0)
    resp = client.post(
        "/api/v1/upload",
        headers=api_key_headers,
        files={
            "file": (
                "big.pdf",
                io.BytesIO(b"%PDF-1.4 " + b"0" * 100),
                "application/pdf",
            )
        },
    )
    assert resp.status_code == 413


def test_upload_requires_api_key(client):
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 401


def test_upload_success(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.ingest_pdf",
        return_value={"chunks_indexed": 2, "graph_documents_extracted": 1, "status": "success"},
    )
    resp = client.post(
        "/api/v1/upload",
        headers=api_key_headers,
        files={
            "file": (
                "sample.pdf",
                io.BytesIO(b"%PDF-1.4 fake pdf content"),
                "application/pdf",
            )
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["details"]["chunks_indexed"] == 2


def test_upload_ingestion_failure_returns_500(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.ingest_pdf", side_effect=ValueError("could not parse pdf")
    )
    resp = client.post(
        "/api/v1/upload",
        headers=api_key_headers,
        files={
            "file": (
                "sample.pdf",
                io.BytesIO(b"%PDF-1.4 fake pdf content"),
                "application/pdf",
            )
        },
    )
    assert resp.status_code == 500
    assert "could not parse pdf" not in resp.text


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------
def test_legacy_query_alias_requires_api_key(client):
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 401


def test_legacy_query_alias_works_with_api_key(client, api_key_headers, mocker):
    mocker.patch(
        "backend.app.answer_query",
        return_value={"answer": "ok", "guardrail_status": "PASSED"},
    )
    resp = client.post("/query", json={"query": "hi"}, headers=api_key_headers)
    assert resp.status_code == 200
