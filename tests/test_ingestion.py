"""Unit tests for backend.ingestion.

PyPDFLoader and the real FAISS/Neo4j clients are mocked so these tests never
touch the filesystem-based PDF parser, OpenAI embeddings, or a real vector
index/graph database.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from tenacity import stop_after_attempt, wait_exponential

from backend import ingestion


# ---------------------------------------------------------------------------
# extract_and_split_pdf
# ---------------------------------------------------------------------------
def test_extract_and_split_pdf_tags_chunks_with_doc_id(mocker):
    raw_docs = [
        Document(page_content="Page one content. " * 50, metadata={"page": 0}),
        Document(page_content="Page two content. " * 50, metadata={"page": 1}),
    ]
    mock_loader_cls = mocker.patch("backend.ingestion.PyPDFLoader")
    mock_loader_cls.return_value.load.return_value = raw_docs

    chunks = ingestion.extract_and_split_pdf("C:\\fake\\path\\report.pdf", "doc-123")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == "doc-123"
        assert chunk.metadata["source_filename"] == "report.pdf"
    mock_loader_cls.assert_called_once_with("C:\\fake\\path\\report.pdf")


def test_extract_and_split_pdf_no_text_returns_empty_list(mocker):
    mock_loader_cls = mocker.patch("backend.ingestion.PyPDFLoader")
    mock_loader_cls.return_value.load.return_value = []

    chunks = ingestion.extract_and_split_pdf("C:\\fake\\path\\empty.pdf", "doc-456")
    assert chunks == []


# ---------------------------------------------------------------------------
# delete_from_faiss
# ---------------------------------------------------------------------------
def test_delete_from_faiss_no_index_returns_zero(mocker, tmp_path):
    fake_index_dir = tmp_path / "faiss_index"
    mocker.patch("backend.ingestion.FAISS_INDEX_DIR", fake_index_dir)

    removed = ingestion.delete_from_faiss("doc-123")
    assert removed == 0


def test_delete_from_faiss_removes_matching_doc_ids(mocker, tmp_path):
    fake_index_dir = tmp_path / "faiss_index"
    fake_index_dir.mkdir()
    (fake_index_dir / "index.faiss").write_bytes(b"fake-index-bytes")
    mocker.patch("backend.ingestion.FAISS_INDEX_DIR", fake_index_dir)

    fake_doc_matching = MagicMock()
    fake_doc_matching.metadata = {"doc_id": "doc-123"}
    fake_doc_other = MagicMock()
    fake_doc_other.metadata = {"doc_id": "doc-999"}

    fake_vector_store = MagicMock()
    fake_vector_store.docstore._dict = {
        "vec-1": fake_doc_matching,
        "vec-2": fake_doc_other,
    }
    mock_faiss_cls = mocker.patch("backend.ingestion.FAISS")
    mock_faiss_cls.load_local.return_value = fake_vector_store

    removed = ingestion.delete_from_faiss("doc-123")

    assert removed == 1
    fake_vector_store.delete.assert_called_once_with(["vec-1"])
    fake_vector_store.save_local.assert_called_once_with(str(fake_index_dir))


def test_delete_from_faiss_no_matches_does_not_save(mocker, tmp_path):
    fake_index_dir = tmp_path / "faiss_index"
    fake_index_dir.mkdir()
    (fake_index_dir / "index.faiss").write_bytes(b"fake-index-bytes")
    mocker.patch("backend.ingestion.FAISS_INDEX_DIR", fake_index_dir)

    fake_doc_other = MagicMock()
    fake_doc_other.metadata = {"doc_id": "doc-999"}
    fake_vector_store = MagicMock()
    fake_vector_store.docstore._dict = {"vec-2": fake_doc_other}
    mock_faiss_cls = mocker.patch("backend.ingestion.FAISS")
    mock_faiss_cls.load_local.return_value = fake_vector_store

    removed = ingestion.delete_from_faiss("doc-123")

    assert removed == 0
    fake_vector_store.delete.assert_not_called()
    fake_vector_store.save_local.assert_not_called()


# ---------------------------------------------------------------------------
# store_in_faiss retry wiring
# ---------------------------------------------------------------------------
def test_store_in_faiss_is_wrapped_with_tenacity_retry():
    """Confirms the tenacity retry decorator is applied with the expected policy."""
    retry_obj = ingestion.store_in_faiss.retry
    assert hasattr(retry_obj.stop, "max_attempt_number")
    assert retry_obj.stop.max_attempt_number == 3


def test_store_in_faiss_success_path_creates_new_index(mocker, tmp_path):
    fake_index_dir = tmp_path / "faiss_index"
    mocker.patch("backend.ingestion.FAISS_INDEX_DIR", fake_index_dir)

    fake_vector_store = MagicMock()
    mock_faiss_cls = mocker.patch("backend.ingestion.FAISS")
    mock_faiss_cls.from_documents.return_value = fake_vector_store

    chunks = [Document(page_content="hello", metadata={"doc_id": "doc-1"})]
    ingestion.store_in_faiss(chunks)

    mock_faiss_cls.from_documents.assert_called_once()
    fake_vector_store.save_local.assert_called_once_with(str(fake_index_dir))
    assert fake_index_dir.exists()


def test_store_in_faiss_retries_on_transient_error_then_succeeds(mocker, tmp_path):
    """A transient (retryable) failure followed by success should not raise."""
    fake_index_dir = tmp_path / "faiss_index"
    mocker.patch("backend.ingestion.FAISS_INDEX_DIR", fake_index_dir)
    # Avoid real sleeping between retry attempts.
    mocker.patch("backend.ingestion.wait_exponential", return_value=wait_exponential(min=0, max=0))

    fake_vector_store = MagicMock()
    mock_faiss_cls = mocker.patch("backend.ingestion.FAISS")
    mock_faiss_cls.from_documents.side_effect = [
        ConnectionError("transient network blip"),
        fake_vector_store,
    ]

    chunks = [Document(page_content="hello", metadata={"doc_id": "doc-1"})]
    # The retry decorator was already bound at module import time with the
    # original wait_exponential, so we exercise it via a freshly-decorated
    # copy that uses a zero-wait policy to keep the test fast.
    from tenacity import retry, retry_if_exception_type

    unretried = ingestion.store_in_faiss.__wrapped__
    fast_retry_fn = retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    )(unretried)

    fast_retry_fn(chunks)
    assert mock_faiss_cls.from_documents.call_count == 2
    fake_vector_store.save_local.assert_called_once_with(str(fake_index_dir))


# ---------------------------------------------------------------------------
# store_in_neo4j
# ---------------------------------------------------------------------------
def test_store_in_neo4j_success_path(mocker):
    mock_transformer_cls = mocker.patch("backend.ingestion.LLMGraphTransformer")
    fake_graph_docs = [MagicMock(), MagicMock()]
    mock_transformer_cls.return_value.convert_to_graph_documents.return_value = (
        fake_graph_docs
    )
    mock_graph = mocker.patch("backend.ingestion.graph")

    chunks = [Document(page_content="hello", metadata={"doc_id": "doc-1"})]
    result = ingestion.store_in_neo4j(chunks)

    assert result == 2
    mock_graph.add_graph_documents.assert_called_once_with(
        fake_graph_docs, baseEntityLabel=True, include_source=True
    )


# ---------------------------------------------------------------------------
# delete_from_neo4j
# ---------------------------------------------------------------------------
def test_delete_from_neo4j_calls_graph_query_with_doc_id(mocker):
    mock_graph = mocker.patch("backend.ingestion.graph")
    ingestion.delete_from_neo4j("doc-123")
    mock_graph.query.assert_called_once()
    args, kwargs = mock_graph.query.call_args
    assert kwargs["params"] == {"doc_id": "doc-123"}


# ---------------------------------------------------------------------------
# ingest_pdf orchestration
# ---------------------------------------------------------------------------
def test_ingest_pdf_raises_when_no_chunks(mocker):
    mocker.patch("backend.ingestion.extract_and_split_pdf", return_value=[])
    with pytest.raises(ValueError):
        ingestion.ingest_pdf("C:\\fake\\empty.pdf", "doc-1")


def test_ingest_pdf_happy_path(mocker):
    chunks = [Document(page_content="hello", metadata={"doc_id": "doc-1"})]
    mocker.patch("backend.ingestion.extract_and_split_pdf", return_value=chunks)
    mock_store_faiss = mocker.patch("backend.ingestion.store_in_faiss")
    mock_store_neo4j = mocker.patch(
        "backend.ingestion.store_in_neo4j", return_value=4
    )

    result = ingestion.ingest_pdf("C:\\fake\\report.pdf", "doc-1")

    assert result == {
        "chunks_indexed": 1,
        "graph_documents_extracted": 4,
        "status": "success",
    }
    mock_store_faiss.assert_called_once_with(chunks)
    mock_store_neo4j.assert_called_once_with(chunks)
