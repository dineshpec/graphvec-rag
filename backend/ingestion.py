import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config import embeddings, llm, graph, settings, FAISS_INDEX_DIR

logger = logging.getLogger(__name__)

# Transient errors worth retrying (network hiccups, rate limits, etc.)
_RETRYABLE_EXC = (ConnectionError, TimeoutError, OSError)


def extract_and_split_pdf(file_path: str, doc_id: str) -> List[Document]:
    """Loads a PDF and splits it into semantic chunks tagged with doc_id."""
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)

    # Tag every chunk with doc_id/source so it can be filtered/deleted later.
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_filename"] = Path(file_path).name

    return chunks


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE_EXC),
)
def store_in_faiss(chunks: List[Document]) -> None:
    """Builds or updates the FAISS vector index, retrying transient failures."""
    if FAISS_INDEX_DIR.exists() and (FAISS_INDEX_DIR / "index.faiss").exists():
        vector_store = FAISS.load_local(
            folder_path=str(FAISS_INDEX_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(str(FAISS_INDEX_DIR))


def delete_from_faiss(doc_id: str) -> int:
    """Removes all chunks tagged with doc_id from the FAISS index."""
    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        return 0

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_INDEX_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    ids_to_delete = [
        doc_uuid
        for doc_uuid, doc in vector_store.docstore._dict.items()
        if doc.metadata.get("doc_id") == doc_id
    ]
    if ids_to_delete:
        vector_store.delete(ids_to_delete)
        vector_store.save_local(str(FAISS_INDEX_DIR))
    return len(ids_to_delete)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE_EXC),
)
def store_in_neo4j(chunks: List[Document]) -> int:
    """
    Extracts entities and relationships via LLMGraphTransformer
    and commits them to Neo4j AuraDB, retrying transient failures.
    """
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[],        # Empty allows discovery of dynamic entity types
        allowed_relationships=[] # Empty allows discovery of dynamic relationship types
    )

    graph_documents = transformer.convert_to_graph_documents(chunks)

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    return len(graph_documents)


def delete_from_neo4j(doc_id: str) -> None:
    """Removes graph nodes/relationships that originated from this doc_id."""
    graph.query(
        """
        MATCH (d:Document)
        WHERE d.doc_id = $doc_id
        OPTIONAL MATCH (d)<-[:MENTIONS]-(e)
        DETACH DELETE d, e
        """,
        params={"doc_id": doc_id},
    )


def ingest_pdf(file_path: str, doc_id: str) -> dict:
    """Orchestrates PDF ingestion across FAISS and Neo4j."""
    chunks = extract_and_split_pdf(file_path, doc_id)
    if not chunks:
        raise ValueError("Could not extract readable text from the PDF.")

    # 1. Store in FAISS
    store_in_faiss(chunks)
    logger.info("faiss_indexed", extra={"doc_id": doc_id, "chunks": len(chunks)})

    # 2. Store in Neo4j
    graph_docs_count = store_in_neo4j(chunks)
    logger.info("neo4j_indexed", extra={"doc_id": doc_id, "graph_documents": graph_docs_count})

    return {
        "chunks_indexed": len(chunks),
        "graph_documents_extracted": graph_docs_count,
        "status": "success"
    }