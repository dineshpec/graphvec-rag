"""Shared pytest fixtures.

Critically, this module sets required environment variables and patches
``langchain_neo4j.Neo4jGraph`` *before* ``backend.config`` (and anything that
imports it, such as ``backend.app``) is ever imported. ``backend.config``
builds a module-level ``Settings`` singleton and a ``Neo4jGraph`` instance at
import time, and the real ``Neo4jGraph.__init__`` calls
``driver.verify_connectivity()`` which would otherwise attempt a real network
connection to Neo4j AuraDB. Doing this at conftest module level (rather than
inside a fixture) guarantees it runs before pytest imports any test module in
this directory, since conftest.py is always loaded first during collection.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- 1. Force safe, fake credentials before any backend module is imported ---
# Using direct assignment (not setdefault) ensures these test values win over
# anything picked up from a real environment, without ever touching the
# repository's .env file on disk.
os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-tests"
os.environ["NEO4J_URI"] = "neo4j+s://test-instance.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "test-user"
os.environ["NEO4J_PASSWORD"] = "test-password"
os.environ["NEO4J_DATABASE"] = "neo4j"
os.environ["API_KEY"] = "test-api-key"
os.environ["ENVIRONMENT"] = "test"

# --- 2. Prevent Neo4jGraph from opening a real connection at import time ---
import langchain_neo4j  # noqa: E402


class _FakeNeo4jGraph:
    """Stand-in for langchain_neo4j.Neo4jGraph that never touches the network."""

    def __init__(self, *args, **kwargs):
        self.query = MagicMock(return_value=[])
        self.add_graph_documents = MagicMock()
        self.refresh_schema = MagicMock()


langchain_neo4j.Neo4jGraph = _FakeNeo4jGraph

# --- 3. Redirect the SQLite document store to a throwaway file under tests/ ---
# document_store.py binds DOC_STORE_DB at import time from backend.config, so
# we patch it on the (not-yet-imported) document_store module's namespace
# before backend.app triggers document_store.init_store() as a side effect of
# import.
TEST_DATA_DIR = Path(__file__).resolve().parent / "_test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_DOC_STORE_DB = TEST_DATA_DIR / "test_doc_store.sqlite3"

import backend.document_store as document_store  # noqa: E402

document_store.DOC_STORE_DB = TEST_DOC_STORE_DB


@pytest.fixture(autouse=True)
def _clean_document_store():
    """Ensures each test starts with a fresh, empty document store table."""
    if TEST_DOC_STORE_DB.exists():
        TEST_DOC_STORE_DB.unlink()
    document_store.init_store()
    yield
    if TEST_DOC_STORE_DB.exists():
        TEST_DOC_STORE_DB.unlink()


@pytest.fixture
def api_key_headers():
    """Valid auth headers matching the test API_KEY set above."""
    return {"X-API-Key": os.environ["API_KEY"]}
