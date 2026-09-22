"""Centralized, validated application configuration.

Uses pydantic-settings so that missing/invalid environment variables fail
fast with a clear error at startup instead of surfacing as obscure runtime
errors later during ingestion or querying.
"""
from pathlib import Path
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph

# Resolve root directory: C:\Users\...\hybrid-rag
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Application settings sourced from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / OpenAI ---
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.0

    # --- Neo4j ---
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # --- Service ---
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    ENVIRONMENT: str = "development"  # development | staging | production
    LOG_LEVEL: str = "INFO"

    # --- Security ---
    API_KEY: str = ""  # If set, required as X-API-Key header on protected routes
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"
    RATE_LIMIT_QUERY: str = "20/minute"
    RATE_LIMIT_UPLOAD: str = "5/minute"

    # --- Ingestion / Retrieval tuning ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 3
    GRAPH_RESULT_LIMIT: int = 20
    MAX_UPLOAD_MB: int = 25

    # --- Evals (advisory RAG-quality scoring, never blocks a response) ---
    ENABLE_EVALS: bool = True

    @field_validator("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    @classmethod
    def _not_blank(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.API_KEY.strip())


@lru_cache
def get_settings() -> "Settings":
    """Cached settings loader so validation happens once per process."""
    return Settings()


settings = get_settings()

# --- Shared model/client singletons ---
embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
)

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    openai_api_key=settings.OPENAI_API_KEY,
)

# Neo4j AuraDB Connection
graph = Neo4jGraph(
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD,
    database=settings.NEO4J_DATABASE,
    refresh_schema=False,
)

FAISS_INDEX_DIR = ROOT_DIR / "backend" / "faiss_index"
DOC_STORE_DB = ROOT_DIR / "backend" / "doc_store.sqlite3"