"""Structured JSON logging configuration with request-id correlation."""
import logging
import sys
import uuid
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

from backend.config import settings

# Holds the current request id so log records emitted anywhere during a
# request's lifecycle (ingestion, retrieval, guardrails) can be correlated.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging() -> None:
    """Configures root logger with JSON output for production log aggregation."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL.upper())

    # Quiet noisy third-party loggers unless explicitly debugging
    for noisy in ("httpx", "httpcore", "neo4j", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
