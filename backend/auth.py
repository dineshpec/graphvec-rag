"""API-key authentication dependency for protected FastAPI routes."""
from fastapi import Header, HTTPException, status

from backend.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Validates the X-API-Key header against the configured API key.

    When ``API_KEY`` is not configured (empty), auth is disabled so local
    development keeps working without extra setup. Set API_KEY in the
    environment to require it in staging/production.
    """
    if not settings.auth_enabled:
        return
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
