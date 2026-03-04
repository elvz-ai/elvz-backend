"""
FastAPI dependencies for authentication and common operations.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """
    Validate API key from X-API-Key header.

    Compares against ELVZ_API_KEY in settings.
    In development mode, skips validation if no key is configured.
    """
    if settings.environment == "development" and not settings.elvz_api_key:
        return x_api_key  # Skip validation in dev if no key configured

    if x_api_key != settings.elvz_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


async def get_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    _: str = Depends(verify_api_key),
) -> str:
    """
    Get user ID from X-User-ID header.

    Requires a valid X-API-Key (validated via dependency).
    In development, falls back to 'dev-user-001' if no X-User-ID is provided.
    In production, X-User-ID is required.
    """
    if x_user_id:
        return x_user_id

    if settings.environment == "development":
        return "dev-user-001"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="X-User-ID header is required",
    )
