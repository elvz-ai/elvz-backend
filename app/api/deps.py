"""
FastAPI dependencies for authentication and common operations.
"""

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
    x_user_id: str = Header(..., alias="X-User-ID"),
    _: str = Depends(verify_api_key),
) -> str:
    """
    Get user ID from X-User-ID header.

    Requires a valid X-API-Key (validated via dependency).
    X-User-ID is always required — the frontend must send the real user ID.
    """
    return x_user_id
