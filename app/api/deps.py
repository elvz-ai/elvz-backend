"""
FastAPI dependencies for authentication and common operations.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

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


async def get_current_user_id(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """
    Extract and validate user ID from JWT token.
    
    For development, allows a default user if no token provided or invalid.
    In production, should always require valid JWT.
    """
    # Development mode - be lenient with auth
    if settings.environment == "development":
        if not authorization:
            return "dev-user-001"
        
        # Try to validate, but fall back to dev user if it fails
        try:
            scheme, token = authorization.split()
            if scheme.lower() == "bearer":
                payload = jwt.decode(
                    token,
                    settings.jwt_secret_key,
                    algorithms=[settings.jwt_algorithm],
                )
                user_id = payload.get("sub")
                if user_id:
                    return user_id
        except (ValueError, JWTError):
            pass  # Fall through to return dev user
        
        return "dev-user-001"
    
    # Production mode - strict validation
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from "Bearer <token>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )
    
    # Decode and validate JWT
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        return user_id
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user_id(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """
    Optionally extract user ID - for endpoints that work with or without auth.
    """
    if not authorization:
        return None
    
    try:
        return await get_current_user_id(authorization)
    except HTTPException:
        return None

