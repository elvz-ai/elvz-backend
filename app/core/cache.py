"""
Redis cache implementation for session management,
tool result caching, and analytics caching.
"""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis
from pydantic import BaseModel

from app.core.config import settings


class RedisCache:
    """
    Redis cache manager with typed key patterns.

    Key Patterns:
    - session:{session_id} - User session state (TTL: 1 hour)
    - voice_profile:{user_id} - Brand voice profile (TTL: 24 hours)
    - analytics:{user_id}:{timeframe} - Cached analytics (TTL: 1 hour)
    - tool_result:{tool_name}:{hash} - Tool call results (TTL: 6 hours)
    - style_profile:{user_id} - Pre-computed writing style features (TTL: 7 days)
    """

    STYLE_PROFILE_TTL = 7 * 24 * 3600  # 7 days
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                str(settings.redis_url),
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    # Session Management
    async def set_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Store session data with 1 hour TTL."""
        key = f"session:{session_id}"
        await self.client.setex(
            key,
            settings.redis_session_ttl,
            json.dumps(data),
        )
    
    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve session data."""
        key = f"session:{session_id}"
        data = await self.client.get(key)
        return json.loads(data) if data else None
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session data."""
        key = f"session:{session_id}"
        await self.client.delete(key)
    
    async def extend_session(self, session_id: str) -> None:
        """Extend session TTL."""
        key = f"session:{session_id}"
        await self.client.expire(key, settings.redis_session_ttl)
    
    # Voice Profile Cache
    async def set_voice_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """Store brand voice profile with 24 hour TTL."""
        key = f"voice_profile:{user_id}"
        await self.client.setex(key, 86400, json.dumps(profile))
    
    async def get_voice_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve brand voice profile."""
        key = f"voice_profile:{user_id}"
        data = await self.client.get(key)
        return json.loads(data) if data else None
    
    async def invalidate_voice_profile(self, user_id: str) -> None:
        """Invalidate voice profile cache."""
        key = f"voice_profile:{user_id}"
        await self.client.delete(key)

    # Style Profile Cache (pre-computed from social posts at webhook time)
    async def set_style_profile(self, user_id: str, features: dict[str, Any]) -> None:
        """Store pre-computed style features. TTL: 7 days."""
        key = f"style_profile:{user_id}"
        await self.client.setex(key, self.STYLE_PROFILE_TTL, json.dumps(features))

    async def get_style_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve cached style profile, or None if not yet computed."""
        key = f"style_profile:{user_id}"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    # Analytics Cache
    async def set_analytics(
        self, user_id: str, timeframe: str, data: dict[str, Any]
    ) -> None:
        """Store analytics data with 1 hour TTL."""
        key = f"analytics:{user_id}:{timeframe}"
        await self.client.setex(key, settings.redis_session_ttl, json.dumps(data))
    
    async def get_analytics(
        self, user_id: str, timeframe: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve analytics data."""
        key = f"analytics:{user_id}:{timeframe}"
        data = await self.client.get(key)
        return json.loads(data) if data else None
    
    # Tool Result Cache
    @staticmethod
    def _hash_params(params: dict[str, Any]) -> str:
        """Create deterministic hash of parameters."""
        sorted_json = json.dumps(params, sort_keys=True)
        return hashlib.sha256(sorted_json.encode()).hexdigest()[:16]
    
    async def set_tool_result(
        self, tool_name: str, params: dict[str, Any], result: Any
    ) -> None:
        """Store tool result with 6 hour TTL."""
        param_hash = self._hash_params(params)
        key = f"tool_result:{tool_name}:{param_hash}"
        await self.client.setex(key, settings.redis_cache_ttl, json.dumps(result))
    
    async def get_tool_result(
        self, tool_name: str, params: dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve cached tool result."""
        param_hash = self._hash_params(params)
        key = f"tool_result:{tool_name}:{param_hash}"
        data = await self.client.get(key)
        return json.loads(data) if data else None
    
    # Generic Cache Operations
    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:
        """Set a generic cache value."""
        serialized = json.dumps(value)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a generic cache value."""
        data = await self.client.get(key)
        return json.loads(data) if data else None
    
    async def delete(self, key: str) -> None:
        """Delete a cache key."""
        await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(await self.client.exists(key))


# Global cache instance
cache = RedisCache()

