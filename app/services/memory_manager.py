"""
4-Layer Memory Management System for Conversational AI.

Layers:
1. Working Memory (Redis) - Current conversation context, 1-hour TTL
2. Short-Term Memory (PostgreSQL) - Recent messages, 50 message limit
3. Long-Term Memory (Vector Store) - Semantic embeddings, 6-month retention
4. User Profile Memory (PostgreSQL + Cache) - Brand voice, preferences
"""

import json
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import cache
from app.core.config import settings
from app.core.database import get_db_context
from app.core.vector_store import vector_store
from app.models.conversation import Conversation, Message
from app.models.user import BrandVoiceProfile, User, UserProfile

logger = structlog.get_logger(__name__)


class MemoryManager:
    """
    Manages multi-layer memory for conversational AI.

    Provides unified interface for:
    - Working memory: Current conversation context
    - Short-term memory: Recent message history
    - Long-term memory: Semantic search over embeddings
    - User profile memory: Brand voice and preferences
    """

    # Key prefixes for Redis
    WORKING_MEMORY_PREFIX = "working_memory"
    USER_PROFILE_PREFIX = "user_profile"

    # ==================== Layer 1: Working Memory (Redis) ====================

    async def get_working_memory(self, conversation_id: str) -> dict:
        """
        Get current conversation context from Redis.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Working memory dict with current context
        """
        key = f"{self.WORKING_MEMORY_PREFIX}:{conversation_id}"
        data = await cache.get(key)

        if data:
            logger.debug("Working memory hit", conversation_id=conversation_id)
            return json.loads(data) if isinstance(data, str) else data

        logger.debug("Working memory miss", conversation_id=conversation_id)
        return {}

    async def update_working_memory(
        self,
        conversation_id: str,
        data: dict,
        merge: bool = True,
    ) -> None:
        """
        Update working memory with new data.

        Args:
            conversation_id: Conversation identifier
            data: Data to store
            merge: If True, merge with existing data; otherwise replace
        """
        key = f"{self.WORKING_MEMORY_PREFIX}:{conversation_id}"

        if merge:
            existing = await self.get_working_memory(conversation_id)
            existing.update(data)
            data = existing

        # Add timestamp
        data["_updated_at"] = datetime.utcnow().isoformat()

        await cache.set(key, json.dumps(data), ttl=settings.memory_working_ttl)
        logger.debug("Working memory updated", conversation_id=conversation_id)

    async def clear_working_memory(self, conversation_id: str) -> None:
        """Clear working memory for a conversation."""
        key = f"{self.WORKING_MEMORY_PREFIX}:{conversation_id}"
        await cache.delete(key)
        logger.debug("Working memory cleared", conversation_id=conversation_id)

    # ==================== Layer 2: Short-Term Memory (PostgreSQL) ====================

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> list[dict]:
        """
        Get recent messages from PostgreSQL.

        Args:
            conversation_id: Conversation identifier
            limit: Max messages to retrieve (default: from settings)
            db: Optional database session

        Returns:
            List of message dicts
        """
        limit = limit or settings.conversation_context_window

        async def _fetch(session: AsyncSession) -> list[dict]:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

            # Reverse to chronological order
            return [msg.to_dict() for msg in reversed(messages)]

        if db:
            return await _fetch(db)

        async with get_db_context() as session:
            return await _fetch(session)

    async def get_conversation_summary(
        self,
        conversation_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict]:
        """
        Get conversation summary including metadata.

        Args:
            conversation_id: Conversation identifier
            db: Optional database session

        Returns:
            Conversation summary dict
        """
        async def _fetch(session: AsyncSession) -> Optional[dict]:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()

            if conversation:
                return {
                    "id": conversation.id,
                    "title": conversation.title,
                    "status": conversation.status,
                    "message_count": conversation.message_count,
                    "metadata": conversation.extra_metadata,
                    "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                    "last_message_at": conversation.last_message_at.isoformat() if conversation.last_message_at else None,
                }
            return None

        if db:
            return await _fetch(db)

        async with get_db_context() as session:
            return await _fetch(session)

    # ==================== Layer 3: Long-Term Memory (Vector Store) ====================

    async def retrieve_relevant_context(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search for relevant context from vector store.

        Args:
            query: Search query
            user_id: User identifier for filtering
            top_k: Number of results (default: from settings)
            filters: Additional metadata filters

        Returns:
            List of relevant context items
        """
        top_k = top_k or settings.memory_rag_top_k

        try:
            # Build filter with user_id
            search_filters = {"user_id": user_id}
            if filters:
                search_filters.update(filters)

            results = await vector_store.search_knowledge(
                query=query,
                user_id=user_id,
                top_k=top_k,
                filter_metadata=search_filters,
            )

            return [
                {
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

        except Exception as e:
            logger.error("Vector search failed", error=str(e), user_id=user_id)
            return []

    async def retrieve_user_content(
        self,
        query: str,
        user_id: str,
        platform: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve user's past content for personalization.

        Args:
            query: Search query
            user_id: User identifier
            platform: Optional platform filter
            top_k: Number of results

        Returns:
            List of user content items
        """
        top_k = top_k or settings.memory_rag_top_k

        try:
            filters = {}
            if platform:
                filters["platform"] = platform

            results = await vector_store.search_user_content(
                query=query,
                user_id=user_id,
                top_k=top_k,
                filter_metadata=filters,
            )

            return [
                {
                    "content": r.content,
                    "score": r.score,
                    "platform": r.metadata.get("platform"),
                    "content_type": r.metadata.get("content_type"),
                    "posted_at": r.metadata.get("posted_at"),
                }
                for r in results
            ]

        except Exception as e:
            logger.error("User content search failed", error=str(e), user_id=user_id)
            return []

    # ==================== Layer 4: User Profile Memory (PostgreSQL + Cache) ====================

    async def get_user_profile(
        self,
        user_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict]:
        """
        Get user profile with brand voice from cache or database.

        Args:
            user_id: User identifier
            db: Optional database session

        Returns:
            User profile dict
        """
        # Try cache first
        cache_key = f"{self.USER_PROFILE_PREFIX}:{user_id}"
        cached = await cache.get(cache_key)

        if cached:
            logger.debug("User profile cache hit", user_id=user_id)
            return json.loads(cached) if isinstance(cached, str) else cached

        # Fetch from database
        async def _fetch(session: AsyncSession) -> Optional[dict]:
            # Get user with profile and brand voice
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                return None

            # Get profile
            profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            profile_result = await session.execute(profile_stmt)
            profile = profile_result.scalar_one_or_none()

            # Get brand voice
            voice_stmt = select(BrandVoiceProfile).where(BrandVoiceProfile.user_id == user_id)
            voice_result = await session.execute(voice_stmt)
            brand_voice = voice_result.scalar_one_or_none()

            profile_data = {
                "user_id": user_id,
                "email": user.email,
                "name": user.name,
                "subscription_tier": user.subscription_tier,
                "profile": None,
                "brand_voice": None,
            }

            if profile:
                profile_data["profile"] = {
                    "brand_name": profile.brand_name,
                    "industry": profile.industry,
                    "brand_voice_description": profile.brand_voice,
                    "tone_preferences": profile.tone_preferences,
                    "target_audience": profile.target_audience,
                    "social_platforms": profile.social_platforms,
                }

            if brand_voice:
                profile_data["brand_voice"] = {
                    "tone_characteristics": brand_voice.tone_characteristics,
                    "vocabulary_patterns": brand_voice.vocabulary_patterns,
                    "personality_traits": brand_voice.personality_traits,
                    "content_patterns": brand_voice.content_patterns,
                    "sample_phrases": brand_voice.sample_phrases,
                    "confidence_score": brand_voice.confidence_score,
                }
                profile_data["brand_voice_context"] = brand_voice.to_prompt_context()

            return profile_data

        if db:
            profile_data = await _fetch(db)
        else:
            async with get_db_context() as session:
                profile_data = await _fetch(session)

        # Cache the result
        if profile_data:
            await cache.set(
                cache_key,
                json.dumps(profile_data),
                ttl=settings.redis_cache_ttl,  # 6 hours
            )
            logger.debug("User profile cached", user_id=user_id)

        return profile_data

    async def invalidate_user_profile_cache(self, user_id: str) -> None:
        """Invalidate cached user profile."""
        cache_key = f"{self.USER_PROFILE_PREFIX}:{user_id}"
        await cache.delete(cache_key)
        logger.debug("User profile cache invalidated", user_id=user_id)

    # ==================== Composite Operations ====================

    async def build_conversation_context(
        self,
        conversation_id: str,
        user_id: str,
        current_query: str,
        db: Optional[AsyncSession] = None,
    ) -> dict:
        """
        Build complete context for the conversation.

        Combines all memory layers into a single context dict.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            current_query: Current user query for RAG
            db: Optional database session

        Returns:
            Complete context dict
        """
        context = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Layer 1: Working memory
        context["working_memory"] = await self.get_working_memory(conversation_id)

        # Layer 2: Recent messages
        context["recent_messages"] = await self.get_recent_messages(
            conversation_id, db=db
        )

        # Layer 3: Relevant context from vector store
        context["retrieved_context"] = await self.retrieve_relevant_context(
            query=current_query,
            user_id=user_id,
        )

        # Layer 4: User profile
        context["user_profile"] = await self.get_user_profile(user_id, db=db)

        return context

    async def save_message_to_memory(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> str:
        """
        Save a message to short-term memory (PostgreSQL).

        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            db: Optional database session

        Returns:
            Message ID
        """
        import uuid

        message_id = str(uuid.uuid4())

        async def _save(session: AsyncSession) -> str:
            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata or {},
            )
            session.add(message)

            # Update conversation last_message_at
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()

            if conversation:
                conversation.last_message_at = datetime.utcnow()

            await session.commit()
            return message_id

        if db:
            return await _save(db)

        async with get_db_context() as session:
            return await _save(session)


# Global instance
memory_manager = MemoryManager()
