"""
Conversation Service.

CRUD operations and business logic for conversations.
"""

import uuid
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.models.conversation import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
)
from app.models.user import User

logger = structlog.get_logger(__name__)


class ConversationService:
    """
    Service for managing conversations.

    Provides CRUD operations and conversation-related business logic.
    """

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            user_id: User identifier
            title: Optional conversation title
            metadata: Optional metadata
            db: Optional database session

        Returns:
            Created conversation
        """
        conversation_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())

        async def _create(session: AsyncSession) -> Conversation:
            await self._ensure_user_exists(user_id, session)
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                thread_id=thread_id,
                title=title,
                status=ConversationStatus.ACTIVE.value,
                extra_metadata=metadata or {},
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation

        if db:
            return await _create(db)

        async with get_db_context() as session:
            return await _create(session)

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: Conversation identifier
            user_id: Optional user ID for ownership check
            db: Optional database session

        Returns:
            Conversation or None
        """
        async def _get(session: AsyncSession) -> Optional[Conversation]:
            stmt = select(Conversation).where(Conversation.id == conversation_id)

            if user_id:
                stmt = stmt.where(Conversation.user_id == user_id)

            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def _ensure_user_exists(self, user_id: str, session: AsyncSession) -> None:
        """
        Ensure a user record exists for the given user_id.
        Creates a placeholder user if not found. Used during development
        when auth is bypassed and a bare user_id is passed directly.
        """
        result = await session.execute(select(User).where(User.id == user_id))
        if result.scalar_one_or_none() is None:
            user = User(
                id=user_id,
                email=f"{user_id}@elvz.local",
                name=user_id,
                hashed_password="dev_placeholder",
            )
            session.add(user)
            await session.flush()  # persist within current transaction

    async def get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Conversation:
        """
        Get existing conversation or create new one.
        Auto-creates a placeholder user if one doesn't exist yet.

        Args:
            conversation_id: Optional conversation ID
            user_id: User identifier
            db: Optional database session

        Returns:
            Conversation (existing or new)
        """
        async def _run(session: AsyncSession) -> Conversation:
            await self._ensure_user_exists(user_id, session)

            if conversation_id:
                result = await session.execute(
                    select(Conversation).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
                conv = result.scalar_one_or_none()
                if conv:
                    return conv

            # Create new conversation (use provided ID or generate one)
            conv_id = conversation_id or str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            conv = Conversation(
                id=conv_id,
                user_id=user_id,
                thread_id=thread_id,
                status=ConversationStatus.ACTIVE.value,
                extra_metadata={},
            )
            session.add(conv)
            await session.commit()
            await session.refresh(conv)
            return conv

        if db:
            return await _run(db)

        async with get_db_context() as session:
            return await _run(session)

    async def list_conversations(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        db: Optional[AsyncSession] = None,
    ) -> list[Conversation]:
        """
        List conversations for a user.

        Args:
            user_id: User identifier
            status: Optional status filter
            limit: Max results
            offset: Pagination offset
            db: Optional database session

        Returns:
            List of conversations
        """
        async def _list(session: AsyncSession) -> list[Conversation]:
            stmt = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.last_message_at.desc().nulls_last())
                .offset(offset)
                .limit(limit)
            )

            if status:
                stmt = stmt.where(Conversation.status == status)
            else:
                # Exclude deleted by default
                stmt = stmt.where(Conversation.status != ConversationStatus.DELETED.value)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _list(db)

        async with get_db_context() as session:
            return await _list(session)

    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Conversation]:
        """
        Update a conversation.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            title: New title
            status: New status
            metadata: New metadata
            db: Optional database session

        Returns:
            Updated conversation or None
        """
        async def _update(session: AsyncSession) -> Optional[Conversation]:
            conversation = await self.get_conversation(conversation_id, user_id, session)
            if not conversation:
                return None

            if title is not None:
                conversation.title = title
            if status is not None:
                conversation.status = status
            if metadata is not None:
                conversation.extra_metadata = metadata

            await session.commit()
            await session.refresh(conversation)
            return conversation

        if db:
            return await _update(db)

        async with get_db_context() as session:
            return await _update(session)

    async def count_conversations(
        self,
        user_id: str,
        status: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> int:
        """Count conversations for a user."""
        async def _count(session: AsyncSession) -> int:
            stmt = select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
            if status:
                stmt = stmt.where(Conversation.status == status)
            else:
                stmt = stmt.where(Conversation.status != ConversationStatus.DELETED.value)
            result = await session.execute(stmt)
            return result.scalar_one()

        if db:
            return await _count(db)
        async with get_db_context() as session:
            return await _count(session)

    async def archive_conversation(
        self,
        conversation_id: str,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """Archive a conversation (soft delete alternative)."""
        async def _archive(session: AsyncSession) -> bool:
            conversation = await self.get_conversation(conversation_id, db=session)
            if not conversation:
                return False
            conversation.status = ConversationStatus.ARCHIVED.value
            await session.commit()
            return True

        if db:
            return await _archive(db)
        async with get_db_context() as session:
            return await _archive(session)

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        hard_delete: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Delete a conversation (soft or hard).

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            hard_delete: If True, permanently delete
            db: Optional database session

        Returns:
            True if deleted
        """
        async def _delete(session: AsyncSession) -> bool:
            conversation = await self.get_conversation(conversation_id, user_id, session) if user_id else await self.get_conversation(conversation_id, db=session)
            if not conversation:
                return False

            if hard_delete:
                await session.delete(conversation)
            else:
                conversation.status = ConversationStatus.DELETED.value

            await session.commit()
            return True

        if db:
            return await _delete(db)

        async with get_db_context() as session:
            return await _delete(session)

    # ==================== Message Operations ====================

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            db: Optional database session

        Returns:
            Created message
        """
        message_id = str(uuid.uuid4())

        async def _add(session: AsyncSession) -> Message:
            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata or {},
            )
            session.add(message)

            # Update conversation last_message_at
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(last_message_at=datetime.utcnow())
            )

            await session.commit()
            return message

        if db:
            return await _add(db)

        async with get_db_context() as session:
            return await _add(session)

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
        db: Optional[AsyncSession] = None,
    ) -> list[Message]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: Conversation identifier
            limit: Max messages
            offset: Pagination offset
            db: Optional database session

        Returns:
            List of messages
        """
        async def _get(session: AsyncSession) -> list[Message]:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
                .offset(offset)
                .limit(limit)
            )

            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def get_conversation_with_messages(
        self,
        conversation_id: str,
        user_id: str,
        message_limit: int = 50,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict]:
        """
        Get conversation with its messages.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            message_limit: Max messages to return
            db: Optional database session

        Returns:
            Dict with conversation and messages
        """
        conversation = await self.get_conversation(conversation_id, user_id, db)
        if not conversation:
            return None

        messages = await self.get_messages(conversation_id, limit=message_limit, db=db)

        return {
            "conversation": conversation.to_dict(),
            "messages": [m.to_dict() for m in messages],
        }


# Global instance
conversation_service = ConversationService()
