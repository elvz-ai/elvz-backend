"""
Artifact Service.

CRUD operations and business logic for generated artifacts.
"""

import uuid
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.models.artifact import (
    Artifact,
    ArtifactBatch,
    ArtifactStatus,
    ArtifactType,
)

logger = structlog.get_logger(__name__)


class ArtifactService:
    """
    Service for managing artifacts.

    Provides CRUD operations for generated content artifacts.
    """

    # ==================== Artifact Operations ====================

    async def create_artifact(
        self,
        conversation_id: str,
        artifact_type: str,
        platform: str,
        content: dict,
        message_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        generation_metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> Artifact:
        """
        Create a new artifact.

        Args:
            conversation_id: Conversation identifier
            artifact_type: Type of artifact
            platform: Target platform
            content: Artifact content
            message_id: Optional message ID
            batch_id: Optional batch ID
            generation_metadata: Generation metadata
            db: Optional database session

        Returns:
            Created artifact
        """
        artifact_id = str(uuid.uuid4())

        async def _create(session: AsyncSession) -> Artifact:
            artifact = Artifact(
                id=artifact_id,
                conversation_id=conversation_id,
                message_id=message_id,
                batch_id=batch_id,
                artifact_type=artifact_type,
                platform=platform,
                content=content,
                status=ArtifactStatus.DRAFT.value,
                generation_metadata=generation_metadata or {},
            )
            session.add(artifact)
            await session.commit()
            await session.refresh(artifact)
            return artifact

        if db:
            return await _create(db)

        async with get_db_context() as session:
            return await _create(session)

    async def get_artifact(
        self,
        artifact_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Get an artifact by ID.

        Args:
            artifact_id: Artifact identifier
            db: Optional database session

        Returns:
            Artifact or None
        """
        async def _get(session: AsyncSession) -> Optional[Artifact]:
            stmt = select(Artifact).where(Artifact.id == artifact_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def list_artifacts(
        self,
        conversation_id: str,
        platform: Optional[str] = None,
        artifact_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        db: Optional[AsyncSession] = None,
    ) -> list[Artifact]:
        """
        List artifacts for a conversation.

        Args:
            conversation_id: Conversation identifier
            platform: Optional platform filter
            artifact_type: Optional type filter
            status: Optional status filter
            limit: Max results
            offset: Pagination offset
            db: Optional database session

        Returns:
            List of artifacts
        """
        async def _list(session: AsyncSession) -> list[Artifact]:
            stmt = (
                select(Artifact)
                .where(Artifact.conversation_id == conversation_id)
                .order_by(Artifact.created_at.desc())
                .offset(offset)
                .limit(limit)
            )

            if platform:
                stmt = stmt.where(Artifact.platform == platform)
            if artifact_type:
                stmt = stmt.where(Artifact.artifact_type == artifact_type)
            if status:
                stmt = stmt.where(Artifact.status == status)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _list(db)

        async with get_db_context() as session:
            return await _list(session)

    async def update_artifact(
        self,
        artifact_id: str,
        content: Optional[dict] = None,
        status: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Update an artifact.

        Args:
            artifact_id: Artifact identifier
            content: New content
            status: New status
            db: Optional database session

        Returns:
            Updated artifact or None
        """
        async def _update(session: AsyncSession) -> Optional[Artifact]:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return None

            if content is not None:
                artifact.content = content
                artifact.was_edited = True
            if status is not None:
                artifact.status = status

            await session.commit()
            await session.refresh(artifact)
            return artifact

        if db:
            return await _update(db)

        async with get_db_context() as session:
            return await _update(session)

    async def add_feedback(
        self,
        artifact_id: str,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Add user feedback to an artifact.

        Args:
            artifact_id: Artifact identifier
            rating: User rating (1-5)
            feedback: User feedback text
            db: Optional database session

        Returns:
            Updated artifact or None
        """
        async def _update(session: AsyncSession) -> Optional[Artifact]:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return None

            if rating is not None:
                artifact.user_rating = rating
            if feedback is not None:
                artifact.user_feedback = feedback

            await session.commit()
            await session.refresh(artifact)
            return artifact

        if db:
            return await _update(db)

        async with get_db_context() as session:
            return await _update(session)

    async def mark_published(
        self,
        artifact_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Mark an artifact as published.

        Args:
            artifact_id: Artifact identifier
            db: Optional database session

        Returns:
            Updated artifact or None
        """
        async def _update(session: AsyncSession) -> Optional[Artifact]:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return None

            artifact.status = ArtifactStatus.PUBLISHED.value
            artifact.was_published = True
            artifact.published_at = datetime.utcnow()

            await session.commit()
            await session.refresh(artifact)
            return artifact

        if db:
            return await _update(db)

        async with get_db_context() as session:
            return await _update(session)

    # ==================== Batch Operations ====================

    async def create_batch(
        self,
        conversation_id: str,
        platforms: list[str],
        topic: Optional[str] = None,
        execution_strategy: str = "sequential",
        db: Optional[AsyncSession] = None,
    ) -> ArtifactBatch:
        """
        Create an artifact batch.

        Args:
            conversation_id: Conversation identifier
            platforms: List of platforms
            topic: Batch topic
            execution_strategy: Execution strategy
            db: Optional database session

        Returns:
            Created batch
        """
        batch_id = str(uuid.uuid4())

        async def _create(session: AsyncSession) -> ArtifactBatch:
            batch = ArtifactBatch(
                id=batch_id,
                conversation_id=conversation_id,
                platforms=platforms,
                topic=topic,
                status="pending",
                execution_strategy=execution_strategy,
            )
            session.add(batch)
            await session.commit()
            await session.refresh(batch)
            return batch

        if db:
            return await _create(db)

        async with get_db_context() as session:
            return await _create(session)

    async def get_batch(
        self,
        batch_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[ArtifactBatch]:
        """
        Get a batch by ID.

        Args:
            batch_id: Batch identifier
            db: Optional database session

        Returns:
            Batch or None
        """
        async def _get(session: AsyncSession) -> Optional[ArtifactBatch]:
            stmt = select(ArtifactBatch).where(ArtifactBatch.id == batch_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def get_batch_with_artifacts(
        self,
        batch_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict]:
        """
        Get batch with its artifacts.

        Args:
            batch_id: Batch identifier
            db: Optional database session

        Returns:
            Dict with batch and artifacts
        """
        batch = await self.get_batch(batch_id, db)
        if not batch:
            return None

        async def _get_artifacts(session: AsyncSession) -> list[Artifact]:
            stmt = (
                select(Artifact)
                .where(Artifact.batch_id == batch_id)
                .order_by(Artifact.created_at.asc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            artifacts = await _get_artifacts(db)
        else:
            async with get_db_context() as session:
                artifacts = await _get_artifacts(session)

        return {
            "batch": batch.to_dict(),
            "artifacts": [a.to_dict() for a in artifacts],
        }

    async def get_batch_artifacts(
        self,
        batch_id: str,
        db: Optional[AsyncSession] = None,
    ) -> list[Artifact]:
        """Get all artifacts for a batch."""
        async def _get(session: AsyncSession) -> list[Artifact]:
            stmt = (
                select(Artifact)
                .where(Artifact.batch_id == batch_id)
                .order_by(Artifact.created_at.asc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _get(db)
        async with get_db_context() as session:
            return await _get(session)

    async def update_artifact_feedback(
        self,
        artifact_id: str,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
        was_edited: bool = False,
        was_published: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """Update artifact with user feedback and usage flags."""
        async def _update(session: AsyncSession) -> Optional[Artifact]:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return None
            if rating is not None:
                artifact.user_rating = rating
            if feedback is not None:
                artifact.user_feedback = feedback
            if was_edited:
                artifact.was_edited = True
            if was_published:
                artifact.was_published = True
                artifact.status = ArtifactStatus.PUBLISHED.value
            await session.commit()
            await session.refresh(artifact)
            return artifact

        if db:
            return await _update(db)
        async with get_db_context() as session:
            return await _update(session)

    async def delete_artifact(
        self,
        artifact_id: str,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """Hard delete an artifact."""
        async def _delete(session: AsyncSession) -> bool:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return False
            await session.delete(artifact)
            await session.commit()
            return True

        if db:
            return await _delete(db)
        async with get_db_context() as session:
            return await _delete(session)

    async def complete_batch(
        self,
        batch_id: str,
        total_tokens: int = 0,
        total_cost: float = 0.0,
        execution_time_ms: int = 0,
        db: Optional[AsyncSession] = None,
    ) -> Optional[ArtifactBatch]:
        """
        Mark a batch as complete.

        Args:
            batch_id: Batch identifier
            total_tokens: Total tokens used
            total_cost: Total cost
            execution_time_ms: Execution time
            db: Optional database session

        Returns:
            Updated batch or None
        """
        async def _update(session: AsyncSession) -> Optional[ArtifactBatch]:
            batch = await self.get_batch(batch_id, session)
            if not batch:
                return None

            batch.status = "complete"
            batch.completed_at = datetime.utcnow()
            batch.total_tokens_used = total_tokens
            batch.total_cost = total_cost
            batch.execution_time_ms = execution_time_ms

            await session.commit()
            await session.refresh(batch)
            return batch

        if db:
            return await _update(db)

        async with get_db_context() as session:
            return await _update(session)


# Global instance
artifact_service = ArtifactService()
