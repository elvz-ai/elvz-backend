"""
Artifact Service.

CRUD operations and business logic for generated artifacts.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.models.artifact import (
    Artifact,
    ArtifactBatch,
    ArtifactStatus,
)
from app.models.user import User

logger = structlog.get_logger(__name__)


class ArtifactService:
    """
    Service for managing artifacts.

    Provides CRUD operations for generated content artifacts.
    """

    # ==================== Artifact Operations ====================

    async def create_artifact(
        self,
        user_id: str,
        artifact_type: str,
        platform: str,
        content: dict,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        generation_metadata: Optional[dict] = None,
        db: Optional[AsyncSession] = None,
    ) -> Artifact:
        """
        Create a new artifact.

        Args:
            user_id: User identifier (required)
            artifact_type: Type of artifact
            platform: Target platform
            content: Artifact content
            conversation_id: Optional conversation identifier
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
                user_id=user_id,
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

    async def get_current_artifact(
        self,
        artifact_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Get the latest version of an artifact.

        If the artifact has child versions, returns the most recent child.
        Otherwise returns the artifact itself (it is the original with no edits).
        """
        async def _get(session: AsyncSession) -> Optional[Artifact]:
            # Find latest child version
            stmt = (
                select(Artifact)
                .where(Artifact.parent_artifact_id == artifact_id)
                .order_by(Artifact.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            child = result.scalar_one_or_none()
            if child:
                return child

            # No children — return the original
            stmt = select(Artifact).where(Artifact.id == artifact_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        if db:
            return await _get(db)
        async with get_db_context() as session:
            return await _get(session)

    async def get_artifact_versions(
        self,
        artifact_id: str,
        db: Optional[AsyncSession] = None,
    ) -> list[Artifact]:
        """
        Get all versions of an artifact, ordered by created_at ascending.

        Includes the original (parent_artifact_id IS NULL) as version 0,
        followed by all child rows.
        """
        async def _get(session: AsyncSession) -> list[Artifact]:
            stmt = (
                select(Artifact)
                .where(
                    (Artifact.id == artifact_id) | (Artifact.parent_artifact_id == artifact_id)
                )
                .order_by(Artifact.created_at.asc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _get(db)
        async with get_db_context() as session:
            return await _get(session)

    async def create_artifact_version(
        self,
        artifact_id: str,
        updates: dict,
        source: str = "user_edit",
        prompt: Optional[str] = None,
        message_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Create a new version row for an artifact.

        1. Resolves the original artifact ID
        2. Gets the current version (latest child or original)
        3. Merges updates into its content
        4. Computes diff
        5. Inserts a new child row with parent_artifact_id = original_id
        """
        async def _create(session: AsyncSession) -> Optional[Artifact]:
            # Resolve original ID (in case artifact_id is already a version)
            original = await self.get_artifact(artifact_id, session)
            if not original:
                return None

            original_id = original.parent_artifact_id or original.id

            # Get current version content
            current = await self.get_current_artifact(original_id, session)
            if not current:
                return None

            old_content = current.content or {}
            new_content = {**old_content, **updates}

            diff = _compute_content_diff(old_content, new_content)
            if not diff:
                return current  # No actual changes

            # Build plain-text edit_diff: describes what changed between parent and this version
            changed_fields = list(diff.get("changed", {}).keys())
            added_fields = list(diff.get("added", {}).keys())
            removed_fields = list(diff.get("removed", {}).keys())
            parts = []
            if changed_fields:
                parts.append(f"changed {', '.join(changed_fields)}")
            if added_fields:
                parts.append(f"added {', '.join(added_fields)}")
            if removed_fields:
                parts.append(f"removed {', '.join(removed_fields)}")
            edit_diff_text = ", ".join(parts) or "modified content"
            if prompt:
                edit_diff_text = f"{prompt} — {edit_diff_text}"

            # Resolve fields from the original row for consistency
            original_row = original if original.parent_artifact_id is None else await self.get_artifact(original_id, session)

            version_id = str(uuid.uuid4())
            version = Artifact(
                id=version_id,
                user_id=original_row.user_id,
                conversation_id=current.conversation_id,
                message_id=message_id,
                batch_id=current.batch_id,
                artifact_type=current.artifact_type,
                platform=current.platform,
                content=new_content,
                status=current.status,
                parent_artifact_id=original_id,
                edit_diff=edit_diff_text,
                was_published=current.was_published,
                generation_metadata=current.generation_metadata,
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
            return version

        if db:
            return await _create(db)
        async with get_db_context() as session:
            return await _create(session)

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
                .where(Artifact.parent_artifact_id.is_(None))  # only originals
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
        Mark an artifact as published. Always operates on the original row.

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

            # Resolve to original row for status updates
            if artifact.parent_artifact_id:
                artifact = await self.get_artifact(artifact.parent_artifact_id, session)
                if not artifact:
                    return None

            artifact.status = ArtifactStatus.PUBLISHED.value
            artifact.was_published = True
            artifact.published_at = datetime.now(timezone.utc)

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
                .where(Artifact.parent_artifact_id.is_(None))  # only originals
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
        """Get all original artifacts for a batch (excludes version rows)."""
        async def _get(session: AsyncSession) -> list[Artifact]:
            stmt = (
                select(Artifact)
                .where(Artifact.batch_id == batch_id)
                .where(Artifact.parent_artifact_id.is_(None))  # only originals
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
        was_published: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """Update artifact with user feedback and usage flags. Operates on original row."""
        async def _update(session: AsyncSession) -> Optional[Artifact]:
            artifact = await self.get_artifact(artifact_id, session)
            if not artifact:
                return None
            # Resolve to original row
            if artifact.parent_artifact_id:
                artifact = await self.get_artifact(artifact.parent_artifact_id, session)
                if not artifact:
                    return None
            if rating is not None:
                artifact.user_rating = rating
            if feedback is not None:
                artifact.user_feedback = feedback
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
            batch.completed_at = datetime.now(timezone.utc)
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

    # ==================== Wizard Operations ====================

    async def create_wizard_batch(
        self,
        user_id: str,
        platforms: list[str],
        topic: Optional[str] = None,
    ) -> dict:
        """
        Create an artifact batch for the wizard flow (no conversation).

        Args:
            user_id: User identifier
            platforms: Target platforms
            topic: Content topic/idea

        Returns:
            Dict with batch_id
        """
        batch_id = str(uuid.uuid4())

        async with get_db_context() as session:
            # Ensure user exists (creates placeholder in dev when auth is bypassed)
            result = await session.execute(select(User).where(User.id == user_id))
            if result.scalar_one_or_none() is None:
                session.add(User(
                    id=user_id,
                    email=f"{user_id}@elvz.local",
                    name=user_id,
                ))
                await session.flush()

            # Create batch (no conversation in wizard flow)
            batch = ArtifactBatch(
                id=batch_id,
                user_id=user_id,
                conversation_id=None,
                platforms=platforms,
                topic=topic,
                status="in_progress",
                execution_strategy="parallel",
            )
            session.add(batch)

            await session.commit()

            return {
                "batch_id": batch_id,
            }


    async def update_artifact_content(
        self,
        artifact_id: str,
        updates: dict,
        source: str = "user_edit",
        prompt: Optional[str] = None,
        message_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[Artifact]:
        """
        Create a new version row with merged content updates.

        Delegates to create_artifact_version(). Kept for backward compatibility.
        """
        return await self.create_artifact_version(
            artifact_id=artifact_id,
            updates=updates,
            source=source,
            prompt=prompt,
            message_id=message_id,
            db=db,
        )


def _compute_content_diff(old: dict, new: dict) -> dict:
    """Compute field-level diff between two content dicts."""
    diff = {"added": {}, "removed": {}, "changed": {}}
    for key in set(old) | set(new):
        if key not in old:
            diff["added"][key] = new[key]
        elif key not in new:
            diff["removed"][key] = old[key]
        elif old[key] != new[key]:
            diff["changed"][key] = {"old": old[key], "new": new[key]}
    # Drop empty sections
    return {k: v for k, v in diff.items() if v}


# Global instance
artifact_service = ArtifactService()
