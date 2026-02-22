"""
Human-in-the-Loop (HITL) Service.

Manages HITL request lifecycle, user responses, and conversation resumption.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_context
from app.models.hitl import HITLRequest, HITLRequestType, HITLStatus

logger = structlog.get_logger(__name__)


class HITLService:
    """
    Service for managing Human-in-the-Loop workflows.
    
    Handles:
    - Creating HITL requests when system needs user input
    - Managing request lifecycle (pending, approved, rejected, expired)
    - Processing user responses
    - Resuming conversations after HITL
    """

    async def create_request(
        self,
        conversation_id: str,
        request_type: str,
        prompt: str,
        options: Optional[list] = None,
        context: Optional[dict] = None,
        artifact_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        requester_notes: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> HITLRequest:
        """
        Create a new HITL request.

        Args:
            conversation_id: Conversation identifier
            request_type: Type of HITL request
            prompt: Question/prompt for user
            options: Optional list of options for user to choose from
            context: Context for resuming conversation
            artifact_id: Optional artifact being reviewed
            timeout_seconds: Request timeout (default from settings)
            requester_notes: Why this request was triggered
            db: Optional database session

        Returns:
            Created HITLRequest
        """
        timeout_seconds = timeout_seconds or settings.hitl_request_timeout_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        async def _create(session: AsyncSession) -> HITLRequest:
            request = HITLRequest(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                request_type=request_type,
                status=HITLStatus.PENDING.value,
                prompt=prompt,
                options=options,
                context=context or {},
                expires_at=expires_at,
                requester_notes=requester_notes,
            )

            session.add(request)
            await session.commit()
            await session.refresh(request)

            logger.info(
                "HITL request created",
                request_id=request.id,
                conversation_id=conversation_id,
                request_type=request_type,
            )

            return request

        if db:
            return await _create(db)

        async with get_db_context() as session:
            return await _create(session)

    async def get_request(
        self,
        request_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[HITLRequest]:
        """Get HITL request by ID."""
        async def _get(session: AsyncSession) -> Optional[HITLRequest]:
            stmt = select(HITLRequest).where(HITLRequest.id == request_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def get_pending_requests(
        self,
        conversation_id: str,
        db: Optional[AsyncSession] = None,
    ) -> list[HITLRequest]:
        """Get all pending HITL requests for a conversation."""
        async def _get(session: AsyncSession) -> list[HITLRequest]:
            stmt = (
                select(HITLRequest)
                .where(
                    HITLRequest.conversation_id == conversation_id,
                    HITLRequest.status == HITLStatus.PENDING.value,
                )
                .order_by(HITLRequest.requested_at.desc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)

    async def respond_to_request(
        self,
        request_id: str,
        response: Optional[str] = None,
        selected_options: Optional[list] = None,
        action: str = "approve",
        notes: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> HITLRequest:
        """
        Respond to a HITL request.

        Args:
            request_id: HITL request identifier
            response: User's text response
            selected_options: User's selected option IDs
            action: approve, reject, or modify
            notes: Additional notes from user
            db: Optional database session

        Returns:
            Updated HITLRequest
        """
        async def _respond(session: AsyncSession) -> HITLRequest:
            stmt = select(HITLRequest).where(HITLRequest.id == request_id)
            result = await session.execute(stmt)
            request = result.scalar_one_or_none()

            if not request:
                raise ValueError(f"HITL request {request_id} not found")

            if request.status != HITLStatus.PENDING.value:
                raise ValueError(f"HITL request {request_id} is not pending")

            # Update request
            request.response = response
            request.selected_options = selected_options
            request.responded_at = datetime.utcnow()
            request.responder_notes = notes

            if action == "approve":
                request.status = HITLStatus.APPROVED.value
            elif action == "reject":
                request.status = HITLStatus.REJECTED.value
            elif action == "modify":
                request.status = HITLStatus.MODIFIED.value
            else:
                raise ValueError(f"Invalid action: {action}")

            await session.commit()
            await session.refresh(request)

            logger.info(
                "HITL request responded",
                request_id=request_id,
                action=action,
            )

            return request

        if db:
            return await _respond(db)

        async with get_db_context() as session:
            return await _respond(session)

    async def expire_old_requests(
        self,
        db: Optional[AsyncSession] = None,
    ) -> int:
        """
        Mark expired pending requests as expired.

        Returns:
            Number of requests expired
        """
        async def _expire(session: AsyncSession) -> int:
            now = datetime.utcnow()

            stmt = select(HITLRequest).where(
                HITLRequest.status == HITLStatus.PENDING.value,
                HITLRequest.expires_at <= now,
            )
            result = await session.execute(stmt)
            requests = result.scalars().all()

            count = 0
            for request in requests:
                request.status = HITLStatus.EXPIRED.value
                count += 1

            if count > 0:
                await session.commit()
                logger.info(f"Expired {count} HITL requests")

            return count

        if db:
            return await _expire(db)

        async with get_db_context() as session:
            return await _expire(session)

    async def cancel_request(
        self,
        request_id: str,
        db: Optional[AsyncSession] = None,
    ) -> HITLRequest:
        """Cancel a pending HITL request."""
        async def _cancel(session: AsyncSession) -> HITLRequest:
            stmt = select(HITLRequest).where(HITLRequest.id == request_id)
            result = await session.execute(stmt)
            request = result.scalar_one_or_none()

            if not request:
                raise ValueError(f"HITL request {request_id} not found")

            if request.status != HITLStatus.PENDING.value:
                raise ValueError(f"HITL request {request_id} is not pending")

            request.status = HITLStatus.CANCELLED.value
            request.responded_at = datetime.utcnow()

            await session.commit()
            await session.refresh(request)

            logger.info("HITL request cancelled", request_id=request_id)

            return request

        if db:
            return await _cancel(db)

        async with get_db_context() as session:
            return await _cancel(session)

    async def get_conversation_requests(
        self,
        conversation_id: str,
        limit: int = 10,
        db: Optional[AsyncSession] = None,
    ) -> list[HITLRequest]:
        """Get recent HITL requests for a conversation."""
        async def _get(session: AsyncSession) -> list[HITLRequest]:
            stmt = (
                select(HITLRequest)
                .where(HITLRequest.conversation_id == conversation_id)
                .order_by(HITLRequest.requested_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        if db:
            return await _get(db)

        async with get_db_context() as session:
            return await _get(session)


# Global instance
hitl_service = HITLService()
