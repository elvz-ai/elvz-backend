"""
Conversation API routes.

Endpoints for managing conversations and messages.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_user_id
from app.core.config import settings
from app.services.conversation_service import conversation_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: list[dict]
    total: int
    page: int
    page_size: int


class ConversationDetailResponse(BaseModel):
    """Response for conversation details."""
    conversation: dict
    messages: list[dict]


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user_id: str = Depends(get_current_user_id),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
) -> ConversationListResponse:
    """
    List user's conversations with pagination.
    
    Args:
        user_id: Current user ID (from auth)
        page: Page number (1-indexed)
        page_size: Items per page
        status: Filter by status (active, archived, deleted)
    
    Returns:
        Paginated list of conversations
    """
    try:
        conversations = await conversation_service.list_conversations(
            user_id=user_id,
            status=status,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        
        total = await conversation_service.count_conversations(
            user_id=user_id,
            status=status,
        )
        
        return ConversationListResponse(
            conversations=[c.to_dict() for c in conversations],
            total=total,
            page=page,
            page_size=page_size,
        )
        
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e), user_id=user_id)
        detail = str(e) if settings.environment == "development" else "Internal server error"
        raise HTTPException(status_code=500, detail=detail)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    include_messages: bool = Query(True),
    message_limit: int = Query(50, ge=1, le=200),
) -> ConversationDetailResponse:
    """
    Get conversation details with messages.
    
    Args:
        conversation_id: Conversation identifier
        user_id: Current user ID (from auth)
        include_messages: Whether to include messages
        message_limit: Max messages to return
    
    Returns:
        Conversation with messages
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        messages = []
        if include_messages:
            messages = await conversation_service.get_messages(
                conversation_id=conversation_id,
                limit=message_limit,
            )
        
        return ConversationDetailResponse(
            conversation=conversation.to_dict(),
            messages=[m.to_dict() for m in messages],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get conversation",
            error=str(e),
            conversation_id=conversation_id,
        )
        detail = str(e) if settings.environment == "development" else "Internal server error"
        raise HTTPException(status_code=500, detail=detail)


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    hard_delete: bool = Query(False),
) -> dict:
    """
    Delete or archive a conversation.
    
    Args:
        conversation_id: Conversation identifier
        user_id: Current user ID (from auth)
        hard_delete: If True, permanently delete; otherwise archive
    
    Returns:
        Success message
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if hard_delete:
            await conversation_service.delete_conversation(conversation_id)
            message = "Conversation permanently deleted"
        else:
            await conversation_service.archive_conversation(conversation_id)
            message = "Conversation archived"
        
        return {
            "message": message,
            "conversation_id": conversation_id,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete conversation",
            error=str(e),
            conversation_id=conversation_id,
        )
        detail = str(e) if settings.environment == "development" else "Internal server error"
        raise HTTPException(status_code=500, detail=detail)


@router.post("/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    message: dict,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Add a message to a conversation.
    
    Args:
        conversation_id: Conversation identifier
        message: Message data (role, content, metadata)
        user_id: Current user ID (from auth)
    
    Returns:
        Created message
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        created_message = await conversation_service.add_message(
            conversation_id=conversation_id,
            role=message.get("role", "user"),
            content=message.get("content", ""),
            metadata=message.get("metadata", {}),
        )
        
        return created_message.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to add message",
            error=str(e),
            conversation_id=conversation_id,
        )
        detail = str(e) if settings.environment == "development" else "Internal server error"
        raise HTTPException(status_code=500, detail=detail)


@router.get("/{conversation_id}/summary")
async def get_conversation_summary(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Get conversation summary with statistics.
    
    Args:
        conversation_id: Conversation identifier
        user_id: Current user ID (from auth)
    
    Returns:
        Conversation summary with stats
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get statistics
        message_count = conversation.message_count
        artifact_count = len(conversation.artifacts) if conversation.artifacts else 0
        
        return {
            **conversation.to_dict(),
            "statistics": {
                "message_count": message_count,
                "artifact_count": artifact_count,
                "platforms": (conversation.extra_metadata or {}).get("platforms", []),
                "total_tokens": (conversation.extra_metadata or {}).get("total_tokens", 0),
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get conversation summary",
            error=str(e),
            conversation_id=conversation_id,
        )
        detail = str(e) if settings.environment == "development" else "Internal server error"
        raise HTTPException(status_code=500, detail=detail)
