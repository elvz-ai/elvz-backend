"""
Main chat endpoint - conversational interface to all Elves.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.platform_orchestrator import orchestrator
from app.api.deps import get_current_user_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    context: Optional[dict] = None
    image: bool = Field(default=False, description="Whether to generate image content")
    video: bool = Field(default=False, description="Whether to generate video content")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Create a LinkedIn post about AI in healthcare",
                "session_id": "abc123",
                "context": {"brand_name": "HealthTech Co"},
                "image": True,
                "video": False
            }
        }


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str
    session_id: str
    elf_used: list[str]
    execution_time_ms: int
    suggestions: list[str] = []
    metadata: dict = {}


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
) -> ChatResponse:
    """
    Main chat endpoint - processes natural language requests.
    
    Automatically routes to the appropriate Elf based on intent classification.
    
    Examples:
    - "Create a LinkedIn post about AI" → Social Media Manager
    - "Audit my website for SEO" → SEO Optimizer
    - "Write a blog post about productivity" → Copywriter
    - "Help me manage my tasks" → AI Assistant
    """
    logger.info(
        "Chat request received",
        user_id=user_id,
        message_length=len(request.message),
    )
    
    try:
        from app.agents.platform_orchestrator.orchestrator import ChatRequest as OrchestratorRequest

        # Merge image/video flags into context
        context = request.context or {}
        context["image"] = request.image
        context["video"] = request.video

        result = await orchestrator.chat(
            OrchestratorRequest(
                user_id=user_id,
                message=request.message,
                session_id=request.session_id,
                context=context,
            )
        )
        
        return ChatResponse(
            response=result.response,
            session_id=result.session_id,
            elf_used=result.elf_used,
            execution_time_ms=result.execution_time_ms,
            suggestions=result.suggestions,
            metadata=result.metadata,
        )
        
    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Get session information."""
    from app.core.cache import cache
    
    session = await cache.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return session


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Delete a session."""
    from app.core.cache import cache
    
    session = await cache.get_session(session_id)
    
    if session and session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    await cache.delete_session(session_id)
    
    return {"message": "Session deleted", "session_id": session_id}

