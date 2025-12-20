"""
AI Assistant Elf API routes.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.elves.ai_assistant import AIAssistantElf
from app.api.deps import get_current_user_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/elves/assistant", tags=["assistant"])

# Initialize Elf
assistant_elf = AIAssistantElf()


class ManageTasksRequest(BaseModel):
    """Request to manage tasks."""
    message: str = Field(..., min_length=3, max_length=2000)
    existing_tasks: Optional[list[dict]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I need to prepare for the Q4 board meeting next week, finish the product roadmap, and review the marketing budget",
            }
        }


class DraftEmailRequest(BaseModel):
    """Request to draft an email."""
    message: str = Field(..., min_length=3, max_length=2000)
    email_context: Optional[dict] = None
    tone: str = Field(default="professional", pattern="^(professional|casual|formal|friendly)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Write a follow-up email to a potential client after our demo meeting yesterday",
                "email_context": {
                    "recipient": "John Smith",
                    "company": "Tech Corp",
                    "meeting_topic": "AI automation demo",
                },
                "tone": "professional",
            }
        }


class ResearchRequest(BaseModel):
    """Request for research."""
    topic: str = Field(..., min_length=3, max_length=500)
    depth: str = Field(default="moderate", pattern="^(brief|moderate|comprehensive)$")
    specific_questions: Optional[list[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Current trends in AI-powered customer service",
                "depth": "moderate",
                "specific_questions": ["What are the leading solutions?", "What's the ROI?"],
            }
        }


class SummarizeMeetingRequest(BaseModel):
    """Request to summarize meeting notes."""
    meeting_notes: str = Field(..., min_length=50, max_length=20000)
    meeting_type: Optional[str] = None  # "standup", "planning", "review", "general"
    
    class Config:
        json_schema_extra = {
            "example": {
                "meeting_notes": "Meeting started at 2pm. John presented the Q3 results...",
                "meeting_type": "review",
            }
        }


class GenerateDocumentRequest(BaseModel):
    """Request to generate a document."""
    document_type: str = Field(
        ..., 
        pattern="^(report|proposal|memo|summary|brief)$"
    )
    topic: str = Field(..., min_length=3, max_length=500)
    key_points: list[str] = Field(default_factory=list)
    needs_research: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_type": "proposal",
                "topic": "Implementing AI chatbot for customer support",
                "key_points": ["Cost savings", "Improved response time", "24/7 availability"],
                "needs_research": True,
            }
        }


@router.post("/manage-tasks")
async def manage_tasks(
    request: ManageTasksRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Process task-related requests.
    
    Extracts and organizes tasks from natural language input.
    """
    logger.info(
        "Manage tasks request",
        user_id=user_id,
    )
    
    try:
        result = await assistant_elf.execute(
            request={
                "message": request.message,
                "existing_tasks": request.existing_tasks,
            },
            context={"user_id": user_id},
        )
        
        return {
            "tasks": result.get("tasks", []),
            "schedule_suggestions": result.get("schedule_suggestions", []),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Manage tasks failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/draft-email")
async def draft_email(
    request: DraftEmailRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Draft an email based on context.
    
    Returns draft with alternatives.
    """
    logger.info(
        "Draft email request",
        user_id=user_id,
    )
    
    try:
        result = await assistant_elf.execute(
            request={
                "message": request.message,
                "email_context": request.email_context,
                "tone": request.tone,
            },
            context={"user_id": user_id},
        )
        
        return {
            "draft": result.get("draft", {}),
            "alternatives": result.get("alternatives", []),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Draft email failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research")
async def research(
    request: ResearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Conduct research on a topic.
    
    Returns organized findings with sources.
    """
    logger.info(
        "Research request",
        user_id=user_id,
        topic=request.topic[:50],
    )
    
    try:
        result = await assistant_elf.execute(
            request={
                "message": request.topic,
                "topic": request.topic,
                "depth": request.depth,
                "specific_questions": request.specific_questions,
            },
            context={"user_id": user_id},
        )
        
        return {
            "topic": request.topic,
            "findings": result.get("findings", {}),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Research failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize-meeting")
async def summarize_meeting(
    request: SummarizeMeetingRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Summarize meeting notes and extract action items.
    """
    logger.info(
        "Summarize meeting request",
        user_id=user_id,
    )
    
    try:
        result = await assistant_elf.execute(
            request={
                "message": request.meeting_notes,
                "meeting_notes": request.meeting_notes,
                "meeting_type": request.meeting_type,
            },
            context={"user_id": user_id},
        )
        
        return {
            "summary": result.get("summary", ""),
            "key_decisions": result.get("key_decisions", []),
            "action_items": result.get("action_items", []),
            "follow_ups": result.get("follow_ups", []),
            "next_steps": result.get("next_steps", ""),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Summarize meeting failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-document")
async def generate_document(
    request: GenerateDocumentRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Generate a formatted document.
    """
    logger.info(
        "Generate document request",
        user_id=user_id,
        doc_type=request.document_type,
    )
    
    try:
        result = await assistant_elf.execute(
            request={
                "message": f"Create a {request.document_type} about {request.topic}",
                "document_type": request.document_type,
                "topic": request.topic,
                "key_points": request.key_points,
                "needs_research": request.needs_research,
            },
            context={"user_id": user_id},
        )
        
        return {
            "document_type": request.document_type,
            "title": result.get("title", request.topic),
            "content": result.get("content", ""),
            "sections": result.get("sections", []),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Generate document failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

