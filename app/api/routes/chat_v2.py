"""
Conversational chat endpoint using LangGraph.

This is the new endpoint that uses the conversational graph
for multi-turn conversations with memory and checkpointing.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.conversational_graph.graph import invoke_conversation, stream_conversation
from app.api.deps import get_current_user_id
from app.services.conversation_service import conversation_service
from app.services.hitl_service import hitl_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat/v2", tags=["chat-v2"])


class ConversationalChatRequest(BaseModel):
    """Conversational chat request."""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    context: Optional[dict] = None


class ConversationalChatResponse(BaseModel):
    """Conversational chat response."""
    response: str
    conversation_id: str
    thread_id: str
    artifacts: list[dict] = []
    suggestions: list[str] = []
    hitl_request: Optional[dict] = None
    metadata: dict = {}


@router.post("", response_model=ConversationalChatResponse)
async def conversational_chat(
    request: ConversationalChatRequest,
    user_id: str = Depends(get_current_user_id),
) -> ConversationalChatResponse:
    """
    Conversational chat endpoint using LangGraph.
    
    Features:
    - Multi-turn conversations with memory
    - State persistence via checkpointing
    - Intent classification and routing
    - Multi-platform artifact generation
    - Human-in-the-loop workflows
    - Guardrails for content safety
    
    Args:
        request: Chat request with message and optional conversation_id
        user_id: Current user ID (from auth)
    
    Returns:
        Chat response with artifacts, suggestions, and HITL requests
    """
    logger.info(
        "Conversational chat request",
        user_id=user_id,
        conversation_id=request.conversation_id,
        message_length=len(request.message),
    )
    
    try:
        # Get or create conversation
        if request.conversation_id:
            conversation = await conversation_service.get_conversation(
                request.conversation_id
            )
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            if conversation.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            conversation_id = request.conversation_id
            thread_id = conversation.thread_id
        else:
            # Create new conversation (service generates its own IDs)
            conversation = await conversation_service.create_conversation(
                user_id=user_id,
                title=request.message[:100],
                metadata=request.context or {},
            )
            conversation_id = conversation.id
            thread_id = conversation.thread_id
        
        # Invoke conversational graph
        result_state = await invoke_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            thread_id=thread_id,
            user_input=request.message,
            config=request.context,
        )
        
        # Extract response
        final_response = result_state.get("final_response", "I'm processing your request.")
        suggestions = result_state.get("suggestions", [])
        artifacts = result_state.get("artifacts", [])
        
        # Check for pending HITL requests
        hitl_requests = await hitl_service.get_pending_requests(conversation_id)
        hitl_request = hitl_requests[0].to_dict() if hitl_requests else None
        
        # Build metadata
        metadata = {
            "intent": result_state.get("current_intent"),
            "nodes_executed": [trace.get("node") for trace in result_state.get("execution_trace", [])],
            "tokens_used": result_state.get("total_tokens_used", 0),
            "cost": result_state.get("total_cost", 0.0),
            "guardrail_passed": result_state.get("guardrail_passed", True),
        }
        
        return ConversationalChatResponse(
            response=final_response,
            conversation_id=conversation_id,
            thread_id=thread_id,
            artifacts=[
                {
                    "id": a.get("id"),
                    "type": a.get("type"),
                    "platform": a.get("platform"),
                    "content": a.get("content"),
                }
                for a in artifacts
            ],
            suggestions=suggestions,
            hitl_request=hitl_request,
            metadata=metadata,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Conversational chat failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/respond-hitl")
async def respond_to_hitl(
    request_id: str,
    response: str,
    selected_options: Optional[list] = None,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Respond to a HITL request and resume conversation.
    
    Args:
        request_id: HITL request identifier
        response: User's response
        selected_options: Selected option IDs
        user_id: Current user ID (from auth)
    
    Returns:
        Updated HITL request
    """
    try:
        # Get HITL request
        hitl_request = await hitl_service.get_request(request_id)
        
        if not hitl_request:
            raise HTTPException(status_code=404, detail="HITL request not found")
        
        # Verify ownership
        conversation = await conversation_service.get_conversation(
            hitl_request.conversation_id
        )
        
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Respond to request
        updated_request = await hitl_service.respond_to_request(
            request_id=request_id,
            response=response,
            selected_options=selected_options,
            action="approve",
        )
        
        # TODO: Resume conversation with user response
        # This would involve re-invoking the graph with the HITL context
        
        return updated_request.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("HITL response failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
