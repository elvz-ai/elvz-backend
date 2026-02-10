"""
Platform Orchestrator - Top-level request coordinator.
Routes requests to Elves and aggregates responses.
"""

import time
import uuid
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from app.agents.platform_orchestrator.intent_classifier import ElfType
from app.agents.platform_orchestrator.response_aggregator import (
    AggregatedResponse,
    response_aggregator,
)
from app.agents.platform_orchestrator.router import RoutingResult, elf_router
from app.core.cache import cache

logger = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    """Incoming chat request."""
    user_id: str
    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    """Response to chat request."""
    response: str
    session_id: str
    elf_used: list[str]
    execution_time_ms: int
    suggestions: list[str] = []
    metadata: dict = {}


class SessionState(BaseModel):
    """User session state."""
    session_id: str
    user_id: str
    previous_elf: Optional[str] = None
    topic: Optional[str] = None
    turn_count: int = 0
    context: dict = {}


class PlatformOrchestrator:
    """
    Main orchestrator for the Elvz.ai platform.
    
    Responsibilities:
    - Route requests to appropriate Elf agents
    - Manage session state and context
    - Aggregate responses from multiple Elves
    - Handle errors and fallbacks
    """
    
    def __init__(self):
        self._elves: dict[ElfType, Any] = {}
    
    def register_elf(self, elf_type: ElfType, elf_instance: Any) -> None:
        """Register an Elf with the orchestrator."""
        self._elves[elf_type] = elf_instance
        elf_router.register_elf(elf_type, elf_instance)
        logger.info("Elf registered with orchestrator", elf=elf_type.value)
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat endpoint - processes user message and returns response.
        
        Args:
            request: ChatRequest with user message and context
            
        Returns:
            ChatResponse with Elf output and metadata
        """
        start_time = time.time()
        
        # Get or create session
        session = await self._get_or_create_session(
            request.session_id, request.user_id
        )
        
        try:
            # Route request to appropriate Elf(s)
            routing = await elf_router.route(
                user_message=request.message,
                user_id=request.user_id,
                session_context={
                    "previous_elf": session.previous_elf,
                    "topic": session.topic,
                    **session.context,
                },
            )
            
            logger.info(
                "Request routed",
                primary_elf=routing.primary_elf.value,
                execution_mode=routing.execution_mode,
            )
            
            # Execute Elf(s)
            results = await self._execute_routing(routing, request, session)
            
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Aggregate results
            aggregated = await self._aggregate_results(
                results, routing, request.message, execution_time_ms
            )
            
            # Update session state
            await self._update_session(session, routing, aggregated)
            
            return ChatResponse(
                response=aggregated.content,
                session_id=session.session_id,
                elf_used=aggregated.elves_used,
                execution_time_ms=aggregated.execution_time_ms,
                suggestions=aggregated.suggestions,
                metadata=aggregated.metadata,
            )
            
        except Exception as e:
            logger.error("Chat processing failed", error=str(e))
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return ChatResponse(
                response=f"I apologize, but I encountered an error processing your request. Please try again.",
                session_id=session.session_id,
                elf_used=[],
                execution_time_ms=execution_time_ms,
                suggestions=["Try rephrasing your request", "Start a new conversation"],
                metadata={"error": str(e)},
            )
    
    async def execute_elf(
        self,
        elf_type: str,
        request_data: dict,
        user_id: str,
    ) -> dict:
        """
        Direct Elf execution (bypassing chat interface).
        
        Args:
            elf_type: Type of Elf to execute
            request_data: Task-specific request data
            user_id: User identifier
            
        Returns:
            Elf execution result
        """
        start_time = time.time()
        
        try:
            elf_enum = ElfType(elf_type.replace("-", "_"))
        except ValueError:
            raise ValueError(f"Unknown Elf type: {elf_type}")
        
        elf = self._elves.get(elf_enum)
        if not elf:
            raise ValueError(f"Elf not registered: {elf_type}")
        
        context = {
            "user_id": user_id,
            "direct_execution": True,
        }
        
        result = await elf.execute(request_data, context)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "result": result,
            "elf_type": elf_type,
            "execution_time_ms": execution_time_ms,
        }
    
    async def _get_or_create_session(
        self, session_id: Optional[str], user_id: str
    ) -> SessionState:
        """Get existing session or create new one."""
        if session_id:
            # Try to retrieve existing session
            session_data = await cache.get_session(session_id)
            if session_data:
                return SessionState(**session_data)
        
        # Create new session
        new_session = SessionState(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
        )
        
        await cache.set_session(new_session.session_id, new_session.model_dump())
        
        return new_session
    
    async def _execute_routing(
        self,
        routing: RoutingResult,
        request: ChatRequest,
        session: SessionState,
    ) -> list[tuple[str, dict]]:
        """Execute Elves based on routing decision."""

        # Merge request context with routing context
        # This ensures flags like image/video are passed through
        merged_context = {
            **routing.context,
            **(request.context or {}),
        }

        # Prepare request for Elf
        elf_request = {
            "message": request.message,
            "user_id": request.user_id,
            **merged_context,
        }

        results = []

        if routing.execution_mode == "single":
            # Single Elf execution
            elf = self._elves.get(routing.primary_elf)
            if elf:
                result = await elf.execute(elf_request, merged_context)
                results.append((routing.primary_elf.value, result))
        
        elif routing.execution_mode == "parallel":
            # Parallel execution
            all_elves = [routing.primary_elf] + routing.additional_elves
            parallel_results = await elf_router.execute_parallel(
                all_elves, elf_request, merged_context
            )

            for elf_type, result in zip(all_elves, parallel_results):
                results.append((elf_type.value, result))

        elif routing.execution_mode == "sequential":
            # Sequential execution
            all_elves = [routing.primary_elf] + routing.additional_elves
            sequential_results = await elf_router.execute_sequential(
                all_elves, elf_request, merged_context
            )

            for elf_type, result in zip(all_elves, sequential_results):
                results.append((elf_type.value, result))
        
        return results
    
    async def _aggregate_results(
        self,
        results: list[tuple[str, dict]],
        routing: RoutingResult,
        user_message: str,
        execution_time_ms: int,
    ) -> AggregatedResponse:
        """Aggregate results from Elf execution."""
        
        if not results:
            return AggregatedResponse(
                content="I wasn't able to process your request. Please try again.",
                elves_used=[],
                execution_time_ms=execution_time_ms,
            )
        
        if len(results) == 1:
            elf_type, result = results[0]
            return await response_aggregator.aggregate_single(
                elf_type, result, user_message, execution_time_ms
            )
        
        return await response_aggregator.aggregate_multiple(
            results, user_message, execution_time_ms
        )
    
    async def _update_session(
        self,
        session: SessionState,
        routing: RoutingResult,
        response: AggregatedResponse,
    ) -> None:
        """Update session state after request processing."""
        session.previous_elf = routing.primary_elf.value
        session.turn_count += 1
        
        # Update context with relevant info from response
        if response.metadata.get("topic"):
            session.topic = response.metadata["topic"]
        
        # Save updated session
        await cache.set_session(session.session_id, session.model_dump())
        
        # Extend session TTL
        await cache.extend_session(session.session_id)


# Global orchestrator instance
orchestrator = PlatformOrchestrator()

