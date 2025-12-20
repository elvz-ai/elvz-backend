"""
Request router for directing requests to appropriate Elf agents.
Handles multi-Elf coordination when needed.
"""

import asyncio
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from app.agents.platform_orchestrator.intent_classifier import (
    ElfType,
    IntentClassification,
    intent_classifier,
)

logger = structlog.get_logger(__name__)


class RoutingResult(BaseModel):
    """Result of routing decision."""
    primary_elf: ElfType
    additional_elves: list[ElfType] = []
    execution_mode: str = "single"  # "single", "sequential", "parallel"
    context: dict = {}


class ElfRouter:
    """
    Routes requests to appropriate Elf agents based on intent classification.
    Supports single-Elf and multi-Elf coordination.
    """
    
    def __init__(self):
        self._elf_registry: dict[ElfType, Any] = {}
    
    def register_elf(self, elf_type: ElfType, elf_instance: Any) -> None:
        """Register an Elf agent."""
        self._elf_registry[elf_type] = elf_instance
        logger.info("Elf registered", elf_type=elf_type.value)
    
    def get_elf(self, elf_type: ElfType) -> Optional[Any]:
        """Get an Elf agent by type."""
        return self._elf_registry.get(elf_type)
    
    async def route(
        self,
        user_message: str,
        user_id: str,
        session_context: Optional[dict] = None,
    ) -> RoutingResult:
        """
        Route a user request to the appropriate Elf(s).
        
        Args:
            user_message: The user's message
            user_id: User identifier
            session_context: Optional session context from previous turns
            
        Returns:
            RoutingResult with Elf assignment and execution mode
        """
        session_context = session_context or {}
        
        # Classify intent
        classification = await intent_classifier.classify(
            user_message=user_message,
            previous_elf=session_context.get("previous_elf"),
            session_topic=session_context.get("topic"),
        )
        
        logger.info(
            "Intent classified",
            primary_elf=classification.primary_elf.value,
            intent=classification.intent.value,
            confidence=classification.confidence,
        )
        
        # Determine execution mode
        if classification.requires_multi_elf:
            execution_mode = self._determine_multi_elf_mode(classification)
            additional_elves = classification.secondary_elves
        else:
            execution_mode = "single"
            additional_elves = []
        
        # Build context for Elf execution
        context = {
            "intent": classification.intent.value,
            "entities": classification.extracted_entities,
            "confidence": classification.confidence,
            "user_id": user_id,
            "session": session_context,
        }
        
        return RoutingResult(
            primary_elf=classification.primary_elf,
            additional_elves=additional_elves,
            execution_mode=execution_mode,
            context=context,
        )
    
    def _determine_multi_elf_mode(self, classification: IntentClassification) -> str:
        """
        Determine how multiple Elves should execute.
        
        - sequential: Output of one feeds into next
        - parallel: Independent execution, results aggregated
        """
        # Define which Elf combinations benefit from parallel execution
        parallel_pairs = {
            (ElfType.SOCIAL_MEDIA, ElfType.COPYWRITER),  # Content can be created in parallel
            (ElfType.SEO, ElfType.COPYWRITER),  # SEO audit + content can run together
        }
        
        primary = classification.primary_elf
        secondaries = set[ElfType](classification.secondary_elves)
        
        for pair in parallel_pairs:
            if primary in pair and any(s in pair for s in secondaries):
                return "parallel"
        
        return "sequential"
    
    async def execute_single(
        self,
        elf_type: ElfType,
        request: dict,
        context: dict,
    ) -> dict:
        """Execute a single Elf."""
        elf = self.get_elf(elf_type)
        if not elf:
            raise ValueError(f"Elf not found: {elf_type.value}")
        
        return await elf.execute(request, context)
    
    async def execute_sequential(
        self,
        elves: list[ElfType],
        request: dict,
        context: dict,
    ) -> list[dict]:
        """Execute multiple Elves sequentially."""
        results = []
        current_context = context.copy()
        
        for elf_type in elves:
            elf = self.get_elf(elf_type)
            if not elf:
                logger.warning(f"Elf not found: {elf_type.value}")
                continue
            
            result = await elf.execute(request, current_context)
            results.append(result)
            
            # Pass result to next Elf's context
            current_context["previous_result"] = result
        
        return results
    
    async def execute_parallel(
        self,
        elves: list[ElfType],
        request: dict,
        context: dict,
    ) -> list[dict]:
        """Execute multiple Elves in parallel."""
        tasks = []
        
        for elf_type in elves:
            elf = self.get_elf(elf_type)
            if elf:
                tasks.append(elf.execute(request, context))
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Elf execution failed", error=str(result))
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results


# Global router instance
elf_router = ElfRouter()

