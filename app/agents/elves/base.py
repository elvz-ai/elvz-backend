"""
Base Elf class with common functionality.
All Elf agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, TypeVar

import structlog
from pydantic import BaseModel

from app.core.llm_clients import llm_client

logger = structlog.get_logger(__name__)

StateT = TypeVar("StateT", bound=BaseModel)


class ElfExecutionResult(BaseModel):
    """Result from Elf execution."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    execution_trace: list[dict] = []
    total_time_ms: int = 0
    tokens_used: int = 0


class BaseElf(ABC):
    """
    Abstract base class for all Elf agents.
    
    Each Elf:
    - Manages a LangGraph workflow
    - Coordinates mini-agents
    - Handles state management
    - Provides error recovery
    """
    
    name: str = "base_elf"
    description: str = "Base Elf description"
    version: str = "1.0"
    
    def __init__(self):
        self._mini_agents: dict[str, Any] = {}
        self._workflow = None
        self._setup_workflow()
    
    @abstractmethod
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, request: dict, context: dict) -> dict:
        """
        Execute the Elf workflow.
        
        Args:
            request: Task-specific request data
            context: Execution context (user info, session, etc.)
            
        Returns:
            Execution result
        """
        pass
    
    def register_mini_agent(self, name: str, agent: Any) -> None:
        """Register a mini-agent."""
        self._mini_agents[name] = agent
        logger.debug("Mini-agent registered", elf=self.name, agent=name)
    
    def get_mini_agent(self, name: str) -> Optional[Any]:
        """Get a mini-agent by name."""
        return self._mini_agents.get(name)
    
    async def _execute_mini_agent(
        self,
        agent_name: str,
        state: dict,
        context: dict,
    ) -> dict:
        """
        Execute a single mini-agent.
        
        Args:
            agent_name: Name of the mini-agent
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state
        """
        agent = self.get_mini_agent(agent_name)
        if not agent:
            logger.error("Mini-agent not found", agent=agent_name)
            return state
        
        start_time = datetime.utcnow()
        
        try:
            result = await agent.execute(state, context)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update execution trace
            if "execution_trace" not in state:
                state["execution_trace"] = []
            
            state["execution_trace"].append({
                "agent": agent_name,
                "status": "completed",
                "time_ms": execution_time,
            })
            
            # Merge result into state
            if isinstance(result, dict):
                state.update(result)
            
            logger.debug(
                "Mini-agent completed",
                agent=agent_name,
                time_ms=execution_time,
            )
            
        except Exception as e:
            logger.error("Mini-agent failed", agent=agent_name, error=str(e))
            
            state["execution_trace"].append({
                "agent": agent_name,
                "status": "failed",
                "error": str(e),
            })
            
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"{agent_name}: {str(e)}")
        
        return state
    
    def _create_initial_state(self, request: dict, context: dict) -> dict:
        """Create initial workflow state."""
        return {
            "user_request": request,
            "context": context,
            "execution_trace": [],
            "errors": [],
            "retry_count": 0,
        }
    
    def _create_final_output(self, state: dict) -> dict:
        """Create final output from state."""
        # Remove internal fields
        output = {k: v for k, v in state.items() 
                  if k not in ["execution_trace", "errors", "retry_count", "context"]}
        
        # Add metadata
        output["_metadata"] = {
            "execution_trace": state.get("execution_trace", []),
            "errors": state.get("errors", []),
        }
        
        return output

