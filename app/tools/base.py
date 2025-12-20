"""
Base classes for tool implementations.
All tools should inherit from BaseTool.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

import structlog
from pydantic import BaseModel

from app.core.cache import cache

logger = structlog.get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    cached: bool = False
    execution_time_ms: int = 0
    timestamp: datetime = datetime.utcnow()


class BaseTool(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all tools.
    
    Features:
    - Async execution
    - Rate limiting
    - Caching
    - Timeout handling
    - Error recovery
    - Result validation
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    version: str = "1.0"
    
    # Configuration
    cache_enabled: bool = True
    cache_ttl: int = 21600  # 6 hours default
    timeout_seconds: int = 30
    max_retries: int = 2
    
    def __init__(self):
        self._rate_limit_remaining: int = 100
        self._rate_limit_reset: Optional[datetime] = None
    
    @abstractmethod
    async def _execute(self, input_data: InputT) -> OutputT:
        """
        Implement the actual tool logic.
        Subclasses must override this method.
        """
        pass
    
    def _get_cache_key(self, input_data: InputT) -> dict[str, Any]:
        """
        Get parameters for cache key generation.
        Override if custom cache key logic is needed.
        """
        return input_data.model_dump()
    
    async def _validate_input(self, input_data: InputT) -> None:
        """
        Validate input before execution.
        Override to add custom validation.
        """
        pass
    
    async def _validate_output(self, output_data: OutputT) -> None:
        """
        Validate output after execution.
        Override to add custom validation.
        """
        pass
    
    async def execute(self, input_data: InputT) -> ToolResult:
        """
        Execute the tool with caching, timeout, and error handling.
        
        Args:
            input_data: Tool input parameters
            
        Returns:
            ToolResult with success/failure status and data
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            await self._validate_input(input_data)
            
            # Check cache
            if self.cache_enabled:
                cache_params = self._get_cache_key(input_data)
                cached_result = await cache.get_tool_result(self.name, cache_params)
                
                if cached_result is not None:
                    logger.debug("Tool cache hit", tool=self.name)
                    return ToolResult(
                        success=True,
                        data=cached_result,
                        cached=True,
                        execution_time_ms=0,
                    )
            
            # Execute with timeout and retries
            result = await self._execute_with_retry(input_data)
            
            # Validate output
            await self._validate_output(result)
            
            # Cache result
            if self.cache_enabled:
                await cache.set_tool_result(
                    self.name,
                    cache_params,
                    result.model_dump() if isinstance(result, BaseModel) else result,
                )
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ToolResult(
                success=True,
                data=result.model_dump() if isinstance(result, BaseModel) else result,
                execution_time_ms=execution_time,
            )
            
        except asyncio.TimeoutError:
            logger.error("Tool timeout", tool=self.name, timeout=self.timeout_seconds)
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self.timeout_seconds}s",
                execution_time_ms=self.timeout_seconds * 1000,
            )
            
        except Exception as e:
            logger.error("Tool execution failed", tool=self.name, error=str(e))
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    async def _execute_with_retry(self, input_data: InputT) -> OutputT:
        """Execute with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._execute(input_data),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    logger.warning(
                        "Tool retry",
                        tool=self.name,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                    )
        
        raise last_error or TimeoutError("Max retries exceeded")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary for registry."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "cache_enabled": self.cache_enabled,
            "timeout_seconds": self.timeout_seconds,
        }

