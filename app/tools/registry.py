"""
Tool registry for managing and accessing tools.
Central registry where all tools are registered and can be looked up by agents.
"""

from typing import Any, Optional, Type

import structlog

from app.tools.base import BaseTool

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Tools are organized by category:
    - social_media: Hashtag APIs, analytics, trending topics
    - seo: Crawlers, keyword research, backlink checkers
    - research: Web search, document parsers
    - calendar: Scheduling, timezone handling
    - email: Email templates, thread analysis
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, list[str]] = {}
    
    def register(
        self,
        tool: BaseTool,
        category: str = "general",
    ) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            category: Tool category for organization
        """
        self._tools[tool.name] = tool
        
        if category not in self._categories:
            self._categories[category] = []
        
        if tool.name not in self._categories[category]:
            self._categories[category].append(tool.name)
        
        logger.info("Tool registered", tool=tool.name, category=category)
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_by_category(self, category: str) -> list[BaseTool]:
        """Get all tools in a category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools."""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def list_categories(self) -> dict[str, list[str]]:
        """List all categories and their tools."""
        return self._categories.copy()
    
    async def execute(
        self,
        tool_name: str,
        input_data: Any,
    ) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input parameters for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        return await tool.execute(input_data)


# Global tool registry instance
tool_registry = ToolRegistry()


def register_tool(category: str = "general"):
    """
    Decorator to register a tool class.
    
    Usage:
        @register_tool(category="social_media")
        class HashtagTool(BaseTool):
            ...
    """
    def decorator(tool_class: Type[BaseTool]):
        tool_instance = tool_class()
        tool_registry.register(tool_instance, category)
        return tool_class
    return decorator

