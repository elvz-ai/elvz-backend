"""
Social Media Manager Mini-Agents.

4 agents for intelligent content generation:
1. PlannerAgent - Decides which agents to run (NEW)
2. ContentAgent - Generates post content (merged strategy + content)
3. OptimizationAgent - Generates hashtags and timing (merged hashtag + timing)
4. VisualAgent - Generates visual descriptions (simplified, no RAG)
"""

from .content import ContentAgent
from .optimization import OptimizationAgent
from .planner import PlannerAgent
from .visual import VisualAgent

__all__ = [
    "ContentAgent",
    "OptimizationAgent",
    "PlannerAgent",
    "VisualAgent",
]

