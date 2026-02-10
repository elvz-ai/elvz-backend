"""
Social Media Manager Mini-Agents.

5 agents for intelligent content generation:
1. PlannerAgent - Decides which agents to run
2. ContentAgent - Generates post content (merged strategy + content)
3. OptimizationAgent - Generates hashtags and timing (merged hashtag + timing)
4. VisualAgent - Generates image descriptions and content
5. VideoAgent - Generates video scripts and recommendations
"""

from .content import ContentAgent
from .optimization import OptimizationAgent
from .planner import PlannerAgent
from .visual import VisualAgent
from .video import VideoAgent

__all__ = [
    "ContentAgent",
    "OptimizationAgent",
    "PlannerAgent",
    "VisualAgent",
    "VideoAgent",
]

