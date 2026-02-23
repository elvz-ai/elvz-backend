"""
Social Media Manager Mini-Agents.

6 agents for intelligent content generation:
1. PlannerAgent - Decides which agents to run
2. ExpertPersonaAgent - Generates dynamic topic-specific system prompt (runs parallel with planner)
3. ContentAgent - Generates post content (merged strategy + content)
4. OptimizationAgent - Generates hashtags and timing (merged hashtag + timing)
5. VisualAgent - Generates image descriptions and content
6. VideoAgent - Generates video scripts and recommendations
"""

from .content import ContentAgent
from .expert_persona import ExpertPersonaAgent
from .optimization import OptimizationAgent
from .planner import PlannerAgent
from .visual import VisualAgent
from .video import VideoAgent

__all__ = [
    "ContentAgent",
    "ExpertPersonaAgent",
    "OptimizationAgent",
    "PlannerAgent",
    "VisualAgent",
    "VideoAgent",
]

