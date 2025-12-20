"""
State definitions for Social Media Manager Elf.
Typed state that flows through the LangGraph workflow.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ContentVariation(BaseModel):
    """Single content variation."""
    version: str  # "hook_focused", "story_focused", "value_focused"
    content: dict  # {"post_text": str, "reasoning": str}
    hashtags: list[dict] = []  # [{"tag": str, "volume": str, "relevance": float}]
    posting_schedule: dict = {}  # {"datetime": str, "timezone": str, "reason": str}
    visual_recommendations: list[dict] = []  # [{"type": str, "description": str}]
    complete_preview: str = ""
    estimated_engagement: dict = {}  # {"reach": int, "engagement_rate": float, "confidence": float}


class StrategyBrief(BaseModel):
    """Content strategy brief from Strategy Agent."""
    tone: str
    key_messages: list[str]
    cta_approach: str
    target_audience: str
    content_pillars: list[str]
    platform_specific_notes: str = ""


class SocialMediaState(BaseModel):
    """
    State schema for Social Media Manager workflow.
    
    Flows through:
    1. Strategy Agent -> strategy
    2. Content Generator -> content_variations (parallel)
    3. Hashtag Research -> hashtags (parallel)
    4. Timing Optimizer -> timing (parallel)
    5. Visual Advisor -> visual_advice (parallel)
    6. Synthesizer -> final_output
    """
    
    # Input
    user_request: dict
    context: dict = {}
    
    # Strategy (from Strategy Agent)
    strategy: Optional[StrategyBrief] = None
    
    # Content (from Content Generator)
    content_variations: list[ContentVariation] = []
    
    # Hashtags (from Hashtag Research)
    hashtags: list[dict] = []
    
    # Timing (from Timing Optimizer)
    timing: Optional[dict] = None
    
    # Visual Advice (from Visual Advisor)
    visual_advice: list[dict] = []
    
    # Final synthesized output
    final_output: Optional[dict] = None
    
    # Metadata
    retry_count: int = 0
    errors: list[str] = []
    execution_trace: list[dict] = []
    
    class Config:
        arbitrary_types_allowed = True


class CreatePostRequest(BaseModel):
    """Request to create a social media post."""
    user_id: str
    platform: str  # "linkedin", "twitter", "facebook", "instagram"
    topic: str
    content_type: str = "thought_leadership"  # "announcement", "thought_leadership", "promotion"
    brand_voice: Optional[str] = None
    goals: list[str] = ["engagement"]  # ["awareness", "engagement", "conversions"]
    additional_context: Optional[str] = None


class CreatePostResponse(BaseModel):
    """Response from create post workflow."""
    post_variations: list[ContentVariation]
    recommendations: dict  # {"best_variation": str, "reason": str, "suggested_action": str}
    execution_time_ms: int
    tokens_used: int

