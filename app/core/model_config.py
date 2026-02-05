"""
Model Configuration for Task-Based Routing.

Defines which models to use for each task type, with fallback support.
Uses OpenRouter for unified access to multiple LLM providers.

Example usage:
    from app.core.model_config import get_model_config
    
    config = get_model_config("content_generation")
    # Returns: {"model": "anthropic/claude-3.5-sonnet", "fallbacks": [...]}
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class TaskType(str, Enum):
    """Task types that require LLM calls."""
    
    # Platform Orchestrator tasks
    INTENT_CLASSIFICATION = "intent_classification"
    RESPONSE_AGGREGATION = "response_aggregation"
    
    # Social Media tasks
    CONTENT_GENERATION = "content_generation"
    HASHTAG_OPTIMIZATION = "hashtag_optimization"
    TIMING_OPTIMIZATION = "timing_optimization"
    VISUAL_DESCRIPTION = "visual_description"
    
    # SEO tasks
    SEO_ANALYSIS = "seo_analysis"
    KEYWORD_RESEARCH = "keyword_research"
    
    # Copywriter tasks
    BLOG_WRITING = "blog_writing"
    AD_COPY = "ad_copy"
    
    # General
    GENERAL = "general"
    FAST = "fast"


class ModelConfig(BaseModel):
    """Configuration for a specific task's model."""
    model: str
    fallbacks: list[str] = []
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Define which models to use for each task type.
# OpenRouter model format: "provider/model-name"
# 
# Popular models:
# - anthropic/claude-3.5-sonnet (fast, smart)
# - openai/gpt-4o (balanced)
# - openai/gpt-4o-mini (fast, cheap)
# - google/gemini-flash-1.5 (fast)
# - meta-llama/llama-3.1-70b-instruct (open source)
# - x-ai/grok-beta (fast)
# =============================================================================

MODEL_CONFIG: dict[str, ModelConfig] = {
    # --- Intent Classification ---
    # Fast model for quick routing decisions
    TaskType.INTENT_CLASSIFICATION: ModelConfig(
        model="openai/gpt-4o-mini",
        fallbacks=["google/gemini-flash-1.5", "anthropic/claude-3-haiku"],
        temperature=0.3,
        max_tokens=500,
    ),
    
    # --- Content Generation ---
    # Fast model for high-quality content
    TaskType.CONTENT_GENERATION: ModelConfig(
        model="x-ai/grok-beta",
        fallbacks=["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
        temperature=0.7,
        max_tokens=2000,
    ),
    
    # --- Hashtag & Timing Optimization ---
    # Fast model for structured output
    TaskType.HASHTAG_OPTIMIZATION: ModelConfig(
        model="x-ai/grok-beta",
        fallbacks=["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
        temperature=0.5,
        max_tokens=1000,
    ),
    
    TaskType.TIMING_OPTIMIZATION: ModelConfig(
        model="x-ai/grok-beta",
        fallbacks=["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
        temperature=0.3,
        max_tokens=500,
    ),
    
    # --- Visual Description ---
    # Creative model for visual recommendations
    TaskType.VISUAL_DESCRIPTION: ModelConfig(
        model="x-ai/grok-beta",
        fallbacks=["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
        temperature=0.7,
        max_tokens=1000,
    ),
    
    # --- SEO Tasks ---
    TaskType.SEO_ANALYSIS: ModelConfig(
        model="anthropic/claude-3.5-sonnet",
        fallbacks=["openai/gpt-4o"],
        temperature=0.5,
        max_tokens=3000,
    ),
    
    TaskType.KEYWORD_RESEARCH: ModelConfig(
        model="openai/gpt-4o-mini",
        fallbacks=["google/gemini-flash-1.5"],
        temperature=0.5,
        max_tokens=1500,
    ),
    
    # --- Copywriter Tasks ---
    TaskType.BLOG_WRITING: ModelConfig(
        model="anthropic/claude-3.5-sonnet",
        fallbacks=["openai/gpt-4o"],
        temperature=0.7,
        max_tokens=4000,
    ),
    
    TaskType.AD_COPY: ModelConfig(
        model="openai/gpt-4o",
        fallbacks=["anthropic/claude-3.5-sonnet"],
        temperature=0.8,
        max_tokens=1000,
    ),
    
    # --- Response Aggregation ---
    TaskType.RESPONSE_AGGREGATION: ModelConfig(
        model="openai/gpt-4o-mini",
        fallbacks=["google/gemini-flash-1.5"],
        temperature=0.5,
        max_tokens=2000,
    ),
    
    # --- General & Fast ---
    TaskType.GENERAL: ModelConfig(
        model="openai/gpt-4o",
        fallbacks=["anthropic/claude-3.5-sonnet", "x-ai/grok-beta"],
        temperature=0.7,
        max_tokens=4000,
    ),
    
    TaskType.FAST: ModelConfig(
        model="openai/gpt-4o-mini",
        fallbacks=["google/gemini-flash-1.5", "anthropic/claude-3-haiku"],
        temperature=0.7,
        max_tokens=2000,
    ),
}


def get_model_config(task: str | TaskType) -> ModelConfig:
    """
    Get model configuration for a specific task.
    
    Args:
        task: Task type (string or TaskType enum)
        
    Returns:
        ModelConfig with model, fallbacks, and settings
    """
    if isinstance(task, str):
        try:
            task = TaskType(task)
        except ValueError:
            # Unknown task, use general config
            return MODEL_CONFIG[TaskType.GENERAL]
    
    return MODEL_CONFIG.get(task, MODEL_CONFIG[TaskType.GENERAL])


def get_model_for_task(task: str | TaskType) -> str:
    """
    Get primary model name for a task.
    
    Args:
        task: Task type
        
    Returns:
        Model name in OpenRouter format (e.g., "openai/gpt-4o")
    """
    return get_model_config(task).model


def get_fallbacks_for_task(task: str | TaskType) -> list[str]:
    """
    Get fallback models for a task.
    
    Args:
        task: Task type
        
    Returns:
        List of fallback model names
    """
    return get_model_config(task).fallbacks
