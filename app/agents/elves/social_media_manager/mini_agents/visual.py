"""
Visual Agent - Generates visual content recommendations.
Uses Grok for description generation (image generation to be added later).
Simplified for speed - no RAG/Pinecone dependency.
"""

import json
from typing import Optional

import structlog

from app.core.llm_clients import LLMMessage, LLMProvider, llm_client
from app.core.config import settings

logger = structlog.get_logger(__name__)


# Platform dimensions
PLATFORM_DIMENSIONS = {
    "linkedin": {
        "feed_post": "1200 x 627",
        "carousel": "1080 x 1080",
    },
    "twitter": {
        "tweet_image": "1200 x 675",
    },
    "instagram": {
        "feed_square": "1080 x 1080",
        "feed_portrait": "1080 x 1350",
        "story": "1080 x 1920",
    },
    "facebook": {
        "feed_image": "1200 x 630",
    },
}

# Visual best practices (fallback knowledge)
VISUAL_TIPS = {
    "linkedin": """- Professional imagery performs best
- Include faces for 38% more engagement
- Blue tones align with platform aesthetics
- Infographics for data-driven content""",
    
    "twitter": """- Bold, high-contrast visuals stand out
- Keep text on images minimal
- GIFs for reactions and engagement
- Memes work well for engagement""",
    
    "instagram": """- High-quality, aesthetic imagery essential
- Consistent visual theme/filter
- Reels outperform static images
- Behind-the-scenes resonates""",
    
    "facebook": """- Native video gets priority
- Emotional imagery performs best
- Text overlays should be minimal
- Live videos have highest engagement""",
}


VISUAL_SYSTEM_PROMPT = """You are a visual content strategist for social media.
Your role is to recommend visual elements that complement written content.

Provide specific, actionable visual recommendations including:
- Type of visual (image, carousel, video, infographic)
- Description of what the visual should contain
- Style and color recommendations
- Platform-specific considerations"""


VISUAL_USER_PROMPT = """Recommend visual content for this {platform} post.

## Post Details
Topic: {topic}
Target Audience: {target_audience}
Platform: {platform}

## Platform Best Practices
{visual_tips}

## Brand Colors (if any)
{brand_colors}

## Strategic Direction (from Planner)
- Visual Style: {style}
- Visual Type: {type}
- Subject: {subject}
- Mood: {mood}

Generate a visual recommendation aligned with this strategy.

Respond with valid JSON:
{{
    "type": "image/carousel/video/infographic",
    "description": "Detailed description of the visual content",
    "style": "Style notes (colors, mood, aesthetic)",
    "dimensions": "Recommended dimensions"
}}"""


class VisualAgent:
    """
    Visual Agent for Social Media Manager.
    
    Generates visual content descriptions using Grok.
    Image generation can be added later with a separate model.
    """
    
    name = "visual_agent"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate visual content recommendations.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with visual advice
        """
        request = state.get("user_request") or {}
        content = state.get("content") or {}

        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        target_audience = content.get("target_audience", "professionals")
        
        logger.info(
            "Visual agent executing",
            platform=platform,
        )
        
        # Get platform-specific info
        visual_tips = VISUAL_TIPS.get(platform, VISUAL_TIPS["linkedin"])
        dimensions = PLATFORM_DIMENSIONS.get(platform, {})
        
        # Get brand colors if available
        context = context or {}
        brand_info = context.get("brand_info") or {}
        brand_colors = brand_info.get("brand_colors", "No specific brand colors")
        
        # Get visual strategy from planner
        plan = state.get("plan") or {}
        visual_strategy = plan.get("visual_strategy") or {}
        
        # Generate visual recommendation using Grok
        visual_advice = await self._generate_visual(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            visual_tips=visual_tips,
            brand_colors=brand_colors,
            dimensions=dimensions,
            visual_strategy=visual_strategy,
        )
        
        return {"visual_advice": visual_advice}
    
    async def _generate_visual(
        self,
        platform: str,
        topic: str,
        target_audience: str,
        visual_tips: str,
        brand_colors: str,
        dimensions: dict,
        visual_strategy: dict,
    ) -> dict:
        """Generate visual recommendation using Grok."""
        
        visual_strategy = visual_strategy or {}
        
        user_prompt = VISUAL_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            visual_tips=visual_tips,
            brand_colors=brand_colors,
            style=visual_strategy.get("style", "Professional"),
            type=visual_strategy.get("type", "Image"),
            subject=visual_strategy.get("subject", topic),
            mood=visual_strategy.get("mood", "Professional"),
        )
        
        messages = [
            LLMMessage(role="system", content=VISUAL_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        from app.core.model_config import TaskType
        
        # Use OpenRouter with task-based routing
        response = await llm_client.generate_for_task(
            task=TaskType.VISUAL_DESCRIPTION,
            messages=messages,
            json_mode=True,
        )
        
        try:
            # Add null check for response
            if response is None or not response.content:
                logger.warning("Visual generation returned empty response")
                raise ValueError("Empty response from LLM")

            result = json.loads(response.content)

            # Add null check for parsed result
            if result is None or not isinstance(result, dict):
                logger.warning("Visual generation returned null or non-dict result", result=result)
                raise ValueError("Invalid result format")

            # Get default dimension for platform
            default_dim = list(dimensions.values())[0] if dimensions else "1200 x 630"

            visual_advice = {
                "type": result.get("type", "image"),
                "description": result.get("description", ""),
                "style": result.get("style", ""),
                "dimensions": result.get("dimensions", default_dim),
                "platform": platform,
                # Placeholder for future image generation
                "image_url": None,
                "generation_status": "description_only",
            }

            logger.debug("Visual recommendation generated", type=visual_advice["type"])
            return visual_advice

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Visual parsing failed", error=str(e))
            # Return fallback
            return {
                "type": "image",
                "description": f"Professional visual related to {topic}",
                "style": "Clean, modern, professional",
                "dimensions": list(dimensions.values())[0] if dimensions else "1200 x 630",
                "platform": platform,
                "image_url": None,
                "generation_status": "description_only",
            }
