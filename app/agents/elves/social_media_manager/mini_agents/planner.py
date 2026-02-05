"""
Planner Agent - Analyzes request and decides which agents to invoke.
Runs FIRST to make intelligent decisions about downstream agent execution.
"""

import json
from typing import Any, Dict

import structlog

from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a social media strategist and planning assistant.
Your role is to analyze a user's request and create a strategic plan that guides all content creation.

You will provide:
1. Strategic direction for the post (target audience, tone, key message)
2. Whether hashtags are needed, and what themes/categories they should focus on
3. Whether visual content is needed, and what type/style it should be

Your strategic plan will be passed to specialized agents (Content, Hashtag, Visual) so they create coherent, aligned output.

Decision Guidelines:
- Hashtags: Usually YES for public posts. NO for replies or private messages.
- Visual: YES for Instagram (visual-first), YES if topic is visual/product-related. NO for text-only thought leadership.

Be strategic but concise. Focus on actionable guidance."""


PLANNER_USER_PROMPT = """Create a strategic plan for this social media post.

## Request Details
Platform: {platform}
Topic: {topic}
Message: {message}
Content Type: {content_type}

## Strategic Analysis Required
Analyze the request and provide:
1. Content strategy (who to target, what tone, key message)
2. Hashtag strategy (if needed, what themes/categories)
3. Visual strategy (if needed, what type and style)

Respond with valid JSON:
{{
    "content_strategy": {{
        "target_audience": "Who this post is for",
        "tone": "professional/conversational/inspirational/educational",
        "key_message": "The core message or takeaway",
        "content_angle": "The specific angle or hook to use"
    }},
    "include_hashtags": true,
    "hashtag_strategy": {{
        "themes": ["theme1", "theme2"],
        "focus": "discovery/community/trending",
        "notes": "Any specific guidance for hashtag selection"
    }},
    "include_visual": true,
    "visual_strategy": {{
        "type": "image/carousel/video/infographic",
        "style": "modern/minimalist/vibrant/professional",
        "subject": "What the visual should show",
        "mood": "The emotional tone of the visual"
    }},
    "reasoning": "Brief explanation of strategic decisions"
}}"""


class PlannerAgent:
    """
    Planner Agent for Social Media Manager.
    
    Runs FIRST to analyze the request and decide which downstream
    agents should be invoked, optimizing LLM calls and execution time.
    """
    
    name = "planner_agent"
    
    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze request and decide which agents to invoke.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Planning decisions
        """
        request = state.get("user_request", {})
        context = context or {}
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        message = request.get("message", "")
        content_type = request.get("content_type", "thought_leadership")
        
        # Check if media is explicitly requested
        media_requested = context.get("media", False)
        
        logger.info(
            "Planner agent executing",
            platform=platform,
            media_requested=media_requested,
        )
        
        # Generate plan using LLM
        plan = await self._generate_plan(
            platform=platform,
            topic=topic,
            message=message,
            content_type=content_type,
        )
        
        # Override visual decision if media=true is explicitly set
        if media_requested:
            plan["include_visual"] = True
            plan["visual_override_reason"] = "Media explicitly requested by user"
            logger.debug("Visual agent forced: media=true in request")
        
        logger.info(
            "Planner decision made",
            include_hashtags=plan.get("include_hashtags"),
            include_visual=plan.get("include_visual"),
            reasoning=plan.get("reasoning", "")[:50],
        )
        
        return {"plan": plan}
    
    async def _generate_plan(
        self,
        platform: str,
        topic: str,
        message: str,
        content_type: str,
    ) -> dict:
        """Generate planning decisions using LLM."""
        
        user_prompt = PLANNER_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            message=message,
            content_type=content_type,
        )
        
        messages = [
            LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        from app.core.model_config import TaskType
        
        # Use fast model for planning (quick decision)
        response = await llm_client.generate_for_task(
            task=TaskType.INTENT_CLASSIFICATION,  # Use fast model
            messages=messages,
            json_mode=True,
        )
        
        try:
            result = json.loads(response.content)
            
            return {
                # Agent routing decisions
                "include_hashtags": result.get("include_hashtags", True),
                "include_visual": result.get("include_visual", False),
                
                # Strategic guidance for Content Agent
                "content_strategy": result.get("content_strategy") or {
                    "target_audience": "professionals",
                    "tone": "professional",
                    "key_message": "",
                    "content_angle": "",
                },
                
                # Strategic guidance for Optimization Agent
                "hashtag_strategy": result.get("hashtag_strategy") or {
                    "themes": [],
                    "focus": "discovery",
                    "notes": "",
                },
                
                # Strategic guidance for Visual Agent
                "visual_strategy": result.get("visual_strategy") or {
                    "type": "image",
                    "style": "professional",
                    "subject": "",
                    "mood": "",
                },
                
                "reasoning": result.get("reasoning", ""),
            }
            
        except json.JSONDecodeError as e:
            logger.error("Planner parsing failed", error=str(e))
            # Return safe defaults
            return {
                "include_hashtags": True,
                "include_visual": platform == "instagram",
                "content_strategy": {
                    "target_audience": "professionals",
                    "tone": "professional",
                    "key_message": "",
                    "content_angle": "",
                },
                "hashtag_strategy": {
                    "themes": [],
                    "focus": "discovery",
                    "notes": "",
                },
                "visual_strategy": {
                    "type": "image",
                    "style": "professional",
                    "subject": "",
                    "mood": "",
                },
                "reasoning": "Fallback: defaulting to include hashtags",
            }
