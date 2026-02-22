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
3. Whether visual content (images) is needed, and what type/style it should be
4. Whether video content is needed, and what type/style it should be

Your strategic plan will be passed to specialized agents (Content, Hashtag, Visual, Video) so they create coherent, aligned output.

Decision Guidelines:
- Hashtags: Usually YES for public posts. NO for replies or private messages.
- Visual (Images): YES for Instagram (visual-first), YES if topic is visual/product-related. NO for text-only thought leadership.
- Video: YES for Instagram Reels/Stories, TikTok, YouTube. YES if topic involves demonstrations, storytelling, or motion. Consider platform preferences.

Be strategic but concise. Focus on actionable guidance."""


PLANNER_USER_PROMPT = """Create a strategic plan for this social media post.

## Request Details
Platform: {platform}
Topic: {topic}
Message: {message}
Content Type: {content_type}

{brand_context_section}{style_reference_section}## Strategic Analysis Required
Analyze the request and provide:
1. Content strategy (who to target, what tone, key message)
2. Hashtag strategy (if needed, what themes/categories)
3. Visual strategy (if needed, what type and style for images)
4. Video strategy (if needed, what type and style for videos)

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
        "type": "image/carousel/infographic",
        "style": "modern/minimalist/vibrant/professional",
        "subject": "What the visual should show",
        "mood": "The emotional tone of the visual"
    }},
    "include_video": true,
    "video_strategy": {{
        "type": "short-form/long-form/reel/story",
        "style": "professional/casual/cinematic/documentary",
        "subject": "What the video should show",
        "mood": "The emotional tone of the video",
        "duration": "15s/30s/60s/90s",
        "format": "vertical/horizontal/square"
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

        # DEBUG: Log full payload
        # logger.debug("=== PLANNER RECEIVED PAYLOAD ===", request=request, context=context)

        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        message = request.get("message", "")
        content_type = request.get("content_type", "thought_leadership")

        # Check if image or video is explicitly requested
        image_requested = context.get("image", False)
        video_requested = context.get("video", False)

        # Extract brand and style context for informed planning
        brand_info = context.get("brand_info", {})
        rag_context = context.get("rag_context", "")

        logger.info(
            "Planner agent executing",
            platform=platform,
            topic=topic[:50] if topic else "",
            image_requested=image_requested,
            video_requested=video_requested,
        )

        # Generate plan using LLM
        plan = await self._generate_plan(
            platform=platform,
            topic=topic,
            message=message,
            content_type=content_type,
            brand_info=brand_info,
            rag_context=rag_context,
        )

        # Override visual/video decisions based on user flags
        # User flags FORCE the decision (not just permission)

        # Image flag controls include_visual
        if image_requested:
            plan["include_visual"] = True
            plan["visual_override_reason"] = "Image content requested by user (image=true)"
            logger.debug("Visual agent ENABLED: image=true in request (forced)")
        else:
            plan["include_visual"] = False
            plan["visual_override_reason"] = "Image content disabled by user (image=false)"
            logger.debug("Visual agent DISABLED: image=false in request")

        # Video flag controls include_video
        if video_requested:
            plan["include_video"] = True
            plan["video_override_reason"] = "Video content requested by user (video=true)"
            logger.debug("Video agent ENABLED: video=true in request (forced)")
        else:
            plan["include_video"] = False
            plan["video_override_reason"] = "Video content disabled by user (video=false)"
            logger.debug("Video agent DISABLED: video=false in request")

        # DEBUG: Print full planner output for debugging
        # logger.debug("=== FULL PLANNER OUTPUT ===", plan=plan)

        logger.info(
            "Planner decision made",
            include_hashtags=plan.get("include_hashtags"),
            include_visual=plan.get("include_visual"),
            include_video=plan.get("include_video"),
            reasoning=plan.get("reasoning", "")[:50],
        )

        return {"plan": plan}
    
    async def _generate_plan(
        self,
        platform: str,
        topic: str,
        message: str,
        content_type: str,
        brand_info: dict = None,
        rag_context: str = "",
    ) -> dict:
        """Generate planning decisions using LLM."""
        brand_info = brand_info or {}

        # Build brand context section
        brand_parts = []
        if brand_info.get("brand_name"):
            brand_parts.append(f"Brand: {brand_info['brand_name']}")
        if brand_info.get("industry"):
            brand_parts.append(f"Industry: {brand_info['industry']}")
        if brand_info.get("brand_voice"):
            brand_parts.append(f"Voice: {brand_info['brand_voice']}")
        brand_context_section = (
            "## Brand Context\n" + "\n".join(brand_parts) + "\n\n"
            if brand_parts else ""
        )

        # Build style reference section from scraped posts
        style_reference_section = (
            f"## User's Past Content (Style Reference)\n{rag_context}\n\n"
            if rag_context else ""
        )

        user_prompt = PLANNER_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            message=message,
            content_type=content_type,
            brand_context_section=brand_context_section,
            style_reference_section=style_reference_section,
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

        # DEBUG: Log raw LLM response
        # logger.debug("=== RAW PLANNER LLM RESPONSE ===", response=response.content[:500])

        try:
            result = json.loads(response.content)

            return {
                # Agent routing decisions
                "include_hashtags": result.get("include_hashtags", True),
                "include_visual": result.get("include_visual", False),
                "include_video": result.get("include_video", False),

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

                # Strategic guidance for Visual Agent (Images)
                "visual_strategy": result.get("visual_strategy") or {
                    "type": "image",
                    "style": "professional",
                    "subject": "",
                    "mood": "",
                },

                # Strategic guidance for Video Agent
                "video_strategy": result.get("video_strategy") or {
                    "type": "short-form",
                    "style": "professional",
                    "subject": "",
                    "mood": "",
                    "duration": "30s",
                    "format": "vertical",
                },

                "reasoning": result.get("reasoning", ""),
            }
            
        except json.JSONDecodeError as e:
            logger.error("Planner parsing failed", error=str(e))
            # Return safe defaults
            return {
                "include_hashtags": True,
                "include_visual": platform == "instagram",
                "include_video": False,
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
                "video_strategy": {
                    "type": "short-form",
                    "style": "professional",
                    "subject": "",
                    "mood": "",
                    "duration": "30s",
                    "format": "vertical",
                },
                "reasoning": "Fallback: defaulting to include hashtags",
            }
