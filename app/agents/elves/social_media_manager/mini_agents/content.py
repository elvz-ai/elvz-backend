"""
Content Agent - Generates social media content with integrated strategy.
Uses Grok for fast, high-quality content generation.
Combines strategy + content in a single LLM call for speed.
"""

import json
from typing import Optional

import structlog

from app.core.llm_clients import LLMMessage, LLMProvider, llm_client
from app.core.config import settings

logger = structlog.get_logger(__name__)


# Platform character limits
PLATFORM_LIMITS = {
    "linkedin": 3000,
    "twitter": 280,
    "instagram": 2200,
    "facebook": 63206,
}

# Platform-specific best practices (fallback knowledge)
PLATFORM_TIPS = {
    "linkedin": """- Professional tone works best
- Hook in first 2 lines (before "see more")
- Use line breaks for readability
- End with question or CTA for engagement""",
    
    "twitter": """- Concise, punchy messages
- Use threads for longer content
- Engage with trending topics
- Visual content gets more engagement""",
    
    "instagram": """- Visual-first platform
- Captions can be longer, but hook matters
- Use line breaks and emojis
- Stories and Reels have highest reach""",
    
    "facebook": """- Longer form content works
- Native video gets priority
- Community engagement important
- Ask questions to drive comments""",
}


CONTENT_SYSTEM_PROMPT = """You are an expert social media content creator.
Your role is to create engaging, platform-optimized content that drives engagement.

You will generate ONE high-quality post that includes:
1. Strategic thinking (tone, target audience, key message)
2. The actual post content with a strong hook
3. A clear call-to-action

Guidelines:
- Write in a natural, authentic voice
- Create a compelling hook in the first line
- Make the content scannable with line breaks
- Include a clear call-to-action
- Stay within platform character limits
- Be concise but impactful"""


CONTENT_USER_PROMPT = """Create a {platform} post about: {topic}

## Platform Details
- Platform: {platform}
- Character Limit: {char_limit}
- Content Type: {content_type}
- Goals: {goals}

## Strategic Direction (from Planner)
- Target Audience: {target_audience}
- Tone: {tone}
- Key Message: {key_message}
- Content Angle: {content_angle}

## Platform Best Practices
{platform_tips}

## Brand Context
{brand_context}

{style_reference}Generate engaging content aligned with the strategic direction above. If style reference posts are provided above, match the user's writing style, vocabulary, and tone.

Respond with valid JSON:
{{
    "post_text": "The complete post content ready to publish",
    "hook": "The attention-grabbing first line",
    "cta": "The call-to-action used",
    "target_audience": "Who this content is for",
    "tone": "The tone used (e.g., professional, conversational)"
}}"""


class ContentAgent:
    """
    Content Agent for Social Media Manager.
    
    Combines strategy + content generation in a single fast LLM call.
    Uses Grok for speed and quality.
    """
    
    name = "content_agent"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate social media content.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with content
        """
        request = state.get("user_request", {})
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        content_type = request.get("content_type", "thought_leadership")
        goals = request.get("goals", ["engagement"])
        
        logger.info(
            "Content agent executing",
            platform=platform,
            topic=topic[:50] if topic else "",
        )
        
        # Get brand context
        brand_context = self._get_brand_context(context)

        # Get RAG context (user's scraped social posts for style matching)
        rag_context = context.get("rag_context", "")

        # Get conversation history for modification context
        conversation_history = context.get("conversation_history", "")

        # Get platform tips
        platform_tips = PLATFORM_TIPS.get(platform, PLATFORM_TIPS["linkedin"])

        # Get content strategy from planner
        plan = state.get("plan") or {}
        content_strategy = plan.get("content_strategy") or {}

        # Generate content using Grok
        content = await self._generate_content(
            platform=platform,
            topic=topic,
            content_type=content_type,
            goals=goals,
            brand_context=brand_context,
            platform_tips=platform_tips,
            content_strategy=content_strategy,
            rag_context=rag_context,
            conversation_history=conversation_history,
        )
        
        return {"content": content}
    
    def _get_brand_context(self, context: dict) -> str:
        """Build brand context from user info."""
        brand_info = context.get("brand_info", {})
        
        if not brand_info:
            return "No specific brand context. Use professional, engaging tone."
        
        parts = []
        if brand_info.get("brand_name"):
            parts.append(f"Brand: {brand_info['brand_name']}")
        if brand_info.get("industry"):
            parts.append(f"Industry: {brand_info['industry']}")
        if brand_info.get("brand_voice"):
            parts.append(f"Voice: {brand_info['brand_voice']}")
        if brand_info.get("target_audience"):
            parts.append(f"Target Audience: {brand_info['target_audience']}")
        
        return "\n".join(parts) if parts else "No specific brand context."
    
    async def _generate_content(
        self,
        platform: str,
        topic: str,
        content_type: str,
        goals: list[str],
        brand_context: str,
        platform_tips: str,
        content_strategy: dict,
        rag_context: str = "",
        conversation_history: str = "",
    ) -> dict:
        """Generate content using Grok."""

        content_strategy = content_strategy or {}

        char_limit = PLATFORM_LIMITS.get(platform, 3000)

        # Build style reference section from scraped social posts
        style_reference = ""
        if rag_context:
            style_reference = f"## Style Reference (User's Past Content)\n{rag_context}\n\n"

        # Prepend conversation history for modification context
        conversation_prefix = ""
        if conversation_history:
            conversation_prefix = (
                f"## Recent Conversation Context\n{conversation_history}\n\n"
            )

        user_prompt = conversation_prefix + CONTENT_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            char_limit=char_limit,
            content_type=content_type,
            goals=", ".join(goals) if isinstance(goals, list) else goals,
            platform_tips=platform_tips,
            brand_context=brand_context,
            style_reference=style_reference,
            target_audience=content_strategy.get("target_audience", "General audience"),
            tone=content_strategy.get("tone", "Professional"),
            key_message=content_strategy.get("key_message", f"About {topic}"),
            content_angle=content_strategy.get("content_angle", "Informative"),
        )
        
        messages = [
            LLMMessage(role="system", content=CONTENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        from app.core.model_config import TaskType
        
        # Use OpenRouter with task-based routing
        response = await llm_client.generate_for_task(
            task=TaskType.CONTENT_GENERATION,
            messages=messages,
            json_mode=True,
        )
        
        try:
            result = json.loads(response.content)
            
            content = {
                "post_text": result.get("post_text", ""),
                "hook": result.get("hook", ""),
                "cta": result.get("cta", ""),
                "target_audience": result.get("target_audience", ""),
                "tone": result.get("tone", ""),
                "platform": platform,
                "character_count": len(result.get("post_text", "")),
            }
            
            logger.debug("Content generated", char_count=content["character_count"])
            return content
            
        except json.JSONDecodeError as e:
            logger.error("Content parsing failed", error=str(e))
            # Return raw content as fallback
            return {
                "post_text": response.content,
                "hook": "",
                "cta": "",
                "platform": platform,
                "character_count": len(response.content),
            }
