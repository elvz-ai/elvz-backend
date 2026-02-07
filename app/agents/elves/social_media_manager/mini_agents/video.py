"""
Video Agent - Generates video content recommendations and scripts.
Creates strategic video content plans for social media platforms.
"""

import json
from typing import Optional

import structlog

from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


# Platform video specifications
PLATFORM_VIDEO_SPECS = {
    "instagram": {
        "reel": {"duration": "15-90s", "format": "vertical", "aspect": "9:16"},
        "story": {"duration": "15s", "format": "vertical", "aspect": "9:16"},
        "feed_video": {"duration": "3-60s", "format": "square/vertical", "aspect": "1:1 or 4:5"},
    },
    "tiktok": {
        "short": {"duration": "15-60s", "format": "vertical", "aspect": "9:16"},
    },
    "youtube": {
        "short": {"duration": "15-60s", "format": "vertical", "aspect": "9:16"},
        "standard": {"duration": "any", "format": "horizontal", "aspect": "16:9"},
    },
    "linkedin": {
        "feed_video": {"duration": "30-90s", "format": "horizontal/square", "aspect": "16:9 or 1:1"},
    },
    "twitter": {
        "tweet_video": {"duration": "up to 140s", "format": "horizontal/square", "aspect": "16:9 or 1:1"},
    },
    "facebook": {
        "feed_video": {"duration": "15-240s", "format": "square/vertical", "aspect": "1:1 or 4:5"},
        "story": {"duration": "15s", "format": "vertical", "aspect": "9:16"},
    },
}

# Video best practices per platform
VIDEO_TIPS = {
    "instagram": """- Reels perform best with trending audio
- Hook viewers in first 3 seconds
- Captions essential (85% watch without sound)
- Vertical format is ideal (9:16)
- Include CTA in first comment""",

    "tiktok": """- Start with a strong hook
- Use trending sounds/effects
- Keep it authentic and relatable
- Text overlays increase engagement
- Post at peak times (6-10pm)""",

    "linkedin": """- Professional, value-driven content
- First 3 seconds critical for retention
- Captions recommended
- Educational/thought leadership performs best
- Shorter videos (30-90s) have better completion rates""",

    "twitter": """- Grab attention immediately
- Keep it concise and punchy
- Native video performs better than links
- Use captions for accessibility
- GIFs work well for reactions""",

    "facebook": """- Native video prioritized by algorithm
- Square or vertical for mobile
- Emotional storytelling resonates
- Live videos get 6x more engagement
- Add captions (most watch without sound)""",

    "youtube": """- Shorts: Hook in first second, vertical format
- Standard: Strong intro, clear value proposition
- Optimize title and thumbnail
- Include timestamps for longer videos
- End screens for viewer retention""",
}


VIDEO_SYSTEM_PROMPT = """You are a video content strategist for social media.
Your role is to create strategic video content recommendations including scripts, storyboards, and technical specifications.

Provide specific, actionable video recommendations including:
- Video type and format
- Script/storyboard outline
- Visual and audio elements
- Platform-specific optimizations
- Technical specifications"""


VIDEO_USER_PROMPT = """Create a video content plan for this {platform} post.

## Post Details
Topic: {topic}
Target Audience: {target_audience}
Platform: {platform}

## Platform Best Practices
{video_tips}

## Strategic Direction (from Planner)
- Video Style: {style}
- Video Type: {type}
- Subject: {subject}
- Mood: {mood}
- Duration: {duration}
- Format: {format}

Generate a comprehensive video plan aligned with this strategy.

Respond with valid JSON:
{{
    "type": "reel/story/short/standard",
    "format": "vertical/horizontal/square",
    "duration": "recommended duration",
    "hook": "First 3 seconds hook to grab attention",
    "script_outline": [
        {{"timestamp": "0-3s", "action": "description"}},
        {{"timestamp": "3-10s", "action": "description"}},
        {{"timestamp": "10-30s", "action": "description"}}
    ],
    "visual_elements": ["element1", "element2"],
    "audio_elements": {{
        "music": "recommended music style/mood",
        "voiceover": "yes/no",
        "sound_effects": ["effect1", "effect2"]
    }},
    "text_overlays": ["text1", "text2"],
    "call_to_action": "CTA for the video",
    "technical_specs": {{
        "aspect_ratio": "9:16",
        "resolution": "1080x1920",
        "fps": "30"
    }}
}}"""


class VideoAgent:
    """
    Video Agent for Social Media Manager.

    Generates video content recommendations, scripts, and storyboards.
    Actual video generation can be integrated later.
    """

    name = "video_agent"

    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate video content recommendations.

        Args:
            state: Current workflow state
            context: Execution context

        Returns:
            Updated state with video recommendations
        """
        request = state.get("user_request") or {}
        content = state.get("content") or {}

        platform = request.get("platform", "instagram")
        topic = request.get("topic", "")
        target_audience = content.get("target_audience", "general audience")

        logger.info(
            "Video agent executing",
            platform=platform,
        )

        # Get platform-specific video info
        video_tips = VIDEO_TIPS.get(platform, VIDEO_TIPS["instagram"])
        video_specs = PLATFORM_VIDEO_SPECS.get(platform, PLATFORM_VIDEO_SPECS["instagram"])

        # Get video strategy from planner
        plan = state.get("plan") or {}
        video_strategy = plan.get("video_strategy") or {}

        # Generate video recommendation
        video_advice = await self._generate_video_plan(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            video_tips=video_tips,
            video_specs=video_specs,
            video_strategy=video_strategy,
        )

        return {"video_advice": video_advice}

    async def _generate_video_plan(
        self,
        platform: str,
        topic: str,
        target_audience: str,
        video_tips: str,
        video_specs: dict,
        video_strategy: dict,
    ) -> dict:
        """Generate video plan using LLM."""

        video_strategy = video_strategy or {}

        user_prompt = VIDEO_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            video_tips=video_tips,
            style=video_strategy.get("style", "professional"),
            type=video_strategy.get("type", "short-form"),
            subject=video_strategy.get("subject", topic),
            mood=video_strategy.get("mood", "engaging"),
            duration=video_strategy.get("duration", "30s"),
            format=video_strategy.get("format", "vertical"),
        )

        messages = [
            LLMMessage(role="system", content=VIDEO_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        from app.core.model_config import TaskType

        # Use task-based routing for video content creation
        response = await llm_client.generate_for_task(
            task=TaskType.CONTENT_CREATION,
            messages=messages,
            json_mode=True,
        )

        try:
            # Add null check for response
            if response is None or not response.content:
                logger.warning("Video generation returned empty response")
                raise ValueError("Empty response from LLM")

            result = json.loads(response.content)

            # Add null check for parsed result
            if result is None or not isinstance(result, dict):
                logger.warning("Video generation returned null or non-dict result", result=result)
                raise ValueError("Invalid result format")

            # Get default video spec for platform
            default_spec = list(video_specs.values())[0] if video_specs else {
                "duration": "30s",
                "format": "vertical",
                "aspect": "9:16"
            }

            video_advice = {
                "type": result.get("type", "short-form"),
                "format": result.get("format", default_spec.get("format", "vertical")),
                "duration": result.get("duration", default_spec.get("duration", "30s")),
                "hook": result.get("hook", ""),
                "script_outline": result.get("script_outline", []),
                "visual_elements": result.get("visual_elements", []),
                "audio_elements": result.get("audio_elements", {}),
                "text_overlays": result.get("text_overlays", []),
                "call_to_action": result.get("call_to_action", ""),
                "technical_specs": result.get("technical_specs", {
                    "aspect_ratio": default_spec.get("aspect", "9:16"),
                    "resolution": "1080x1920" if default_spec.get("format") == "vertical" else "1920x1080",
                    "fps": "30",
                }),
                "platform": platform,
                "generation_status": "script_generated",
            }

            logger.debug("Video plan generated", type=video_advice["type"])

            return video_advice

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Video parsing failed", error=str(e))
            # Return fallback
            default_spec = list(video_specs.values())[0] if video_specs else {
                "duration": "30s",
                "format": "vertical",
                "aspect": "9:16"
            }

            return {
                "type": "short-form",
                "format": default_spec.get("format", "vertical"),
                "duration": default_spec.get("duration", "30s"),
                "hook": f"Engaging hook about {topic}",
                "script_outline": [
                    {"timestamp": "0-3s", "action": "Strong visual hook"},
                    {"timestamp": "3-15s", "action": f"Present key information about {topic}"},
                    {"timestamp": "15-30s", "action": "Call to action"},
                ],
                "visual_elements": ["Professional visuals", "On-brand graphics"],
                "audio_elements": {
                    "music": "Upbeat, engaging background music",
                    "voiceover": "yes",
                    "sound_effects": [],
                },
                "text_overlays": [f"Key points about {topic}"],
                "call_to_action": "Engage with this content",
                "technical_specs": {
                    "aspect_ratio": default_spec.get("aspect", "9:16"),
                    "resolution": "1080x1920",
                    "fps": "30",
                },
                "platform": platform,
                "generation_status": "script_generated",
            }
