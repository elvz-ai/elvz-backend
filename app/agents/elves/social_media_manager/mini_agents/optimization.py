"""
Optimization Agent - Generates hashtags and optimal posting times.
Uses Grok for fast, combined optimization in a single LLM call.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog

from app.core.llm_clients import LLMMessage, LLMProvider, llm_client
from app.core.config import settings

logger = structlog.get_logger(__name__)


# Platform hashtag limits
HASHTAG_LIMITS = {
    "linkedin": 5,
    "twitter": 2,
    "instagram": 11,
    "facebook": 3,
}

# Best posting times by platform (fallback data)
BEST_TIMES = {
    "linkedin": {
        "best_hour": 10,
        "best_days": ["Tuesday", "Wednesday", "Thursday"],
        "notes": "Business hours, mid-week performs best",
    },
    "twitter": {
        "best_hour": 12,
        "best_days": ["Wednesday", "Thursday", "Friday"],
        "notes": "Lunch breaks and commute times",
    },
    "instagram": {
        "best_hour": 11,
        "best_days": ["Monday", "Wednesday", "Friday"],
        "notes": "Mid-morning, visual content peaks mid-week",
    },
    "facebook": {
        "best_hour": 13,
        "best_days": ["Wednesday", "Thursday", "Friday"],
        "notes": "Afternoon engagement, end-of-week activity",
    },
}


OPTIMIZATION_SYSTEM_PROMPT = """You are a social media optimization expert.
Your role is to recommend optimal hashtags and posting times for maximum engagement.

For hashtags:
- Balance popular (reach) and niche (relevance) tags
- Only use relevant, professional hashtags
- Consider the platform's hashtag culture

For timing:
- Consider the target audience's active hours
- Account for platform-specific engagement patterns
- Provide actionable recommendations"""


OPTIMIZATION_USER_PROMPT = """Optimize this {platform} post for maximum engagement.

## Post Content Summary
Topic: {topic}
Target Audience: {target_audience}
Current Time: {current_time}
Timezone: {timezone}

## Platform Info
- Hashtag Limit: {hashtag_limit}
- Best Times: {best_times}

## Strategic Direction (from Planner)
- Hashtag Focus: {focus}
- Themes: {themes}
- Strategy Notes: {notes}

Generate hashtags aligned with this strategy and optimal posting time.

Respond with valid JSON:
{{
    "hashtags": [
        {{"tag": "#hashtag1", "relevance": "high/medium/low"}},
        {{"tag": "#hashtag2", "relevance": "high/medium/low"}}
    ],
    "best_hour": 10,
    "best_days": ["Tuesday", "Wednesday"]
}}"""


class OptimizationAgent:
    """
    Optimization Agent for Social Media Manager.
    
    Combines hashtag research + timing optimization in a single fast LLM call.
    Uses Grok for speed and quality.
    """
    
    name = "optimization_agent"
    
    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any] = None,
        target_audience: str = None,
    ) -> Dict[str, Any]:
        """
        Generate hashtags and optimal posting time.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with hashtags and timing
        """
        request = state.get("user_request", {})
        # Get inputs
        user_request = state.get("user_request", {})
        platform = user_request.get("platform", "linkedin")
        topic = user_request.get("topic", "")
        
        # Target audience might come from Content Agent (if sequential) 
        # or be None (if parallel). We handle both.
        if not target_audience:
            content_output = state.get("content_output", {})
            target_audience = content_output.get("target_audience")
        
        timezone = context.get("timezone", "UTC")
        
        logger.info(
            "Optimization agent executing",
            platform=platform,
            timezone=timezone,
        )
        
        # Get platform-specific info
        hashtag_limit = HASHTAG_LIMITS.get(platform, 5)
        best_times = BEST_TIMES.get(platform, BEST_TIMES["linkedin"])
        
        # Get hashtag strategy from planner
        plan = state.get("plan") or {}
        hashtag_strategy = plan.get("hashtag_strategy") or {}
        
        # Generate optimization using Grok
        optimization = await self._generate_optimization(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            timezone=timezone,
            hashtag_limit=hashtag_limit,
            best_times=best_times,
            hashtag_strategy=hashtag_strategy,
        )
        
        # Build timing options from LLM result
        timing = self._build_timing_options(
            optimization.get("best_hour", best_times["best_hour"]),
            optimization.get("best_days", best_times["best_days"]),
            timezone,
            platform,
        )
        
        return {
            "hashtags": optimization.get("hashtags", []),
            "timing": timing,
        }
    
    async def _generate_optimization(
        self,
        platform: str,
        topic: str,
        target_audience: str,
        timezone: str,
        hashtag_limit: int,
        best_times: dict,
        hashtag_strategy: dict,
    ) -> dict:
        """Generate optimization using Grok."""
        
        hashtag_strategy = hashtag_strategy or {}
        
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        # Prepare prompt inputs
        # If target_audience not provided (parallel execution), let LLM infer it
        target_audience_ctx = target_audience or "Infer based on topic and platform"
        
        user_prompt = OPTIMIZATION_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            target_audience=target_audience_ctx,
            current_time=current_time,
            timezone=timezone,
            hashtag_limit=hashtag_limit,
            best_times=f"{best_times['best_days']} at {best_times['best_hour']}:00 - {best_times['notes']}",
            focus=hashtag_strategy.get("focus", "discovery"),
            themes=", ".join(hashtag_strategy.get("themes", [])),
            notes=hashtag_strategy.get("notes", ""),
        )
        
        messages = [
            LLMMessage(role="system", content=OPTIMIZATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        from app.core.model_config import TaskType
        
        # Use OpenRouter with task-based routing
        response = await llm_client.generate_for_task(
            task=TaskType.HASHTAG_OPTIMIZATION,
            messages=messages,
            json_mode=True,
        )
        
        try:
            result = json.loads(response.content)
            
            # Limit hashtags to platform max
            hashtags = result.get("hashtags", [])[:hashtag_limit]
            
            logger.debug("Optimization generated", hashtag_count=len(hashtags))
            
            return {
                "hashtags": hashtags,
                "best_hour": result.get("best_hour", 10),
                "best_days": result.get("best_days", ["Tuesday", "Wednesday"]),
            }
            
        except json.JSONDecodeError as e:
            logger.error("Optimization parsing failed", error=str(e))
            # Return fallback
            return {
                "hashtags": [],
                "best_hour": best_times["best_hour"],
                "best_days": best_times["best_days"],
            }
    
    def _build_timing_options(
        self,
        best_hour: int,
        best_days: list[str],
        timezone: str,
        platform: str,
    ) -> dict:
        """Build 3 timing options from LLM result."""
        
        now = datetime.utcnow()
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        # Ensure valid hour
        best_hour = max(0, min(23, best_hour))
        
        # Quick post: ~18 hours from now
        quick_post = now + timedelta(hours=18)
        quick_post = quick_post.replace(hour=best_hour, minute=0, second=0, microsecond=0)
        
        # Skip weekends for LinkedIn
        if platform == "linkedin":
            while quick_post.weekday() >= 5:
                quick_post += timedelta(days=1)
        
        # Optimal: Next occurrence of best day
        optimal_times = []
        for day_name in best_days[:2]:
            target_weekday = day_map.get(day_name, 1)
            days_ahead = target_weekday - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            
            next_occurrence = now + timedelta(days=days_ahead)
            next_occurrence = next_occurrence.replace(
                hour=best_hour, minute=0, second=0, microsecond=0
            )
            
            # Ensure at least 24 hours from now
            if (next_occurrence - now).total_seconds() < 24 * 3600:
                next_occurrence += timedelta(days=7)
            
            optimal_times.append(next_occurrence)
        
        optimal_times.sort()
        
        # Ensure we have at least 2 options
        while len(optimal_times) < 2:
            last = optimal_times[-1] if optimal_times else quick_post
            optimal_times.append(last + timedelta(days=2))
        
        return {
            "options": [
                {
                    "label": "Quick Post",
                    "description": "Post within 12-24 hours",
                    "datetime": quick_post.strftime("%Y-%m-%d %H:%M"),
                    "day_of_week": quick_post.strftime("%A"),
                },
                {
                    "label": "Optimal",
                    "description": "Best time this week",
                    "datetime": optimal_times[0].strftime("%Y-%m-%d %H:%M"),
                    "day_of_week": optimal_times[0].strftime("%A"),
                },
                {
                    "label": "Alternative",
                    "description": "Second best time",
                    "datetime": optimal_times[1].strftime("%Y-%m-%d %H:%M"),
                    "day_of_week": optimal_times[1].strftime("%A"),
                },
            ],
            "timezone": timezone,
            "confidence": 0.85,
        }
