"""
Timing Optimizer Agent - Calculates optimal posting times.
Uses Tool-Augmented Generation with analytics data.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from app.core.llm_clients import LLMMessage, llm_client
from app.core.config import settings
from app.tools.registry import tool_registry

logger = structlog.get_logger(__name__)


TIMING_SYSTEM_PROMPT = """You are a social media timing optimization specialist.
Your role is to recommend optimal posting times for maximum engagement.

## Expertise
- Platform-specific engagement patterns
- Timezone and audience location analysis
- Industry-specific timing patterns
- Content type timing optimization

## Guidelines
- Consider the target audience's active hours
- Account for different days of the week
- Balance between competition and visibility
- Provide specific time recommendations with reasoning"""


TIMING_USER_PROMPT_SIMPLE = """Based on the analytics data, what is the BEST time of day and day of week to post on {platform} for {content_type} content?

## Analytics/Tool Data
{tool_results}

Just tell me the best hour (0-23) and best days of the week. Respond with valid JSON:
{{
    "best_hour": 10,
    "best_days": ["Tuesday", "Wednesday", "Thursday"],
    "confidence": 0.0-1.0
}}"""


TIMING_USER_PROMPT_DETAILED = """Based on the analytics data, what is the BEST time of day and day of week to post on {platform} for {content_type} content targeting {target_audience}?

## Analytics/Tool Data
{tool_results}

Respond with valid JSON:
{{
    "best_hour": 10,
    "best_days": ["Tuesday", "Wednesday", "Thursday"],
    "confidence": 0.0-1.0,
    "reasoning": "why these times are optimal"
}}"""


class TimingOptimizerAgent:
    """
    Timing Optimizer Agent for Social Media Manager.
    
    Method: Tool-Augmented Generation
    - Uses analytics tools for user-specific data
    - Considers platform patterns and audience behavior
    - Provides timezone-aware recommendations
    """
    
    name = "timing_optimizer_agent"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Calculate optimal posting time.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with timing recommendation
        """
        request = state.get("user_request", {})
        strategy = state.get("strategy", {})
        
        platform = request.get("platform", "linkedin")
        content_type = request.get("content_type", "thought_leadership")
        user_id = context.get("user_id", "")
        timezone = context.get("timezone", "UTC")
        
        logger.info(
            "Timing optimizer executing",
            platform=platform,
            timezone=timezone,
        )
        
        # Get analytics/timing data from tools
        tool_results = await self._get_timing_data(user_id, platform, timezone)
        
        # Generate recommendation
        timing = await self._generate_recommendation(
            platform=platform,
            content_type=content_type,
            target_audience=strategy.get("target_audience", "professionals"),
            timezone=timezone,
            tool_results=tool_results,
        )
        
        return {"timing": timing}
    
    async def _get_timing_data(
        self,
        user_id: str,
        platform: str,
        timezone: str,
    ) -> str:
        """Get timing data from tools."""
        try:
            from app.tools.social_media_tools import OptimalTimingInput
            
            tool = tool_registry.get("optimal_timing")
            if tool:
                input_data = OptimalTimingInput(
                    user_id=user_id,
                    platform=platform,
                    timezone=timezone,
                )
                result = await tool.execute(input_data)
                
                if result.success and result.data:
                    best_times = result.data.get("best_times", [])
                    formatted = []
                    for t in best_times:
                        formatted.append(
                            f"- {t.get('day', 'Unknown')} at {t.get('time', '00:00')}: "
                            f"score={t.get('score', 0):.2f}"
                        )
                    
                    formatted.append(f"\nReasoning: {result.data.get('reasoning', 'N/A')}")
                    return "\n".join(formatted)
        
        except Exception as e:
            logger.warning("Timing tool failed", error=str(e))
        
        # Return platform-specific defaults
        return self._get_fallback_data(platform)
    
    def _get_fallback_data(self, platform: str) -> str:
        """Get fallback timing data."""
        platform_times = {
            "linkedin": """Best times for LinkedIn:
- Tuesday at 10:00 AM: score=0.95 (peak professional activity)
- Wednesday at 9:00 AM: score=0.92 (mid-week engagement)
- Thursday at 2:00 PM: score=0.88 (afternoon check-ins)
Note: Business hours perform best, avoid weekends""",
            
            "twitter": """Best times for Twitter:
- Wednesday at 12:00 PM: score=0.93 (lunch break engagement)
- Thursday at 9:00 AM: score=0.90 (morning scroll)
- Friday at 11:00 AM: score=0.87 (end-of-week activity)
Note: Real-time engagement matters most""",
            
            "instagram": """Best times for Instagram:
- Monday at 11:00 AM: score=0.94 (week start motivation)
- Wednesday at 7:00 PM: score=0.91 (evening browsing)
- Friday at 10:00 AM: score=0.89 (weekend preview)
Note: Visual content performs best mid-week""",
            
            "facebook": """Best times for Facebook:
- Wednesday at 1:00 PM: score=0.92 (peak engagement)
- Thursday at 2:00 PM: score=0.89 (afternoon activity)
- Friday at 10:00 AM: score=0.86 (weekend planning)
Note: Video content gets more reach""",
        }
        
        return platform_times.get(platform, platform_times["linkedin"])
    
    async def _generate_recommendation(
        self,
        platform: str,
        content_type: str,
        target_audience: str,
        timezone: str,
        tool_results: str,
    ) -> dict:
        """
        Generate 3 timing recommendations:
        1. Quick post: Within 12-24 hours
        2. Optimal: Within 1 week (best day/time)
        3. Alternative: Within 1 week (second best day/time)
        """
        
        # Get current date/time
        now = datetime.utcnow()
        
        # Use simple or detailed prompt based on REASONING setting
        prompt_template = TIMING_USER_PROMPT_DETAILED if settings.include_reasoning else TIMING_USER_PROMPT_SIMPLE
        
        user_prompt = prompt_template.format(
            platform=platform,
            content_type=content_type,
            target_audience=target_audience,
            tool_results=tool_results,
        )
        
        messages = [
            LLMMessage(role="system", content=TIMING_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            
            best_hour = result.get("best_hour", 10)
            best_days = result.get("best_days", ["Tuesday", "Wednesday", "Thursday"])
            confidence = result.get("confidence", 0.8)
            reasoning = result.get("reasoning", "") if settings.include_reasoning else ""
            
            # Generate 3 times using Python logic
            times = self._generate_three_times(
                now=now,
                platform=platform,
                best_hour=best_hour,
                best_days=best_days,
            )
            
            # Build timing response with 3 options
            timing = {
                "options": [
                    {
                        "label": "Quick Post",
                        "description": "Post within 12-24 hours",
                        "datetime": times[0].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[0].strftime("%A"),
                    },
                    {
                        "label": "Optimal",
                        "description": "Best time this week",
                        "datetime": times[1].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[1].strftime("%A"),
                    },
                    {
                        "label": "Alternative",
                        "description": "Second best time this week",
                        "datetime": times[2].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[2].strftime("%A"),
                    },
                ],
                "timezone": timezone,
                "confidence": confidence,
            }
            
            # Add reasoning if enabled
            if settings.include_reasoning and reasoning:
                timing["reason"] = reasoning
            
            logger.debug("Timing recommended", options=len(timing["options"]))
            return timing
            
        except json.JSONDecodeError as e:
            logger.error("Timing parsing failed", error=str(e))
            # Return fallback with 3 times
            times = self._generate_three_times(
                now=now,
                platform=platform,
                best_hour=10,
                best_days=["Tuesday", "Wednesday", "Thursday"],
            )
            
            return {
                "options": [
                    {
                        "label": "Quick Post",
                        "description": "Post within 12-24 hours",
                        "datetime": times[0].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[0].strftime("%A"),
                    },
                    {
                        "label": "Optimal",
                        "description": "Best time this week",
                        "datetime": times[1].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[1].strftime("%A"),
                    },
                    {
                        "label": "Alternative",
                        "description": "Second best time this week",
                        "datetime": times[2].strftime("%Y-%m-%d %H:%M"),
                        "day_of_week": times[2].strftime("%A"),
                    },
                ],
                "timezone": timezone,
                "confidence": 0.7,
            }
    
    def _generate_three_times(
        self,
        now: datetime,
        platform: str,
        best_hour: int,
        best_days: list[str],
    ) -> list[datetime]:
        """
        Generate 3 posting times:
        1. Within 12-24 hours (quick post)
        2. Best day/time within 1 week (optimal)
        3. Second best day/time within 1 week (alternative)
        """
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        # Ensure best_hour is valid
        best_hour = max(0, min(23, best_hour))
        
        # --- TIME 1: Quick post (12-24 hours from now) ---
        quick_post = now + timedelta(hours=18)  # ~18 hours from now
        quick_post = quick_post.replace(minute=0, second=0, microsecond=0)
        
        # Adjust to best hour if reasonable
        if 6 <= best_hour <= 22:
            # If the best hour is reasonable, try to use it
            if quick_post.hour < best_hour:
                quick_post = quick_post.replace(hour=best_hour)
            elif quick_post.hour > best_hour:
                # Push to next day at best hour
                quick_post = (quick_post + timedelta(days=1)).replace(hour=best_hour)
        
        # Skip weekends for LinkedIn
        if platform == "linkedin":
            while quick_post.weekday() >= 5:
                quick_post += timedelta(days=1)
        
        # --- TIME 2 & 3: Best times within 1 week ---
        best_day_numbers = [day_map.get(d, 1) for d in best_days[:2] if d in day_map]
        if not best_day_numbers:
            best_day_numbers = [1, 2]  # Default: Tuesday, Wednesday
        
        optimal_times = []
        for target_weekday in best_day_numbers:
            # Find next occurrence of this weekday
            days_ahead = target_weekday - now.weekday()
            if days_ahead <= 0:  # Already passed this week
                days_ahead += 7
            
            next_occurrence = now + timedelta(days=days_ahead)
            next_occurrence = next_occurrence.replace(
                hour=best_hour, minute=0, second=0, microsecond=0
            )
            
            # Make sure it's at least 24 hours from now
            if (next_occurrence - now).total_seconds() < 24 * 3600:
                next_occurrence += timedelta(days=7)
            
            optimal_times.append(next_occurrence)
        
        # Sort optimal times
        optimal_times.sort()
        
        # Ensure we have at least 2 optimal times
        while len(optimal_times) < 2:
            last_time = optimal_times[-1] if optimal_times else quick_post
            next_time = last_time + timedelta(days=2)
            next_time = next_time.replace(hour=best_hour, minute=0, second=0, microsecond=0)
            optimal_times.append(next_time)
        
        return [quick_post, optimal_times[0], optimal_times[1]]

