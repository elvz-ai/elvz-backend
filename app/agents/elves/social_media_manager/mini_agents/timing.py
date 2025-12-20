"""
Timing Optimizer Agent - Calculates optimal posting times.
Uses Tool-Augmented Generation with analytics data.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from app.core.llm_clients import LLMMessage, llm_client
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


TIMING_USER_PROMPT = """Recommend optimal posting times for this content:

## Current Date/Time
Today is: {current_date}
Current time: {current_time} {timezone}

## Details
Platform: {platform}
Content Type: {content_type}
Target Audience: {target_audience}
User Timezone: {timezone}

## Analytics/Tool Data
{tool_results}

IMPORTANT: Recommend a posting time that is IN THE FUTURE (after {current_date}). 
Do NOT recommend dates in the past.

Respond with valid JSON:
{{
    "recommended_datetime": "YYYY-MM-DD HH:MM",
    "timezone": "timezone",
    "day_of_week": "Monday/Tuesday/etc",
    "time_slot": "HH:MM",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "alternative_times": [
        {{"datetime": "YYYY-MM-DD HH:MM", "reason": "why"}}
    ]
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
        """Generate timing recommendation using LLM."""
        
        # Get current date/time
        now = datetime.utcnow()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")
        
        user_prompt = TIMING_USER_PROMPT.format(
            platform=platform,
            content_type=content_type,
            target_audience=target_audience,
            timezone=timezone,
            tool_results=tool_results,
            current_date=current_date,
            current_time=current_time,
        )
        
        messages = [
            LLMMessage(role="system", content=TIMING_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            
            # Get recommended datetime
            recommended_dt_str = result.get("recommended_datetime", "")
            
            # Validate and fix the datetime to ensure it's in the future
            recommended_dt = self._validate_and_fix_datetime(
                recommended_dt_str, platform, now
            )
            
            # Also fix alternative times
            alternatives = []
            for alt in result.get("alternative_times", []):
                alt_dt_str = alt.get("datetime", "")
                alt_dt = self._validate_and_fix_datetime(alt_dt_str, platform, now)
                alternatives.append({
                    "datetime": alt_dt.strftime("%Y-%m-%d %H:%M"),
                    "reason": alt.get("reason", "Alternative optimal time"),
                })
            
            timing = {
                "datetime": recommended_dt.strftime("%Y-%m-%d %H:%M"),
                "timezone": result.get("timezone", timezone),
                "reason": result.get("reasoning", "Optimal engagement window"),
                "confidence": result.get("confidence", 0.8),
                "alternatives": alternatives,
            }
            
            logger.debug("Timing recommended", datetime=timing["datetime"])
            return timing
            
        except json.JSONDecodeError as e:
            logger.error("Timing parsing failed", error=str(e))
            # Return fallback
            next_best = self._get_next_best_time(platform, now)
            
            return {
                "datetime": next_best.strftime("%Y-%m-%d %H:%M"),
                "timezone": timezone,
                "reason": "Default optimal time for " + platform,
                "confidence": 0.7,
                "alternatives": [],
            }
    
    def _validate_and_fix_datetime(
        self, 
        dt_str: str, 
        platform: str, 
        now: datetime
    ) -> datetime:
        """Validate datetime string and ensure it's in the future."""
        try:
            # Try to parse the datetime
            if not dt_str:
                return self._get_next_best_time(platform, now)
            
            # Handle various formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"]:
                try:
                    parsed_dt = datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                # Could not parse, use fallback
                return self._get_next_best_time(platform, now)
            
            # Check if it's in the past
            if parsed_dt <= now:
                # It's in the past - find next occurrence of the same day/time
                # Extract the hour and minute
                target_hour = parsed_dt.hour
                target_minute = parsed_dt.minute
                target_weekday = parsed_dt.weekday()
                
                # Start from tomorrow
                next_dt = now + timedelta(days=1)
                next_dt = next_dt.replace(
                    hour=target_hour, 
                    minute=target_minute, 
                    second=0, 
                    microsecond=0
                )
                
                # Find the next occurrence of the same weekday
                while next_dt.weekday() != target_weekday:
                    next_dt += timedelta(days=1)
                
                # For LinkedIn, skip weekends
                if platform == "linkedin":
                    while next_dt.weekday() >= 5:
                        next_dt += timedelta(days=1)
                
                return next_dt
            
            return parsed_dt
            
        except Exception as e:
            logger.warning("Date validation failed", error=str(e), date_str=dt_str)
            return self._get_next_best_time(platform, now)
    
    def _get_next_best_time(self, platform: str, from_datetime: datetime) -> datetime:
        """Calculate next best posting time from given datetime."""
        # Best hours by platform
        best_hours = {
            "linkedin": 10,  # 10 AM
            "twitter": 12,   # 12 PM
            "instagram": 11, # 11 AM
            "facebook": 13,  # 1 PM
        }
        
        target_hour = best_hours.get(platform, 10)
        
        # Find next occurrence of that hour
        if from_datetime.hour < target_hour:
            next_time = from_datetime.replace(
                hour=target_hour, minute=0, second=0, microsecond=0
            )
        else:
            next_time = (from_datetime + timedelta(days=1)).replace(
                hour=target_hour, minute=0, second=0, microsecond=0
            )
        
        # Skip weekends for LinkedIn
        if platform == "linkedin":
            while next_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_time += timedelta(days=1)
        
        return next_time

