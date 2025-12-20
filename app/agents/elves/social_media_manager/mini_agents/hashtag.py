"""
Hashtag Research Agent - Finds optimal hashtags for content.
Uses Tool-Augmented Generation for data-driven recommendations.
"""

import json
from typing import Any, Optional

import structlog

from app.core.llm_clients import LLMMessage, llm_client
from app.tools.registry import tool_registry

logger = structlog.get_logger(__name__)


HASHTAG_SYSTEM_PROMPT = """You are a social media optimization specialist focusing on hashtag strategy.
Your role is to recommend optimal hashtags for maximum reach and engagement.

## Expertise
- Hashtag research and analysis
- Platform algorithm understanding
- Trend identification
- Niche community discovery

## Guidelines
- Balance popular and niche hashtags
- Consider hashtag relevance to content
- Recommend mix of volumes (high/medium/low competition)
- Avoid banned or overused hashtags

## Strategy
For optimal results, recommend:
- 1-2 high-volume hashtags for reach
- 2-3 medium-volume hashtags for engagement
- 2-3 niche hashtags for targeted audience"""


HASHTAG_USER_PROMPT = """Research and recommend hashtags for this content:

## Content Summary
Platform: {platform}
Topic: {topic}
Key Messages: {key_messages}

## Tool Results (Hashtag Research)
{tool_results}

Based on the tool results and the content context, recommend 5-7 optimal hashtags.

Respond with valid JSON:
{{
    "hashtags": [
        {{
            "tag": "hashtag without #",
            "volume": "high/medium/low",
            "relevance_score": 0.0-1.0,
            "rationale": "why this hashtag"
        }}
    ],
    "strategy_notes": "overall hashtag strategy explanation"
}}"""


class HashtagResearchAgent:
    """
    Hashtag Research Agent for Social Media Manager.
    
    Method: Tool-Augmented Generation (TAG)
    - Uses hashtag research tool for volume/trend data
    - LLM synthesizes tool results into recommendations
    - Balances reach and relevance
    """
    
    name = "hashtag_research_agent"
    
    # Platform-specific hashtag limits
    HASHTAG_LIMITS = {
        "twitter": 2,
        "linkedin": 5,
        "instagram": 11,  # Optimal is 9-11, max is 30
        "facebook": 3,
    }
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Research and recommend hashtags.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with hashtag recommendations
        """
        request = state.get("user_request", {})
        strategy = state.get("strategy", {})
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        
        logger.info(
            "Hashtag research agent executing",
            platform=platform,
            topic=topic[:50],
        )
        
        # Extract keywords from topic and strategy
        keywords = self._extract_keywords(topic, strategy)
        
        # Call hashtag research tool
        tool_results = await self._research_hashtags(keywords, platform)
        
        # Generate recommendations using LLM
        hashtags = await self._generate_recommendations(
            platform=platform,
            topic=topic,
            key_messages=strategy.get("key_messages", []),
            tool_results=tool_results,
        )
        
        return {"hashtags": hashtags}
    
    def _extract_keywords(self, topic: str, strategy: dict) -> list[str]:
        """Extract relevant keywords for hashtag research."""
        keywords = []
        
        # From topic
        words = topic.split()
        keywords.extend([w.lower() for w in words if len(w) > 3][:5])
        
        # From strategy
        key_messages = strategy.get("key_messages", [])
        for msg in key_messages:
            keywords.extend([w.lower() for w in msg.split() if len(w) > 4][:2])
        
        content_pillars = strategy.get("content_pillars", [])
        keywords.extend(content_pillars)
        
        # Deduplicate
        return list(set(keywords))[:10]
    
    async def _research_hashtags(
        self,
        keywords: list[str],
        platform: str,
    ) -> str:
        """Use hashtag research tool to get data."""
        try:
            # Import here to avoid circular imports
            from app.tools.social_media_tools import HashtagSearchInput
            
            tool = tool_registry.get("hashtag_research")
            if tool:
                input_data = HashtagSearchInput(
                    keywords=keywords,
                    platform=platform,
                    max_results=15,
                )
                result = await tool.execute(input_data)
                
                if result.success and result.data:
                    hashtags = result.data.get("hashtags", [])
                    formatted = []
                    for h in hashtags:
                        formatted.append(
                            f"- #{h.get('tag', '')}: volume={h.get('volume', 'unknown')}, "
                            f"relevance={h.get('relevance_score', 0):.2f}"
                        )
                    return "\n".join(formatted)
        
        except Exception as e:
            logger.warning("Hashtag tool failed", error=str(e))
        
        # Return fallback data
        return self._get_fallback_data(keywords, platform)
    
    def _get_fallback_data(self, keywords: list[str], platform: str) -> str:
        """Get fallback hashtag data when tool unavailable."""
        # Generate basic hashtag suggestions from keywords
        suggestions = []
        
        for kw in keywords[:5]:
            clean_kw = kw.replace(" ", "").lower()
            suggestions.append(f"- #{clean_kw}: volume=medium, relevance=0.80")
            suggestions.append(f"- #{clean_kw}tips: volume=low, relevance=0.70")
        
        # Add platform-specific generic hashtags
        platform_hashtags = {
            "linkedin": ["#leadership", "#innovation", "#business", "#growth"],
            "twitter": ["#tech", "#startup", "#marketing"],
            "instagram": ["#instagood", "#business", "#motivation"],
            "facebook": ["#community", "#smallbusiness"],
        }
        
        for tag in platform_hashtags.get(platform, []):
            suggestions.append(f"- {tag}: volume=high, relevance=0.60")
        
        return "\n".join(suggestions)
    
    async def _generate_recommendations(
        self,
        platform: str,
        topic: str,
        key_messages: list[str],
        tool_results: str,
    ) -> list[dict]:
        """Generate hashtag recommendations using LLM."""
        
        user_prompt = HASHTAG_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            key_messages=", ".join(key_messages) if key_messages else "Not specified",
            tool_results=tool_results,
        )
        
        messages = [
            LLMMessage(role="system", content=HASHTAG_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            hashtags = result.get("hashtags", [])
            
            # Limit to platform max
            max_hashtags = self.HASHTAG_LIMITS.get(platform, 5)
            hashtags = hashtags[:max_hashtags]
            
            # Ensure proper format
            formatted = []
            for h in hashtags:
                formatted.append({
                    "tag": f"#{h.get('tag', '').lstrip('#')}",
                    "volume": h.get("volume", "medium"),
                    "relevance": h.get("relevance_score", 0.7),
                    "rationale": h.get("rationale", ""),
                })
            
            logger.debug("Hashtags recommended", count=len(formatted))
            return formatted
            
        except json.JSONDecodeError as e:
            logger.error("Hashtag parsing failed", error=str(e))
            # Return simple fallback
            topic_tag = topic.split()[0].lower() if topic else "business"
            return [
                {"tag": f"#{topic_tag}", "volume": "medium", "relevance": 0.8},
                {"tag": "#professional", "volume": "high", "relevance": 0.6},
            ]

