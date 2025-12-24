"""
Visual Advisor Agent - Provides visual content recommendations.
Uses RAG + Few-Shot for design guidance.
"""

import json
from typing import Any, Optional

import structlog

from app.core.llm_clients import LLMMessage, llm_client
from app.core.vector_store import vector_store

logger = structlog.get_logger(__name__)


VISUAL_SYSTEM_PROMPT = """You are a visual content strategist for social media.
Your role is to recommend visual elements that complement written content.

## Expertise
- Social media visual design trends
- Platform-specific visual requirements
- Brand consistency in visuals
- Engagement-driving visual patterns

## Guidelines
- Recommend visuals that enhance the message
- Consider platform-specific dimensions
- Balance brand colors with engagement patterns
- Suggest accessible and inclusive visuals

## Visual Types
- Images (photos, graphics, illustrations)
- Carousels (multi-image posts)
- Videos (short-form, long-form)
- Infographics (data visualization)
- Text graphics (quote cards, tips)"""


VISUAL_USER_PROMPT = """Recommend visual content for this post:

## Content Summary
Platform: {platform}
Topic: {topic}
Content Type: {content_type}
Key Messages: {key_messages}

## Brand Guidelines
{brand_guidelines}

## Visual Best Practices
{visual_practices}

Recommend visual elements that will enhance engagement.

Respond with valid JSON:
{{
    "primary_recommendation": {{
        "type": "image/carousel/video/infographic/text_graphic",
        "description": "detailed description of recommended visual",
        "rationale": "why this visual works"
    }},
    "alternative_options": [
        {{
            "type": "visual type",
            "description": "description",
            "rationale": "why"
        }}
    ],
    "design_specs": {{
        "dimensions": "recommended dimensions",
        "color_palette": ["color1", "color2"],
        "style": "style description",
        "key_elements": ["element1", "element2"]
    }},
    "accessibility_notes": "accessibility considerations"
}}"""


class VisualAdvisorAgent:
    """
    Visual Advisor Agent for Social Media Manager.
    
    Method: RAG + Few-Shot
    - Retrieves visual best practices from knowledge base
    - Uses brand style guidelines
    - Generates specific visual recommendations
    """
    
    name = "visual_advisor_agent"
    
    # Platform-specific dimensions
    PLATFORM_DIMENSIONS = {
        "linkedin": {
            "feed_post": "1200 x 627",
            "carousel": "1080 x 1080",
            "profile_banner": "1584 x 396",
        },
        "twitter": {
            "tweet_image": "1200 x 675",
            "header": "1500 x 500",
        },
        "instagram": {
            "feed_square": "1080 x 1080",
            "feed_portrait": "1080 x 1350",
            "story": "1080 x 1920",
            "reel": "1080 x 1920",
        },
        "facebook": {
            "feed_image": "1200 x 630",
            "story": "1080 x 1920",
        },
    }
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate visual recommendations.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with visual recommendations
        """
        request = state.get("user_request", {})
        strategy = state.get("strategy", {})
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        content_type = request.get("content_type", "thought_leadership")
        
        # Get user_id for filtering
        user_id = context.get("user_id")
        
        logger.info(
            "Visual advisor executing",
            platform=platform,
            content_type=content_type,
            user_id=user_id,
        )
        
        # Get brand guidelines
        brand_guidelines = self._get_brand_guidelines(context)
        
        # Retrieve visual best practices (with user_id for personalized knowledge)
        visual_practices = await self._retrieve_visual_practices(platform, content_type, user_id)
        
        # Generate recommendations
        visual_advice = await self._generate_recommendations(
            platform=platform,
            topic=topic,
            content_type=content_type,
            key_messages=strategy.get("key_messages", []),
            brand_guidelines=brand_guidelines,
            visual_practices=visual_practices,
        )
        
        return {"visual_advice": visual_advice}
    
    def _get_brand_guidelines(self, context: dict) -> str:
        """Get brand visual guidelines from context."""
        brand_info = context.get("brand_info", {})
        
        if not brand_info:
            return "No specific brand guidelines. Use professional, clean visuals."
        
        parts = []
        
        if brand_info.get("brand_colors"):
            parts.append(f"Brand Colors: {brand_info['brand_colors']}")
        
        if brand_info.get("visual_style"):
            parts.append(f"Visual Style: {brand_info['visual_style']}")
        
        if brand_info.get("logo_usage"):
            parts.append(f"Logo Usage: {brand_info['logo_usage']}")
        
        return "\n".join(parts) if parts else "No specific brand guidelines."
    
    async def _retrieve_visual_practices(
        self,
        platform: str,
        content_type: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Retrieve visual best practices from Pinecone knowledge base."""
        try:
            query = f"Visual design best practices for {content_type} on {platform}"
            
            results = await vector_store.search_knowledge(
                query=query,
                user_id=user_id,  # Filter by user_id for personalized knowledge
                platform=platform,
                category="visual",
                top_k=3,
            )
            
            if results:
                practices = [r.content for r in results]
                
                # Debug: Print retrieved visual practices to terminal
                print("\n" + "="*70)
                print(f"🎨 PINECONE VISUAL PRACTICES RETRIEVED (Visual Agent)")
                print(f"   Query: {query}")
                print(f"   User ID: {user_id}")
                print(f"   Platform: {platform}")
                print(f"   Results: {len(results)}")
                print("-"*70)
                for i, r in enumerate(results, 1):
                    # Get content from either content field or metadata.text
                    chunk_text = r.content or r.metadata.get("text", "N/A")
                    print(f"   [{i}] Score: {r.score:.3f}")
                    print(f"       Text: {chunk_text}")
                    print(f"       Category: {r.metadata.get('category', 'N/A')}")
                    print(f"       User: {r.metadata.get('user_id', 'N/A')}")
                    print()
                print("="*70 + "\n")
                
                logger.info(
                    "Retrieved visual practices from Pinecone",
                    count=len(results),
                    user_id=user_id,
                    platform=platform
                )
                return "\n".join(f"- {p}" for p in practices)
            
        except Exception as e:
            logger.warning("Visual practices retrieval from Pinecone failed", error=str(e))
        
        # Return fallback practices
        return self._get_fallback_practices(platform)
    
    def _get_fallback_practices(self, platform: str) -> str:
        """Get fallback visual practices."""
        practices = {
            "linkedin": """- Professional imagery performs best
- Include faces for 38% more engagement
- Blue tones align with platform aesthetics
- Infographics for data-driven content
- Carousels for educational content""",
            
            "twitter": """- Bold, high-contrast visuals stand out
- Memes and humor work well
- Keep text on images minimal
- GIFs for reactions and engagement
- Thread visuals for storytelling""",
            
            "instagram": """- High-quality, aesthetic imagery essential
- Consistent visual theme/filter
- User-generated content performs well
- Behind-the-scenes resonates
- Reels outperform static images""",
            
            "facebook": """- Native video gets priority
- Emotional imagery performs best
- Text overlays should be minimal
- Live videos have highest engagement
- 360 photos for immersive experiences""",
        }
        
        return practices.get(platform, practices["linkedin"])
    
    async def _generate_recommendations(
        self,
        platform: str,
        topic: str,
        content_type: str,
        key_messages: list[str],
        brand_guidelines: str,
        visual_practices: str,
    ) -> list[dict]:
        """Generate visual recommendations using LLM."""
        
        user_prompt = VISUAL_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            content_type=content_type,
            key_messages=", ".join(key_messages) if key_messages else "Not specified",
            brand_guidelines=brand_guidelines,
            visual_practices=visual_practices,
        )
        
        messages = [
            LLMMessage(role="system", content=VISUAL_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            
            # Format recommendations
            recommendations = []
            
            # Primary recommendation
            primary = result.get("primary_recommendation", {})
            if primary:
                recommendations.append({
                    "type": primary.get("type", "image"),
                    "description": primary.get("description", ""),
                    "rationale": primary.get("rationale", ""),
                    "is_primary": True,
                })
            
            # Alternatives
            for alt in result.get("alternative_options", [])[:2]:
                recommendations.append({
                    "type": alt.get("type", "image"),
                    "description": alt.get("description", ""),
                    "rationale": alt.get("rationale", ""),
                    "is_primary": False,
                })
            
            # Add design specs to primary
            if recommendations:
                specs = result.get("design_specs", {})
                recommendations[0]["specs"] = {
                    "dimensions": specs.get("dimensions") or self._get_dimensions(platform),
                    "colors": specs.get("color_palette", []),
                    "style": specs.get("style", ""),
                    "elements": specs.get("key_elements", []),
                }
                recommendations[0]["accessibility"] = result.get("accessibility_notes", "")
            
            logger.debug("Visual recommendations generated", count=len(recommendations))
            return recommendations
            
        except json.JSONDecodeError as e:
            logger.error("Visual parsing failed", error=str(e))
            # Return fallback
            return [{
                "type": "image",
                "description": f"Professional visual related to {topic}",
                "rationale": "Default recommendation due to generation error",
                "is_primary": True,
                "specs": {
                    "dimensions": self._get_dimensions(platform),
                    "colors": ["#0077B5", "#FFFFFF"],  # LinkedIn blue
                    "style": "clean and professional",
                    "elements": [],
                },
            }]
    
    def _get_dimensions(self, platform: str, image_type: str = "feed_post") -> str:
        """Get recommended dimensions for platform."""
        platform_dims = self.PLATFORM_DIMENSIONS.get(platform, {})
        
        # Try to get specific type, fall back to first available
        if image_type in platform_dims:
            return platform_dims[image_type]
        
        if platform_dims:
            return list(platform_dims.values())[0]
        
        return "1200 x 630"  # Safe default

