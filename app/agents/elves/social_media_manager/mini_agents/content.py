"""
Content Generator Agent - Creates multiple content variations.
Uses Few-Shot + RAFT for brand-consistent, high-quality content.
"""

import json
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from app.core.llm_clients import LLMMessage, llm_client
from app.core.vector_store import vector_store
from app.core.cache import cache

logger = structlog.get_logger(__name__)


CONTENT_SYSTEM_PROMPT = """You are a world-class social media content creator.
Your role is to generate engaging, on-brand content that resonates with the target audience.

## Expertise
- Compelling copywriting that drives engagement
- Platform-specific content optimization
- Emotional storytelling and hooks
- Conversion-focused messaging

## Guidelines
- Write in the specified brand voice
- Hook readers in the first line
- Include clear calls-to-action
- Optimize for platform algorithms
- Create content that sparks conversation

## Constraints
- Stay within platform character limits
- Maintain brand consistency
- Avoid controversial topics
- Use inclusive language"""


CONTENT_USER_PROMPT = """Generate social media content based on this strategy:

## Strategy Brief
- Tone: {tone}
- Key Messages: {key_messages}
- CTA Approach: {cta_approach}
- Target Audience: {target_audience}

## Request Details
- Platform: {platform}
- Topic: {topic}
- Content Type: {content_type}

## Brand Voice
{brand_voice}

## High-Performing Examples (for inspiration)
{examples}

Create 3 variations:
1. Hook-focused: Attention-grabbing opening that stops the scroll
2. Story-focused: Narrative approach with relatable scenario
3. Value-focused: Educational/helpful content with clear takeaways

For each variation, provide the post text and explain your creative reasoning.

Respond with valid JSON:
{{
    "variations": [
        {{
            "version": "hook_focused",
            "post_text": "The actual post content...",
            "reasoning": "Why this approach works..."
        }},
        {{
            "version": "story_focused", 
            "post_text": "The actual post content...",
            "reasoning": "Why this approach works..."
        }},
        {{
            "version": "value_focused",
            "post_text": "The actual post content...",
            "reasoning": "Why this approach works..."
        }}
    ]
}}"""


class ContentGeneratorAgent:
    """
    Content Generator Agent for Social Media Manager.
    
    Method: Few-Shot + RAFT
    - Retrieves high-performing content examples (few-shot)
    - Uses brand voice profile for consistency (RAFT)
    - Generates multiple variations with different angles
    """
    
    name = "content_generator_agent"
    
    # Platform character limits
    PLATFORM_LIMITS = {
        "twitter": 280,
        "linkedin": 3000,
        "facebook": 63206,
        "instagram": 2200,
    }
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate content variations.
        
        Args:
            state: Current workflow state with strategy
            context: Execution context with brand info
            
        Returns:
            Updated state with content variations
        """
        request = state.get("user_request", {})
        strategy = state.get("strategy", {})
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        content_type = request.get("content_type", "thought_leadership")
        
        # Get LLM-extracted search keywords
        search_keywords = request.get("search_keywords", [])
        
        logger.info(
            "Content generator executing",
            platform=platform,
            topic=topic[:50] if topic else "",
            search_keywords=search_keywords,
        )
        
        # Get user_id for filtering
        user_id = context.get("user_id")
        
        # Get brand voice profile (RAFT)
        brand_voice = await self._get_brand_voice(context)
        
        # Retrieve few-shot examples (with user_id and search_keywords)
        examples = await self._retrieve_examples(platform, content_type, topic, user_id, search_keywords)
        
        # Generate content variations
        variations = await self._generate_content(
            platform=platform,
            topic=topic,
            content_type=content_type,
            strategy=strategy,
            brand_voice=brand_voice,
            examples=examples,
        )
        
        return {"content_variations": variations}
    
    async def _get_brand_voice(self, context: dict) -> str:
        """Get brand voice profile for RAFT-based generation."""
        user_id = context.get("user_id")
        
        if not user_id:
            return "Professional, engaging, and authentic tone."
        
        # Try cache first
        cached_profile = await cache.get_voice_profile(user_id)
        if cached_profile:
            return self._format_voice_profile(cached_profile)
        
        # Try to build from context
        brand_info = context.get("brand_info", {})
        if brand_info.get("brand_voice"):
            return brand_info["brand_voice"]
        
        return "Professional, engaging, and authentic tone."
    
    def _format_voice_profile(self, profile: dict) -> str:
        """Format voice profile for prompt injection."""
        parts = []
        
        if profile.get("tone_characteristics"):
            tones = [f"{k}: {v:.0%}" for k, v in profile["tone_characteristics"].items() 
                     if v > 0.5]
            if tones:
                parts.append(f"Tone: {', '.join(tones)}")
        
        if profile.get("personality_traits"):
            parts.append(f"Personality: {', '.join(profile['personality_traits'])}")
        
        if profile.get("vocabulary_patterns"):
            common = profile["vocabulary_patterns"].get("common_words", [])
            if common:
                parts.append(f"Key vocabulary: {', '.join(common[:10])}")
        
        return "\n".join(parts) if parts else "Professional, engaging, and authentic tone."
    
    async def _retrieve_examples(
        self,
        platform: str,
        content_type: str,
        topic: str,
        user_id: Optional[str] = None,
        search_keywords: Optional[list[str]] = None,
    ) -> str:
        """
        Retrieve content examples using two-phase search:
        1. Search using LLM-extracted keywords for topic-specific content
        2. Search for platform examples
        """
        all_results = []
        
        # Phase 1: Search using keywords
        if search_keywords:
            try:
                keyword_query = " ".join(search_keywords)
                
                print("\n" + "="*70)
                print(f"🔑 CONTENT AGENT - KEYWORD SEARCH")
                print(f"   Keywords: {search_keywords}")
                print("-"*70)
                
                keyword_results = await vector_store.search_content_examples(
                    query=keyword_query,
                    user_id=user_id,
                    top_k=2,
                )
                
                if keyword_results:
                    print(f"   Found: {len(keyword_results)} results")
                    for i, r in enumerate(keyword_results, 1):
                        chunk_text = r.content or r.metadata.get("text", "N/A")
                        print(f"   [{i}] Score: {r.score:.3f} - {chunk_text[:60]}...")
                    all_results.extend(keyword_results)
                else:
                    print("   No keyword results found")
                print("="*70)
                
            except Exception as e:
                logger.warning("Keyword example search failed", error=str(e))
        
        # Phase 2: Search for platform examples
        try:
            platform_query = f"High performing {content_type} post on {platform}"
            
            platform_results = await vector_store.search_content_examples(
                query=platform_query,
                user_id=user_id,
                platform=platform,
                top_k=2,
            )
            
            if platform_results:
                all_results.extend(platform_results)
                
        except Exception as e:
            logger.warning("Platform example search failed", error=str(e))
        
        # Combine and format results
        if all_results:
            seen_ids = set()
            unique_results = []
            for r in all_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    unique_results.append(r)
            
            examples = []
            for i, r in enumerate(unique_results[:3], 1):
                content = r.content or r.metadata.get("text", "")
                if content:
                    examples.append(f"Example {i} (from knowledge base):\n{content}")
            
            if examples:
                logger.info(
                    "Retrieved examples from Pinecone",
                    count=len(examples),
                    user_id=user_id,
                )
                return "\n\n".join(examples)
        
        # Return platform-specific example templates
        return self._get_fallback_examples(platform)
    
    def _get_fallback_examples(self, platform: str) -> str:
        """Get fallback examples when vector store unavailable."""
        examples = {
            "linkedin": """Example 1:
Stop scrolling. This changed everything for me.

Last year, I was working 60-hour weeks and burning out fast.

Then I discovered the 3-hour rule:
→ 3 hours of deep work in the morning
→ No meetings before noon
→ Batch all admin tasks

The result? 40% more output in half the time.

Here's the exact framework I used: [details]

What's your productivity secret? Drop it below 👇

Example 2:
I made a $50,000 mistake last quarter.

And I'm grateful for it.

Here's what happened:
We rushed to market without proper testing. Customer complaints flooded in.

What I learned:
1. Speed without quality = disaster
2. Customer trust > quick wins
3. The right foundation matters more than timing

Now we have a 3-week testing protocol. Zero regrets.

What's a mistake that taught you the most?""",
            
            "twitter": """Example 1:
Hot take: Most "productivity tips" are just procrastination with extra steps.

The only hack that works?
Do the hard thing first.

That's it. That's the tweet.

Example 2:
Things I wish I knew 5 years ago:

• Your network IS your net worth
• Ship fast, iterate faster
• Perfect is the enemy of done
• Rest is not optional
• Document everything

What would you add?""",
            
            "instagram": """Example 1:
Real talk: Success isn't linear 📈

Some days you're on top of the world.
Others? You question everything.

But here's the secret no one tells you:

The messy middle is where growth happens. ✨

Double tap if you needed this reminder 💙

Example 2:
POV: You finally stopped comparing yourself to others 🌟

What changed for me:
✨ Set MY own goals
✨ Celebrated small wins
✨ Unfollowed accounts that triggered comparison

Your journey is YOUR journey. Own it. 💪

Save this for when you need a reminder 🔖""",
        }
        
        return examples.get(platform, examples["linkedin"])
    
    async def _generate_content(
        self,
        platform: str,
        topic: str,
        content_type: str,
        strategy: dict,
        brand_voice: str,
        examples: str,
    ) -> list[dict]:
        """Generate content variations using LLM."""
        
        user_prompt = CONTENT_USER_PROMPT.format(
            tone=strategy.get("tone", "professional"),
            key_messages=", ".join(strategy.get("key_messages", [])),
            cta_approach=strategy.get("cta_approach", "engage"),
            target_audience=strategy.get("target_audience", "professionals"),
            platform=platform,
            topic=topic,
            content_type=content_type,
            brand_voice=brand_voice,
            examples=examples,
        )
        
        messages = [
            LLMMessage(role="system", content=CONTENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_smart(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            variations = result.get("variations", [])
            
            # Validate and format variations
            formatted = []
            for var in variations:
                formatted.append({
                    "version": var.get("version", "variation"),
                    "content": {
                        "post_text": var.get("post_text", ""),
                        "reasoning": var.get("reasoning", ""),
                    },
                })
            
            logger.debug("Content generated", variations=len(formatted))
            return formatted
            
        except json.JSONDecodeError as e:
            logger.error("Content parsing failed", error=str(e))
            # Return simple fallback
            return [{
                "version": "default",
                "content": {
                    "post_text": f"Check out our thoughts on {topic}! #professional",
                    "reasoning": "Fallback content due to generation error",
                },
            }]

