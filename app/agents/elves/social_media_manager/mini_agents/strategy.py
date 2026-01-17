"""
Content Strategy Agent - Creates strategic briefs for content generation.
Uses RAG + Dynamic Context for knowledge-grounded strategies.
"""

import json
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from app.core.llm_clients import LLMMessage, llm_client
from app.core.vector_store import vector_store

logger = structlog.get_logger(__name__)


STRATEGY_SYSTEM_PROMPT = """You are an expert content strategist specializing in social media marketing.
Your role is to create strategic briefs that guide content creation for maximum engagement.

## Expertise
- Social media strategy across all major platforms
- Audience psychology and engagement patterns
- Brand voice development and consistency
- Content marketing best practices

## Guidelines
- Consider the target platform's unique characteristics
- Balance brand consistency with platform trends
- Provide actionable, specific recommendations
- Think about the content funnel (awareness → consideration → conversion)

## Constraints
- Stay within platform best practices
- Respect brand voice guidelines
- Be specific and measurable in recommendations
- Consider content compliance requirements"""


STRATEGY_USER_PROMPT = """Create a content strategy brief for the following:

## Request
- Platform: {platform}
- Topic: {topic}
- Content Type: {content_type}
- Goals: {goals}

## Brand Context
{brand_context}

## Additional Context
{additional_context}

## Relevant Best Practices (from knowledge base)
{knowledge_context}

Create a strategic brief with:
1. Recommended tone and voice
2. Key messages to convey (2-3 main points)
3. Call-to-action approach
4. Target audience considerations
5. Content pillars/angles to explore
6. Platform-specific recommendations

Respond with valid JSON:
{{
    "tone": "description of recommended tone",
    "key_messages": ["message 1", "message 2", "message 3"],
    "cta_approach": "description of CTA strategy",
    "target_audience": "audience description",
    "content_pillars": ["pillar 1", "pillar 2"],
    "platform_specific_notes": "platform-specific recommendations"
}}"""


class StrategyAgent:
    """
    Content Strategy Agent for Social Media Manager.
    
    Method: RAG + Dynamic Context
    - Retrieves relevant best practices from knowledge base
    - Injects user-specific context (brand info, analytics)
    - Generates strategic brief for content creation
    """
    
    name = "strategy_agent"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate content strategy brief.
        
        Args:
            state: Current workflow state with user request
            context: Execution context with user info
            
        Returns:
            Updated state with strategy brief
        """
        request = state.get("user_request", {})
        
        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        content_type = request.get("content_type", "thought_leadership")
        goals = request.get("goals", ["engagement"])
        
        logger.info(
            "Strategy agent executing",
            platform=platform,
            topic=topic[:50] if topic else "",
        )
        
        # Use shared knowledge_context from orchestrator (single Pinecone query)
        knowledge_context = state.get("knowledge_context", "")
        
        # Fallback to embedded knowledge if no results
        if not knowledge_context:
            knowledge_context = self._get_fallback_knowledge(platform, content_type)
        
        # Build brand context
        brand_context = self._build_brand_context(context)
        
        # Generate strategy
        strategy = await self._generate_strategy(
            platform=platform,
            topic=topic,
            content_type=content_type,
            goals=goals,
            brand_context=brand_context,
            knowledge_context=knowledge_context,
            additional_context=request.get("additional_context", ""),
        )
        
        return {"strategy": strategy}
    
    async def _retrieve_knowledge(
        self,
        platform: str,
        topic: str,
        content_type: str,
        user_id: Optional[str] = None,
        search_keywords: Optional[list[str]] = None,
    ) -> str:
        """
        Retrieve relevant knowledge from Pinecone using two-phase search:
        1. First search for topic-specific content using search_keywords
        2. Then search for platform best practices
        """
        all_results = []
        
        # Phase 1: Search for topic-specific knowledge using LLM-extracted keywords
        if search_keywords:
            try:
                keyword_query = " ".join(search_keywords)
                
                print("\n" + "="*70)
                print(f"🔑 PHASE 1: KEYWORD SEARCH (Strategy Agent)")
                print(f"   Keywords: {search_keywords}")
                print(f"   Query: {keyword_query}")
                print(f"   User ID: {user_id}")
                print("-"*70)
                
                keyword_results = await vector_store.search_knowledge(
                    query=keyword_query,
                    user_id=user_id,
                    top_k=3,
                )
                
                if keyword_results:
                    print(f"   Found: {len(keyword_results)} results")
                    for i, r in enumerate(keyword_results, 1):
                        chunk_text = r.content or r.metadata.get("text", "N/A")
                        print(f"   [{i}] Score: {r.score:.3f}")
                        print(f"       Text: {chunk_text[:100]}...")
                        print(f"       User: {r.metadata.get('user_id', 'N/A')}")
                    all_results.extend(keyword_results)
                else:
                    print("   No results found for keywords")
                print("="*70)
                
            except Exception as e:
                logger.warning("Keyword search failed", error=str(e))
        
        # Phase 2: Search for platform best practices
        try:
            platform_query = f"Best practices for {content_type} on {platform}"
            
            print("\n" + "="*70)
            print(f"📋 PHASE 2: PLATFORM BEST PRACTICES (Strategy Agent)")
            print(f"   Query: {platform_query}")
            print(f"   Platform: {platform}")
            print("-"*70)
            
            platform_results = await vector_store.search_knowledge(
                query=platform_query,
                user_id=user_id,
                platform=platform,
                top_k=2,
            )
            
            if platform_results:
                print(f"   Found: {len(platform_results)} results")
                for i, r in enumerate(platform_results, 1):
                    chunk_text = r.content or r.metadata.get("text", "N/A")
                    print(f"   [{i}] Score: {r.score:.3f}")
                    print(f"       Text: {chunk_text[:100]}...")
                all_results.extend(platform_results)
            else:
                print("   No platform results found")
            print("="*70 + "\n")
            
        except Exception as e:
            logger.warning("Platform search failed", error=str(e))
        
        # Combine and deduplicate results
        if all_results:
            seen_ids = set()
            unique_results = []
            for r in all_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    unique_results.append(r)
            
            knowledge_items = [r.content or r.metadata.get("text", "") for r in unique_results]
            
            logger.info(
                "Retrieved knowledge from Pinecone (two-phase)",
                keyword_results=len(search_keywords) if search_keywords else 0,
                total_results=len(unique_results),
                user_id=user_id,
            )
            return "\n".join(f"- {item}" for item in knowledge_items if item)
        
        # Fallback to embedded knowledge
        return self._get_fallback_knowledge(platform, content_type)
    
    def _get_fallback_knowledge(self, platform: str, content_type: str) -> str:
        """Get fallback best practices when vector store unavailable."""
        platform_tips = {
            "linkedin": [
                "Professional tone works best, but personality is welcome",
                "Optimal length: 1300 characters for feed posts",
                "Use line breaks for readability",
                "First line should hook attention (appears in preview)",
                "Include relevant hashtags (3-5 is optimal)",
            ],
            "twitter": [
                "Concise and punchy content performs best",
                "Thread format for longer content",
                "Engage with replies quickly",
                "Use 1-2 highly relevant hashtags",
                "Visual content gets 150% more retweets",
            ],
            "instagram": [
                "Visual-first platform - image quality matters",
                "Captions can be up to 2200 characters",
                "Use hashtags strategically (up to 30, but 9-11 is optimal)",
                "Stories drive engagement, feed for portfolio",
                "Reels have highest organic reach",
            ],
            "facebook": [
                "Longer form content can work well",
                "Video content prioritized by algorithm",
                "Native content outperforms external links",
                "Community building is key",
                "Best posting times: 1-4 PM",
            ],
        }
        
        tips = platform_tips.get(platform, platform_tips["linkedin"])
        return "\n".join(f"- {tip}" for tip in tips)
    
    def _build_brand_context(self, context: dict) -> str:
        """Build brand context string from execution context."""
        brand_info = context.get("brand_info", {})
        
        if not brand_info:
            return "No specific brand context provided. Use professional, engaging tone."
        
        parts = []
        
        if brand_info.get("brand_name"):
            parts.append(f"Brand: {brand_info['brand_name']}")
        
        if brand_info.get("industry"):
            parts.append(f"Industry: {brand_info['industry']}")
        
        if brand_info.get("brand_voice"):
            parts.append(f"Brand Voice: {brand_info['brand_voice']}")
        
        if brand_info.get("target_audience"):
            parts.append(f"Target Audience: {brand_info['target_audience']}")
        
        return "\n".join(parts) if parts else "No specific brand context provided."
    
    async def _generate_strategy(
        self,
        platform: str,
        topic: str,
        content_type: str,
        goals: list[str],
        brand_context: str,
        knowledge_context: str,
        additional_context: str,
    ) -> dict:
        """Generate strategy using LLM."""
        
        user_prompt = STRATEGY_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            content_type=content_type,
            goals=", ".join(goals),
            brand_context=brand_context,
            knowledge_context=knowledge_context,
            additional_context=additional_context or "None provided",
        )

        logger.info("Strategy user prompt", user_prompt=user_prompt)
        
        messages = [
            LLMMessage(role="system", content=STRATEGY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate(messages, json_mode=True)

        logger.info("Strategy response", response=response.content)
        
        try:
            strategy = json.loads(response.content)
            logger.debug("Strategy generated", tone=strategy.get("tone"))
            return strategy
        except json.JSONDecodeError as e:
            logger.error("Strategy parsing failed", error=str(e))
            # Return default strategy
            return {
                "tone": "professional and engaging",
                "key_messages": [f"Key insight about {topic}"],
                "cta_approach": "Encourage engagement through thoughtful question",
                "target_audience": "Professional audience interested in " + topic,
                "content_pillars": ["expertise", "value"],
                "platform_specific_notes": f"Optimize for {platform} algorithm",
            }

