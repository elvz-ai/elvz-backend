"""
Intent Classification for routing user requests to appropriate Elves.
Uses LLM-based classification with structured output.
"""

import json
from enum import Enum
from typing import Optional

import structlog
from pydantic import BaseModel

from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


class ElfType(str, Enum):
    """Types of Elf agents available."""
    SOCIAL_MEDIA = "social_media"
    SEO = "seo"
    COPYWRITER = "copywriter"
    ASSISTANT = "assistant"
    MULTI = "multi"  # When multiple Elves needed


class IntentType(str, Enum):
    """Types of user intents."""
    # Social Media
    CREATE_POST = "create_post"
    ANALYZE_PERFORMANCE = "analyze_performance"
    GENERATE_CALENDAR = "generate_calendar"
    HASHTAG_RESEARCH = "hashtag_research"
    
    # SEO
    AUDIT_SITE = "audit_site"
    KEYWORD_RESEARCH = "keyword_research"
    OPTIMIZE_PAGE = "optimize_page"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    
    # Copywriter
    WRITE_BLOG = "write_blog"
    WRITE_AD = "write_ad"
    WRITE_PRODUCT = "write_product"
    REWRITE_CONTENT = "rewrite_content"
    
    # Assistant
    MANAGE_TASKS = "manage_tasks"
    DRAFT_EMAIL = "draft_email"
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    
    # General
    GENERAL_QUESTION = "general_question"
    CLARIFICATION = "clarification"


class IntentClassification(BaseModel):
    """Result of intent classification."""
    primary_elf: ElfType
    secondary_elves: list[ElfType] = []
    intent: IntentType
    confidence: float
    requires_multi_elf: bool = False
    extracted_entities: dict = {}
    reasoning: str = ""


# Intent classification prompt
CLASSIFICATION_SYSTEM_PROMPT = """You are an intent classifier for Elvz.ai, a multi-agent AI platform.
Your task is to classify user requests to route them to the appropriate Elf agent.

Available Elves:
1. social_media - Social media content creation, scheduling, hashtags, performance analysis
2. seo - Website audits, keyword research, technical SEO, content optimization
3. copywriter - Blog posts, ad copy, product descriptions, content writing
4. assistant - Task management, emails, research, document creation, general help

Analyze the user's request and determine:
1. The primary Elf that should handle this request
2. Any secondary Elves that might be needed
3. The specific intent/task type
4. Key entities mentioned (platforms, topics, URLs, etc.)

Respond with valid JSON only."""

CLASSIFICATION_USER_PROMPT = """Classify this user request:

"{user_message}"

User context:
- Previous Elf used: {previous_elf}
- Session topic: {session_topic}

Respond with JSON:
{{
    "primary_elf": "social_media|seo|copywriter|assistant",
    "secondary_elves": [],
    "intent": "create_post|audit_site|write_blog|manage_tasks|...",
    "confidence": 0.0-1.0,
    "requires_multi_elf": true/false,
    "extracted_entities": {{
        "platform": "linkedin/twitter/instagram/facebook or null",
        "topic": "main topic of the request",
        "url": "any URL mentioned or null",
        "search_keywords": ["keyword1", "keyword2", "keyword3"]
    }},
    "reasoning": "Brief explanation of classification"
}}

IMPORTANT for search_keywords:
- Extract KEY CONCEPTS that should be looked up in a knowledge base
- Include: proper nouns, product names, company names, brand names, specific topics
- Do NOT include: generic words like "post", "create", "content", "write", platform names
- These keywords will be used to search a vector database for relevant information
- Example: "create a linkedin post on Klara AI governance" → ["Klara", "AI governance"]
- Example: "write about machine learning trends 2025" → ["machine learning", "ML trends", "2025"]
- Example: "post about our new product launch" → ["product launch"]"""


class IntentClassifier:
    """
    Classifies user intents to route to appropriate Elf agents.
    Uses fast LLM (GPT-4o-mini) for efficient classification.
    """
    
    # Keyword-based quick classification for common patterns
    QUICK_PATTERNS = {
        "social_media": [
            "post", "linkedin", "twitter", "instagram", "facebook",
            "hashtag", "social media", "engagement", "followers",
            "content calendar", "schedule post"
        ],
        "seo": [
            "seo", "keyword", "ranking", "backlink", "audit",
            "search engine", "meta", "sitemap", "crawl", "organic traffic"
        ],
        "copywriter": [
            "write", "blog", "article", "ad copy", "product description",
            "landing page", "headline", "copy", "content"
        ],
        "assistant": [
            "task", "schedule", "meeting", "email", "research",
            "summarize", "document", "reminder", "calendar"
        ],
    }
    
    async def classify(
        self,
        user_message: str,
        previous_elf: Optional[str] = None,
        session_topic: Optional[str] = None,
    ) -> IntentClassification:
        """
        Classify user intent and determine routing.
        
        Args:
            user_message: The user's message
            previous_elf: The Elf used in the previous turn
            session_topic: The overall topic of the session
            
        Returns:
            IntentClassification with routing information
        """
        logger.info("Classifying intent", message=user_message[:100])
        
        # Try quick pattern matching first
        quick_result = self._quick_classify(user_message)
        if quick_result and quick_result.confidence > 0.9:
            logger.debug("Quick classification succeeded", elf=quick_result.primary_elf)
            return quick_result
        
        # Use LLM for complex classification
        try:
            llm_result = await self._llm_classify(
                user_message, previous_elf, session_topic
            )
            return llm_result
        except Exception as e:
            logger.error("LLM classification failed", error=str(e))
            # Fall back to quick classification or default
            return quick_result or IntentClassification(
                primary_elf=ElfType.ASSISTANT,
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.5,
                reasoning="Fallback classification due to error",
            )
    
    def _quick_classify(self, message: str) -> Optional[IntentClassification]:
        """
        Quick keyword-based classification for obvious intents.
        Returns None if not confident enough.
        """
        message_lower = message.lower()
        scores = {}
        
        for elf_type, patterns in self.QUICK_PATTERNS.items():
            score = sum(1 for p in patterns if p in message_lower)
            if score > 0:
                scores[elf_type] = score
        
        if not scores:
            return None
        
        # Get highest scoring Elf
        best_elf = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_elf] / max(total_score, 1)
        
        # Only return if highly confident
        if confidence < 0.6:
            return None
        
        # Determine intent based on keywords
        intent = self._determine_intent(message_lower, best_elf)
        
        # Extract basic entities
        entities = self._extract_entities(message_lower)
        
        return IntentClassification(
            primary_elf=ElfType(best_elf),
            intent=intent,
            confidence=min(confidence + 0.2, 0.95),  # Boost but cap confidence
            extracted_entities=entities,
            reasoning=f"Keyword match: {scores}",
        )
    
    def _determine_intent(self, message: str, elf_type: str) -> IntentType:
        """Determine specific intent based on message and Elf type."""
        intent_patterns = {
            "social_media": {
                IntentType.CREATE_POST: ["create", "write", "generate", "make a post"],
                IntentType.ANALYZE_PERFORMANCE: ["analyze", "performance", "metrics", "stats"],
                IntentType.GENERATE_CALENDAR: ["calendar", "schedule", "plan"],
                IntentType.HASHTAG_RESEARCH: ["hashtag", "tags"],
            },
            "seo": {
                IntentType.AUDIT_SITE: ["audit", "check", "analyze site"],
                IntentType.KEYWORD_RESEARCH: ["keyword", "search terms"],
                IntentType.OPTIMIZE_PAGE: ["optimize", "improve"],
                IntentType.COMPETITOR_ANALYSIS: ["competitor", "competition"],
            },
            "copywriter": {
                IntentType.WRITE_BLOG: ["blog", "article", "post"],
                IntentType.WRITE_AD: ["ad", "advertisement", "campaign"],
                IntentType.WRITE_PRODUCT: ["product description", "product page"],
                IntentType.REWRITE_CONTENT: ["rewrite", "revise", "improve"],
            },
            "assistant": {
                IntentType.MANAGE_TASKS: ["task", "todo", "reminder"],
                IntentType.DRAFT_EMAIL: ["email", "message", "reply"],
                IntentType.RESEARCH: ["research", "find", "look up"],
                IntentType.SUMMARIZE: ["summarize", "summary", "brief"],
            },
        }
        
        patterns = intent_patterns.get(elf_type, {})
        
        for intent, keywords in patterns.items():
            if any(k in message for k in keywords):
                return intent
        
        # Default intents per Elf
        defaults = {
            "social_media": IntentType.CREATE_POST,
            "seo": IntentType.AUDIT_SITE,
            "copywriter": IntentType.WRITE_BLOG,
            "assistant": IntentType.GENERAL_QUESTION,
        }
        
        return defaults.get(elf_type, IntentType.GENERAL_QUESTION)
    
    def _extract_entities(self, message: str) -> dict:
        """Extract basic entities from message."""
        entities = {}
        
        # Platform detection
        platforms = ["linkedin", "twitter", "instagram", "facebook", "tiktok"]
        for platform in platforms:
            if platform in message:
                entities["platform"] = platform
                break
        
        # URL detection (simple)
        if "http" in message or "www." in message:
            words = message.split()
            for word in words:
                if "http" in word or "www." in word:
                    entities["url"] = word.strip(".,;")
                    break
        
        # Topic extraction - find what comes after "about" or "on" or "for"
        topic = self._extract_topic(message)
        if topic:
            entities["topic"] = topic
            # Use topic as primary search keyword for quick classification
            entities["search_keywords"] = [topic]
        
        return entities
    
    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract the topic from the user's message."""
        import re
        
        # Common patterns for topic extraction
        patterns = [
            r'about\s+(.+?)(?:\s+for|\s+on|\s+with|\.|$)',  # "about AI trends"
            r'on\s+(.+?)(?:\s+for|\s+about|\.|$)',           # "on machine learning"
            r'regarding\s+(.+?)(?:\s+for|\s+on|\.|$)',       # "regarding AI"
            r'topic[:\s]+(.+?)(?:\.|$)',                      # "topic: AI trends"
            r'post about\s+(.+?)(?:\s+for|\s+on|\.|$)',      # "post about AI"
            r'article about\s+(.+?)(?:\s+for|\s+on|\.|$)',   # "article about AI"
            r'content about\s+(.+?)(?:\s+for|\s+on|\.|$)',   # "content about AI"
            r'write about\s+(.+?)(?:\s+for|\s+on|\.|$)',     # "write about AI"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = re.sub(r'\s+(please|thanks|thank you|asap).*$', '', topic, flags=re.IGNORECASE)
                if len(topic) > 3:  # Minimum topic length
                    return topic
        
        # Fallback: If message has "create/write/generate" + "post/content", 
        # take everything after the action word as potential topic context
        action_match = re.search(
            r'(?:create|write|generate|make)\s+(?:a\s+)?(?:linkedin\s+)?(?:post|content|article)\s+(.+)',
            message, 
            re.IGNORECASE
        )
        if action_match:
            topic = action_match.group(1).strip()
            # Remove trailing platform mentions
            topic = re.sub(r'\s+(?:on|for)\s+(?:linkedin|twitter|instagram|facebook).*$', '', topic, flags=re.IGNORECASE)
            if len(topic) > 3:
                return topic
        
        return None
    
    async def _llm_classify(
        self,
        message: str,
        previous_elf: Optional[str],
        session_topic: Optional[str],
    ) -> IntentClassification:
        """Use LLM for detailed intent classification."""
        
        user_prompt = CLASSIFICATION_USER_PROMPT.format(
            user_message=message,
            previous_elf=previous_elf or "None",
            session_topic=session_topic or "None",
        )
        
        messages = [
            LLMMessage(role="system", content=CLASSIFICATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            
            return IntentClassification(
                primary_elf=ElfType(result.get("primary_elf", "assistant")),
                secondary_elves=[ElfType(e) for e in result.get("secondary_elves", [])],
                intent=IntentType(result.get("intent", "general_question")),
                confidence=result.get("confidence", 0.8),
                requires_multi_elf=result.get("requires_multi_elf", False),
                extracted_entities=result.get("extracted_entities", {}),
                reasoning=result.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse classification", error=str(e))
            raise


# Global classifier instance
intent_classifier = IntentClassifier()

