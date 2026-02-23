"""
RAG (Retrieval Augmented Generation) Retriever Service.

Handles semantic search with query reformulation, token budgeting,
and result ranking for the conversational AI.
"""

import asyncio
import re
from collections import Counter
from typing import Optional

import structlog

from app.core.config import settings
from app.core.vector_store import vector_store
from app.services.memory_manager import memory_manager

logger = structlog.get_logger(__name__)


class RAGRetriever:
    """
    Semantic search and retrieval service for RAG.

    Features:
    - Query reformulation based on conversation context
    - Multi-namespace search (knowledge, examples, user history)
    - Token budget management
    - Result ranking and deduplication
    """

    def __init__(self):
        # Token counter for budget management
        self.tokenizer = None
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("Tiktoken not available, using character-based estimation")

    # ------------------------------------------------------------------
    # Style feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_style_features(self, posts: list[dict]) -> dict:
        """
        Extract Level-3 writing style patterns from a list of posts.

        Analyzes structure, voice, hooks, engagement patterns, and hashtags
        to produce a structured style profile for the LLM.
        """
        captions = [p["content"] for p in posts if p.get("content")]
        if not captions:
            return {}

        # 1. Avg word count
        avg_words = round(sum(len(c.split()) for c in captions) / len(captions))

        # 2. Emoji usage
        emoji_re = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F9FF\U00002702-\U000027B0]+",
            flags=re.UNICODE,
        )
        avg_emojis = round(
            sum(len(emoji_re.findall(c)) for c in captions) / len(captions), 1
        )

        # 3. Line breaks (paragraph rhythm)
        avg_breaks = round(
            sum(c.count("\n") for c in captions) / len(captions), 1
        )

        # 4. Question mark usage
        question_pct = round(
            sum(1 for c in captions if "?" in c) / len(captions) * 100
        )

        # 5. Avg hashtag count
        avg_tags = round(
            sum(len(p.get("hashtags") or []) for p in posts) / len(posts), 1
        )

        # 6. Dominant hook style (first line pattern)
        hooks = []
        for c in captions:
            first = c.split("\n")[0][:120]
            if "?" in first:
                hooks.append("question")
            elif re.match(r"^\d", first):
                hooks.append("number/statistic")
            elif re.match(r"^(I |We |My )", first):
                hooks.append("personal story")
            else:
                hooks.append("bold statement")
        hook_style = Counter(hooks).most_common(1)[0][0]

        # 7. CTA style — ends with question?
        cta_question_pct = round(
            sum(1 for c in captions if c.strip().endswith("?")) / len(captions) * 100
        )

        # 8. Confidence level based on post count
        n = len(posts)
        if n >= 15:
            confidence = "HIGH"
        elif n >= 5:
            confidence = "MEDIUM"
        elif n >= 2:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"

        return {
            "avg_word_count": avg_words,
            "avg_emojis": avg_emojis,
            "avg_line_breaks": avg_breaks,
            "question_frequency_pct": question_pct,
            "avg_hashtags": avg_tags,
            "dominant_hook_style": hook_style,
            "cta_question_pct": cta_question_pct,
            "posts_analyzed": n,
            "confidence": confidence,
        }

    def _format_style_features(self, features: dict) -> str:
        """Format extracted style features as a structured LLM prompt section."""
        if not features:
            return ""
        return (
            f"== Writing Style Profile"
            f" (Confidence: {features['confidence']},"
            f" from {features['posts_analyzed']} posts) ==\n"
            f"- Post length: ~{features['avg_word_count']} words\n"
            f"- Emoji usage: {features['avg_emojis']} per post\n"
            f"- Paragraph breaks: {features['avg_line_breaks']} per post\n"
            f"- Uses questions in body: {features['question_frequency_pct']}% of posts\n"
            f"- Hashtag count: ~{features['avg_hashtags']} per post\n"
            f"- Dominant hook style: {features['dominant_hook_style']}\n"
            f"- Ends with a question (CTA): {features['cta_question_pct']}% of posts"
        )

    # ------------------------------------------------------------------
    # Two-step social history retrieval
    # ------------------------------------------------------------------

    async def _search_social_history_two_step(
        self,
        query: str,
        user_id: str,
        platform: Optional[str] = None,
        top_k: int = 10,
    ) -> tuple[list[dict], dict]:
        """
        Three-tier social history retrieval.

        Tier 1 — Redis cache (~0.1ms): style pre-computed at webhook time.
        Tier 2 — PostgreSQL (~5-20ms): durable fallback if Redis was flushed.
        Tier 3 — Qdrant scroll (~100-300ms): first-time user with no cached data yet.

        Semantic search always runs in real time (topic changes per query).
        Style features are stable between messages so they are cached.
        """
        # Avoid circular import: webhooks.py imports rag_retriever at module level
        from app.core.cache import cache
        from app.core.database import get_db_context
        from app.models.user_style_profile import UserStyleProfile

        # --- Tier 1: Redis (~0.1ms) ---
        cached_style = await cache.get_style_profile(user_id)

        # --- Tier 2: PostgreSQL (~5-20ms) ---
        if not cached_style:
            try:
                async with get_db_context() as db:
                    row = await db.get(UserStyleProfile, user_id)
                    if row:
                        cached_style = row.features
                        # Re-warm Redis so next request is instant
                        await cache.set_style_profile(user_id, cached_style)
                        logger.info(
                            "Style profile loaded from PostgreSQL",
                            user_id=user_id,
                            confidence=cached_style.get("confidence"),
                        )
            except Exception as e:
                logger.warning("PostgreSQL style profile lookup failed", error=str(e))

        # --- Fast path: style is known, only run semantic search ---
        if cached_style:
            semantic_results = await vector_store.search_social_content(
                user_id=user_id, query=query, platform=platform, top_k=top_k
            )
            min_score = self._get_min_score("social_history")
            semantic_items = [
                {
                    "type": "social_history",
                    "content": r.content,
                    "score": r.score,
                    "platform": r.metadata.get("platform"),
                    "posted_at": r.metadata.get("posted_at", ""),
                    "hashtags": r.metadata.get("hashtags", []),
                    "engagement": r.metadata.get("engagement"),
                    "performance_score": r.metadata.get("performance_score", 0),
                    "source": "semantic",
                }
                for r in semantic_results
                if r.score >= min_score
            ]
            logger.info(
                "Social history retrieved (style from cache)",
                user_id=user_id,
                platform=platform,
                semantic_count=len(semantic_items),
                style_confidence=cached_style.get("confidence", "unknown"),
            )
            return semantic_items[:top_k], cached_style

        # --- Tier 3: Qdrant scroll (first-time user, no cached data yet) ---
        baseline_task = vector_store.fetch_top_social_by_score(
            user_id=user_id, top_n=10
        )
        semantic_task = vector_store.search_social_content(
            user_id=user_id, query=query, platform=platform, top_k=top_k
        )
        top_posts, semantic_results = await asyncio.gather(baseline_task, semantic_task)

        min_score = self._get_min_score("social_history")
        semantic_items = [
            {
                "type": "social_history",
                "content": r.content,
                "score": r.score,
                "platform": r.metadata.get("platform"),
                "posted_at": r.metadata.get("posted_at", ""),
                "hashtags": r.metadata.get("hashtags", []),
                "engagement": r.metadata.get("engagement"),
                "performance_score": r.metadata.get("performance_score", 0),
                "source": "semantic",
            }
            for r in semantic_results
            if r.score >= min_score
        ]

        for p in top_posts:
            p["source"] = "style_baseline"

        seen: set[str] = set()
        merged: list[dict] = []
        for item in semantic_items + top_posts:
            key = (item.get("content") or "")[:80]
            if key and key not in seen:
                merged.append(item)
                seen.add(key)

        style_features = self._extract_style_features(top_posts[:10])

        # Lazily populate cache so subsequent requests skip the Qdrant scroll
        if style_features:
            try:
                await cache.set_style_profile(user_id, style_features)
                async with get_db_context() as db:
                    existing = await db.get(UserStyleProfile, user_id)
                    if existing:
                        existing.features = style_features
                        existing.posts_analyzed = style_features["posts_analyzed"]
                        existing.confidence = style_features["confidence"]
                    else:
                        db.add(UserStyleProfile(
                            user_id=user_id,
                            features=style_features,
                            posts_analyzed=style_features["posts_analyzed"],
                            confidence=style_features["confidence"],
                        ))
            except Exception as e:
                logger.warning("Failed to lazily cache style profile", error=str(e))

        logger.info(
            "Social history retrieved (Qdrant scroll — cold start)",
            user_id=user_id,
            platform=platform,
            baseline_count=len(top_posts),
            semantic_count=len(semantic_items),
            style_confidence=style_features.get("confidence", "none"),
        )

        return merged[:top_k], style_features

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: ~4 chars per token
        return len(text) // 4

    async def retrieve(
        self,
        query: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        context_types: Optional[list[str]] = None,
        platforms: Optional[list[str]] = None,
        modalities: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> dict:
        """
        Retrieve relevant context for RAG.

        Args:
            query: Search query
            user_id: User identifier
            conversation_id: Optional conversation for context
            context_types: Types to search ["knowledge", "examples", "user_history"]
            platforms: Platform filters for user content
            top_k: Results per source
            token_budget: Max tokens for all retrieved content

        Returns:
            Dict with retrieved context organized by type
        """
        context_types = context_types or ["knowledge", "user_history"]
        modalities = modalities or ["text"]
        top_k = top_k or settings.memory_rag_top_k
        token_budget = token_budget or settings.memory_token_budget

        # Reformulate query if we have conversation context
        search_query = query
        if conversation_id:
            search_query = await self._reformulate_query(query, conversation_id)

        results = {
            "query": query,
            "reformulated_query": search_query,
            "knowledge": [],
            "examples": [],
            "user_history": [],
            "social_history": [],
            "style_features": {},
            "total_tokens": 0,
            "sources_used": [],
        }

        remaining_budget = token_budget

        # Fire all searches in parallel (instead of sequential)
        search_coros = [
            self._search_by_type(
                ctx_type=ctx_type,
                query=search_query,
                user_id=user_id,
                platforms=platforms,
                modalities=modalities,
                top_k=top_k,
            )
            for ctx_type in context_types
        ]
        search_outputs = await asyncio.gather(*search_coros, return_exceptions=True)

        # Apply token budget sequentially (preserves priority order)
        for ctx_type, output in zip(context_types, search_outputs):
            if remaining_budget <= 0:
                break

            if isinstance(output, Exception):
                logger.error(f"Parallel search failed for {ctx_type}", error=str(output))
                continue

            items = output

            # Extract style_features sentinel before token budget
            if ctx_type == "social_history":
                sentinel = next(
                    (i for i in items if i.get("type") == "_style_features"), None
                )
                if sentinel:
                    items = [i for i in items if i.get("type") != "_style_features"]
                    results["style_features"] = sentinel["features"]

            # Apply token budget
            filtered_items, tokens_used = self._apply_token_budget(
                items, remaining_budget
            )

            if filtered_items:
                results[ctx_type] = filtered_items
                results["sources_used"].append(ctx_type)
                results["total_tokens"] += tokens_used
                remaining_budget -= tokens_used

        logger.info(
            "RAG retrieval complete",
            query_length=len(query),
            sources=results["sources_used"],
            total_tokens=results["total_tokens"],
        )

        return results

    async def _reformulate_query(
        self,
        query: str,
        conversation_id: str,
    ) -> str:
        """
        Reformulate query based on conversation context.

        Handles pronoun resolution and context expansion.

        Args:
            query: Original query
            conversation_id: Conversation for context

        Returns:
            Reformulated query
        """
        # Get recent messages for context
        recent_messages = await memory_manager.get_recent_messages(
            conversation_id, limit=5
        )

        if not recent_messages:
            return query

        # Check for pronouns or references that need resolution
        reference_words = ["it", "this", "that", "they", "them", "the post", "the content"]
        query_lower = query.lower()

        needs_reformulation = any(
            word in query_lower for word in reference_words
        )

        if not needs_reformulation:
            return query

        # Build context from recent messages
        context_parts = []
        for msg in recent_messages[-3:]:  # Last 3 messages
            if msg.get("role") == "user":
                context_parts.append(f"User asked about: {msg.get('content', '')[:200]}")
            elif msg.get("role") == "assistant":
                # Extract topic/platform from assistant responses
                metadata = msg.get("metadata", {})
                if metadata.get("platform"):
                    context_parts.append(f"Platform: {metadata['platform']}")
                if metadata.get("topic"):
                    context_parts.append(f"Topic: {metadata['topic']}")

        # Simple reformulation: append context
        if context_parts:
            context_str = " | ".join(context_parts[-2:])
            reformulated = f"{query} (context: {context_str})"
            logger.debug(
                "Query reformulated",
                original=query,
                reformulated=reformulated[:200],
            )
            return reformulated

        return query

    def _get_min_score(self, ctx_type: str) -> float:
        """Return the minimum relevance score threshold for a context type."""
        thresholds = {
            "knowledge": settings.rag_min_score_knowledge,
            "examples": settings.rag_min_score_knowledge,
            "user_history": settings.rag_min_score_user_history,
            "social_history": settings.rag_min_score_user_history,
            "conversation": settings.rag_min_score_conversation,
        }
        return thresholds.get(ctx_type, settings.rag_min_score_knowledge)

    def _filter_by_score(self, results: list[dict], min_score: float, ctx_type: str) -> list[dict]:
        """Filter results below the minimum relevance score."""
        filtered = [r for r in results if r.get("score", 0) >= min_score]
        dropped = len(results) - len(filtered)
        if dropped > 0:
            logger.info(
                "RAG relevance filtering",
                ctx_type=ctx_type,
                original=len(results),
                kept=len(filtered),
                dropped=dropped,
                min_score=min_score,
            )
        return filtered

    async def _search_by_type(
        self,
        ctx_type: str,
        query: str,
        user_id: str,
        platforms: Optional[list[str]] = None,
        modalities: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search a specific context type.

        Args:
            ctx_type: Type of context to search
            query: Search query
            user_id: User identifier
            platforms: Platform filters
            top_k: Number of results

        Returns:
            List of search results filtered by relevance score
        """
        min_score = self._get_min_score(ctx_type)

        try:
            if ctx_type == "knowledge":
                results = await vector_store.search_knowledge(
                    query=query,
                    user_id=user_id,
                    modalities=modalities,
                    top_k=top_k,
                )
                items = [
                    {
                        "type": "knowledge",
                        "content": r.content,
                        "score": r.score,
                        "source": r.metadata.get("source", "knowledge_base"),
                        "category": r.metadata.get("category"),
                    }
                    for r in results
                ]
                return self._filter_by_score(items, min_score, ctx_type)

            elif ctx_type == "examples":
                results = await vector_store.search_content_examples(
                    query=query,
                    user_id=user_id,
                    platform=platforms[0] if platforms else None,
                    modalities=modalities,
                    top_k=top_k,
                )
                items = [
                    {
                        "type": "example",
                        "content": r.content,
                        "score": r.score,
                        "platform": r.metadata.get("platform"),
                        "performance_score": r.metadata.get("performance_score"),
                    }
                    for r in results
                ]
                return self._filter_by_score(items, min_score, ctx_type)

            elif ctx_type == "user_history":
                results = await vector_store.search_user_content(
                    query=query,
                    user_id=user_id,
                    modalities=modalities,
                    top_k=top_k,
                )
                items = [
                    {
                        "type": "user_content",
                        "content": r.content,
                        "score": r.score,
                        "platform": r.metadata.get("platform"),
                        "posted_at": r.metadata.get("posted_at"),
                        "engagement": r.metadata.get("engagement_metrics"),
                    }
                    for r in results
                ]
                return self._filter_by_score(items, min_score, ctx_type)

            elif ctx_type == "social_history":
                # Two-step retrieval: style baseline (always) + semantic (optional)
                items, style_features = await self._search_social_history_two_step(
                    query=query,
                    user_id=user_id,
                    platform=platforms[0] if platforms else None,
                    top_k=top_k,
                )
                # Attach style_features as a sentinel so retrieve() can surface it
                if style_features:
                    items.append({"type": "_style_features", "features": style_features})
                return items  # score filtering already applied inside two_step

            elif ctx_type == "conversation":
                # Search past conversation turns (user queries + QA responses)
                results = await vector_store.search_knowledge(
                    query=query,
                    user_id=user_id,
                    modalities=modalities,
                    top_k=top_k,
                )
                # Filter to only conversation content_types
                items = [
                    {
                        "type": "conversation",
                        "content": r.content,
                        "score": r.score,
                        "conversation_id": r.metadata.get("conversation_id"),
                        "content_type": r.metadata.get("content_type"),
                        "created_at": r.metadata.get("created_at"),
                    }
                    for r in results
                    if r.metadata.get("content_type") in ("user_query", "qa_response")
                ]
                return self._filter_by_score(items, min_score, ctx_type)

            else:
                logger.warning(f"Unknown context type: {ctx_type}")
                return []

        except Exception as e:
            logger.error(f"Search failed for {ctx_type}", error=str(e))
            return []

    def _apply_token_budget(
        self,
        items: list[dict],
        budget: int,
    ) -> tuple[list[dict], int]:
        """
        Filter items to fit within token budget.

        Args:
            items: List of search results
            budget: Remaining token budget

        Returns:
            Tuple of (filtered items, tokens used)
        """
        if not items:
            return [], 0

        filtered = []
        total_tokens = 0

        for item in items:
            content = item.get("content", "")
            tokens = self.count_tokens(content)

            if total_tokens + tokens <= budget:
                filtered.append(item)
                total_tokens += tokens
            else:
                # Try to include truncated content
                remaining = budget - total_tokens
                if remaining > 100:  # Minimum useful content
                    truncated_content = self._truncate_to_tokens(content, remaining)
                    item["content"] = truncated_content
                    item["truncated"] = True
                    filtered.append(item)
                    total_tokens = budget
                break

        return filtered, total_tokens

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens) + "..."
        else:
            # Fallback: character-based truncation
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."

    async def retrieve_for_artifact(
        self,
        topic: str,
        platform: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> dict:
        """
        Specialized retrieval for artifact generation.

        Retrieves:
        - User's past content on this platform for style matching
        - High-performing examples for inspiration
        - Knowledge about best practices

        Args:
            topic: Content topic
            platform: Target platform
            user_id: User identifier
            conversation_id: Optional conversation context

        Returns:
            Organized context for artifact generation
        """
        # Build query
        query = f"{topic} {platform} social media post"

        # Allocate token budget
        total_budget = settings.memory_token_budget
        user_budget = int(total_budget * 0.5)  # 50% for user content
        examples_budget = int(total_budget * 0.3)  # 30% for examples
        knowledge_budget = int(total_budget * 0.2)  # 20% for knowledge

        context = {
            "topic": topic,
            "platform": platform,
            "user_content": [],
            "examples": [],
            "knowledge": [],
            "total_tokens": 0,
        }

        # Get user's past content for this platform
        user_results = await self._search_by_type(
            ctx_type="user_history",
            query=query,
            user_id=user_id,
            platforms=[platform],
            top_k=10,
        )
        user_filtered, user_tokens = self._apply_token_budget(user_results, user_budget)
        context["user_content"] = user_filtered
        context["total_tokens"] += user_tokens

        # Get high-performing examples
        example_results = await self._search_by_type(
            ctx_type="examples",
            query=query,
            user_id=user_id,
            platforms=[platform],
            top_k=5,
        )
        examples_filtered, examples_tokens = self._apply_token_budget(
            example_results, examples_budget
        )
        context["examples"] = examples_filtered
        context["total_tokens"] += examples_tokens

        # Get best practices knowledge
        knowledge_results = await self._search_by_type(
            ctx_type="knowledge",
            query=f"{platform} best practices {topic}",
            user_id=user_id,
            top_k=3,
        )
        knowledge_filtered, knowledge_tokens = self._apply_token_budget(
            knowledge_results, knowledge_budget
        )
        context["knowledge"] = knowledge_filtered
        context["total_tokens"] += knowledge_tokens

        logger.info(
            "Artifact context retrieved",
            platform=platform,
            user_content_count=len(context["user_content"]),
            examples_count=len(context["examples"]),
            knowledge_count=len(context["knowledge"]),
            total_tokens=context["total_tokens"],
        )

        return context

    def format_context_for_prompt(self, context: dict) -> str:
        """
        Format retrieved context for LLM prompt.

        Args:
            context: Retrieved context dict

        Returns:
            Formatted string for prompt injection
        """
        parts = []

        # Past conversation turns (user queries + QA responses)
        if context.get("conversation"):
            parts.append("=== Relevant Past Conversations ===")
            for item in context["conversation"][:4]:
                content_type = item.get("content_type", "conversation")
                label = "User asked" if content_type == "user_query" else "You answered"
                content = item.get("content", "")[:400]
                parts.append(f"[{label}] {content}")
            parts.append("")

        # User's scraped social posts — style profile + examples
        if context.get("social_history"):
            # 1. Style profile summary (Level-3 structured features)
            style_features = context.get("style_features", {})
            if style_features:
                parts.append(self._format_style_features(style_features))
                parts.append("")

            # 2. Topic-relevant posts (semantic matches — optional)
            semantic = [
                p for p in context["social_history"]
                if p.get("source") == "semantic"
            ]
            if semantic:
                parts.append("=== Topic-Relevant Posts (Style + Topic Context) ===")
                for item in semantic[:3]:
                    pf = item.get("platform", "unknown")
                    content = (item.get("content") or "")[:500]
                    tags = " ".join((item.get("hashtags") or [])[:5])
                    eng = item.get("engagement") or {}
                    perf = item.get("performance_score", 0)
                    parts.append(f"[{pf}] {content}")
                    if tags:
                        parts.append(f"  Tags: {tags}")
                    if eng:
                        parts.append(
                            f"  Engagement: {eng.get('likes', 0)} likes,"
                            f" {eng.get('comments', 0)} comments,"
                            f" {eng.get('shares', 0)} shares"
                            f" (performance score: {perf})"
                        )
                parts.append("")

            # 3. Style baseline posts (top by performance × recency — always present)
            baseline = [
                p for p in context["social_history"]
                if p.get("source") == "style_baseline"
            ]
            if baseline:
                parts.append("=== Your Best Posts (Style Baseline) ===")
                for item in baseline[:3]:
                    pf = item.get("platform", "unknown")
                    content = (item.get("content") or "")[:500]
                    tags = " ".join((item.get("hashtags") or [])[:5])
                    perf = item.get("performance_score", 0)
                    parts.append(f"[{pf}] {content}")
                    if tags:
                        parts.append(f"  Tags: {tags}")
                    parts.append(f"  Performance score: {perf}")
                parts.append("")

        # User's past content (from elvz_memory)
        if context.get("user_content"):
            parts.append("=== Your Past Content (Style Reference) ===")
            for item in context["user_content"][:5]:
                platform = item.get("platform", "unknown")
                content = item.get("content", "")[:500]
                parts.append(f"[{platform}] {content}")
            parts.append("")

        # Examples
        if context.get("examples"):
            parts.append("=== High-Performing Examples ===")
            for item in context["examples"][:3]:
                platform = item.get("platform", "unknown")
                content = item.get("content", "")[:400]
                score = item.get("performance_score", "N/A")
                parts.append(f"[{platform}, Score: {score}] {content}")
            parts.append("")

        # Knowledge
        if context.get("knowledge"):
            parts.append("=== Best Practices ===")
            for item in context["knowledge"][:3]:
                content = item.get("content", "")[:300]
                parts.append(f"- {content}")
            parts.append("")

        return "\n".join(parts) if parts else ""


# Global instance
rag_retriever = RAGRetriever()
