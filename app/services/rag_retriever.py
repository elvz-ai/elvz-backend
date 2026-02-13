"""
RAG (Retrieval Augmented Generation) Retriever Service.

Handles semantic search with query reformulation, token budgeting,
and result ranking for the conversational AI.
"""

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
            "total_tokens": 0,
            "sources_used": [],
        }

        remaining_budget = token_budget

        # Search each context type
        for ctx_type in context_types:
            if remaining_budget <= 0:
                break

            items = await self._search_by_type(
                ctx_type=ctx_type,
                query=search_query,
                user_id=user_id,
                platforms=platforms,
                top_k=top_k,
            )

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

    async def _search_by_type(
        self,
        ctx_type: str,
        query: str,
        user_id: str,
        platforms: Optional[list[str]] = None,
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
            List of search results
        """
        try:
            if ctx_type == "knowledge":
                # Search knowledge base
                results = await vector_store.search_knowledge(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                )
                return [
                    {
                        "type": "knowledge",
                        "content": r.content,
                        "score": r.score,
                        "source": r.metadata.get("source", "knowledge_base"),
                        "category": r.metadata.get("category"),
                    }
                    for r in results
                ]

            elif ctx_type == "examples":
                # Search content examples
                results = await vector_store.search_content_examples(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                    platform=platforms[0] if platforms else None,
                )
                return [
                    {
                        "type": "example",
                        "content": r.content,
                        "score": r.score,
                        "platform": r.metadata.get("platform"),
                        "performance_score": r.metadata.get("performance_score"),
                    }
                    for r in results
                ]

            elif ctx_type == "user_history":
                # Search user's past content
                filters = {}
                if platforms:
                    filters["platform"] = {"$in": platforms}

                results = await vector_store.search_user_content(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                )
                return [
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

        # User's past content
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
