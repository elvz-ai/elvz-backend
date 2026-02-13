"""
Context Builder Node.

Builds the LLM context from retrieved memory with token budgeting.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)


class ContextBuilderNode:
    """
    Builds LLM context from retrieved memory.

    Handles:
    - Token budget management
    - Context prioritization
    - Prompt formatting
    """

    def __init__(self):
        self.tokenizer = None
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("Tiktoken not available")

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Build context for LLM prompts.

        Args:
            state: Current conversation state

        Returns:
            Updated state with formatted context
        """
        start_time = time.time()
        state["current_node"] = "context_builder"

        add_stream_event(state, "node_started", node="context_builder")

        try:
            # Build context parts
            context_parts = []
            token_budget = settings.memory_token_budget
            tokens_used = 0

            # 1. User profile context (highest priority)
            if state.get("user_profile"):
                profile_context = self._format_user_profile(state["user_profile"])
                profile_tokens = self._count_tokens(profile_context)

                if tokens_used + profile_tokens <= token_budget:
                    context_parts.append(("user_profile", profile_context))
                    tokens_used += profile_tokens

            # 2. Working memory context
            if state.get("working_memory"):
                memory_context = self._format_working_memory(state["working_memory"])
                memory_tokens = self._count_tokens(memory_context)

                if tokens_used + memory_tokens <= token_budget:
                    context_parts.append(("working_memory", memory_context))
                    tokens_used += memory_tokens

            # 3. RAG context (already formatted and budget-managed)
            if state.get("rag_context"):
                rag_tokens = self._count_tokens(state["rag_context"])
                remaining_budget = token_budget - tokens_used

                if rag_tokens <= remaining_budget:
                    context_parts.append(("rag", state["rag_context"]))
                    tokens_used += rag_tokens
                else:
                    # Truncate RAG context
                    truncated = self._truncate_to_budget(
                        state["rag_context"], remaining_budget
                    )
                    context_parts.append(("rag", truncated))
                    tokens_used += remaining_budget

            # 4. Decomposed query context
            if state.get("decomposed_queries"):
                query_context = self._format_queries(state["decomposed_queries"])
                query_tokens = self._count_tokens(query_context)

                if tokens_used + query_tokens <= token_budget:
                    context_parts.append(("queries", query_context))
                    tokens_used += query_tokens

            # Combine all context
            state["rag_context"] = self._combine_context(context_parts)

            # Update working memory with context metadata
            state["working_memory"]["context_built"] = True
            state["working_memory"]["context_tokens"] = tokens_used

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "context_builder",
                "completed",
                execution_time,
                metadata={"tokens_used": tokens_used},
            )
            add_stream_event(
                state,
                "node_completed",
                content={"tokens_used": tokens_used},
                node="context_builder",
            )

            logger.info("Context built", tokens_used=tokens_used)

        except Exception as e:
            logger.error("Context building failed", error=str(e))
            state["errors"].append(f"Context building error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "context_builder", "failed", execution_time, str(e))

        return state

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_tokens]) + "..."
        else:
            max_chars = max_tokens * 4
            return text[:max_chars] + "..."

    def _format_user_profile(self, profile: dict) -> str:
        """Format user profile for context."""
        parts = ["=== User Profile ==="]

        if profile.get("profile"):
            p = profile["profile"]
            if p.get("brand_name"):
                parts.append(f"Brand: {p['brand_name']}")
            if p.get("industry"):
                parts.append(f"Industry: {p['industry']}")

        if profile.get("brand_voice_context"):
            parts.append(f"\n{profile['brand_voice_context']}")

        return "\n".join(parts)

    def _format_working_memory(self, memory: dict) -> str:
        """Format working memory for context."""
        parts = ["=== Current Context ==="]

        if memory.get("shared_topic"):
            parts.append(f"Topic: {memory['shared_topic']}")
        if memory.get("shared_tone"):
            parts.append(f"Tone: {memory['shared_tone']}")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _format_queries(self, queries: list) -> str:
        """Format decomposed queries for context."""
        if not queries:
            return ""

        parts = ["=== Generation Tasks ==="]
        for q in queries:
            platform = q.get("platform", "unknown")
            topic = q.get("topic", "")
            parts.append(f"- {platform}: {topic or 'Generate content'}")

        return "\n".join(parts)

    def _combine_context(self, context_parts: list) -> str:
        """Combine all context parts into final string."""
        return "\n\n".join(content for _, content in context_parts if content)


# Create node instance
context_builder_node = ContextBuilderNode()
