"""
Query Decomposer Node.

Decomposes complex multi-platform queries into sub-queries.
"""

import json
import time
from typing import Optional

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    DecomposedQuery,
    add_execution_trace,
    add_stream_event,
)
from app.core.config import settings
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


DECOMPOSITION_PROMPT = """Analyze this query and decompose it for content generation.

Query: "{user_query}"

Detected Intent: {intent_type}
Detected Entities: {entities}

Available platforms: LinkedIn, Instagram, Facebook, Twitter, TikTok
Maximum total queries: 5

Rules:
1. Multiple PLATFORMS → one query per platform (e.g., "LinkedIn and Twitter" → 2 queries)
2. Multiple POSTS for the same platform → that many queries with the same platform
   (e.g., "2 LinkedIn posts" → 2 queries both with platform: linkedin)
3. Combination → expand fully (e.g., "2 for LinkedIn and 1 for Twitter" → 3 queries)
4. Single platform, single post → one query (variation_index: 1, variation_total: 1)
5. Never exceed 5 total queries — cap silently if the user asks for more

For same-platform batches, set variation_index (1-based) and variation_total on each query.
For single posts or multi-platform posts, set variation_index: 1 and variation_total: 1.

Respond in JSON:
{{
    "is_multi_platform": true/false,
    "shared_topic": "common topic",
    "shared_tone": "professional|casual|humorous|inspirational|null",
    "queries": [
        {{
            "platform": "linkedin",
            "query": "specific task for this platform",
            "topic": "topic for this query",
            "priority": 1,
            "variation_index": 1,
            "variation_total": 1
        }}
    ]
}}
"""


class QueryDecomposerNode:
    """
    Decomposes complex queries into platform-specific sub-queries.

    Handles:
    - Multi-platform requests (e.g., "Create posts for LinkedIn and Facebook")
    - Implicit multi-platform (e.g., "Post about AI on all platforms")
    - Single platform queries (pass through)
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Decompose query into platform-specific sub-queries.

        Args:
            state: Current conversation state

        Returns:
            Updated state with decomposed queries
        """
        start_time = time.time()
        state["current_node"] = "query_decomposer"

        add_stream_event(state, "node_started", node="query_decomposer")

        try:
            intent = state.get("current_intent") or {}

            # Only decompose for artifact or multi_platform intents
            if intent.get("type") not in ["artifact", "multi_platform"]:
                # For Q&A, clarification, etc., create single passthrough query
                state["decomposed_queries"] = [
                    DecomposedQuery(
                        platform="general",
                        query=state["current_input"],
                        topic=(intent.get("entities") or {}).get("topic", ""),
                        priority=1,
                        status="pending",
                        variation_index=1,
                        variation_total=1,
                    )
                ]
                state["decomposition_complete"] = True

            else:
                # ALL artifact + multi_platform requests → LLM decomposition.
                # The LLM handles: single post, multi-platform, quantity batches,
                # and combinations (e.g. "2 for LinkedIn and 1 for Twitter").
                decomposition = await self._decompose_query(
                    state["current_input"],
                    intent,
                )

                state["decomposed_queries"] = decomposition["queries"]
                state["is_multi_platform"] = decomposition["is_multi_platform"]

                # Store shared context in working memory
                state["working_memory"]["shared_topic"] = decomposition.get("shared_topic")
                state["working_memory"]["shared_tone"] = decomposition.get("shared_tone")

            state["decomposition_complete"] = True
            state["active_query_index"] = 0

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "query_decomposer",
                "completed",
                execution_time,
                metadata={
                    "query_count": len(state["decomposed_queries"]),
                    "is_multi_platform": state.get("is_multi_platform", False),
                    "platforms": [q.get("platform") for q in state["decomposed_queries"]],
                    "queries": [
                        {
                            "platform": q.get("platform"),
                            "topic": (q.get("topic") or "")[:50],  # Truncate safely
                        }
                        for q in state["decomposed_queries"]
                    ],
                }
            )
            add_stream_event(
                state,
                "node_completed",
                content={
                    "query_count": len(state["decomposed_queries"]),
                    "is_multi_platform": state.get("is_multi_platform", False),
                },
                node="query_decomposer",
            )

            logger.info(
                "Query decomposed",
                query_count=len(state["decomposed_queries"]),
                is_multi_platform=state.get("is_multi_platform", False),
            )

        except Exception as e:
            logger.error("Query decomposition failed", error=str(e))
            # Fallback: single query
            state["decomposed_queries"] = [
                DecomposedQuery(
                    platform="linkedin",
                    query=state["current_input"],
                    topic="",
                    priority=1,
                    status="pending",
                )
            ]
            state["decomposition_complete"] = True
            state["errors"].append(f"Decomposition error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "query_decomposer", "failed", execution_time, str(e))

        return state

    async def _decompose_query(
        self,
        user_query: str,
        intent: dict,
    ) -> dict:
        """
        Use LLM to decompose multi-platform query.

        Args:
            user_query: User's query
            intent: Classified intent

        Returns:
            Decomposition result dict
        """
        prompt = DECOMPOSITION_PROMPT.format(
            user_query=user_query,
            intent_type=intent.get("type", "artifact"),
            entities=json.dumps(intent.get("entities", {})),
        )

        messages = [
            LLMMessage(
                role="system",
                content="You decompose content generation requests. Respond with valid JSON only.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await llm_client.generate_fast(messages, json_mode=True)
            result = json.loads(response.content)

            # Validate and limit platforms
            queries = result.get("queries", [])
            if len(queries) > settings.max_platforms_per_query:
                queries = queries[:settings.max_platforms_per_query]

            # Convert to DecomposedQuery objects
            decomposed = [
                DecomposedQuery(
                    platform=(q.get("platform") or "linkedin").lower(),
                    query=q.get("query") or user_query,
                    topic=q.get("topic") or result.get("shared_topic") or "",
                    priority=q.get("priority", i + 1),
                    status="pending",
                    variation_index=q.get("variation_index", 1),
                    variation_total=q.get("variation_total", 1),
                )
                for i, q in enumerate(queries)
            ]

            return {
                "is_multi_platform": len(decomposed) > 1,
                "shared_topic": result.get("shared_topic"),
                "shared_tone": result.get("shared_tone"),
                "queries": decomposed,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse decomposition JSON: {e}")
            return self._fallback_decompose(user_query, intent)

    def _fallback_decompose(self, user_query: str, intent: dict) -> dict:
        """Fallback decomposition using simple pattern matching."""
        query_lower = user_query.lower()
        platforms = []

        # Detect platforms
        platform_map = {
            "linkedin": "linkedin",
            "instagram": "instagram",
            "facebook": "facebook",
            "twitter": "twitter",
            "tiktok": "tiktok",
        }

        for name, platform in platform_map.items():
            if name in query_lower:
                platforms.append(platform)

        # Check for "all platforms"
        if "all platform" in query_lower or len(platforms) == 0:
            platforms = ["linkedin", "instagram", "facebook"]

        # Limit platforms
        platforms = platforms[:settings.max_platforms_per_query]

        # Extract topic (simple heuristic)
        topic = (intent.get("entities") or {}).get("topic", "")

        queries = [
            DecomposedQuery(
                platform=p,
                query=f"Generate {p} post about {topic}" if topic else user_query,
                topic=topic,
                priority=i + 1,
                status="pending",
            )
            for i, p in enumerate(platforms)
        ]

        return {
            "is_multi_platform": len(queries) > 1,
            "shared_topic": topic,
            "shared_tone": None,
            "queries": queries,
        }

    def _extract_platform(self, query: str) -> Optional[str]:
        """Extract platform from query text."""
        query_lower = query.lower()

        platforms = ["linkedin", "instagram", "facebook", "twitter", "tiktok"]
        for p in platforms:
            if p in query_lower:
                return p

        return None


# Create node instance
query_decomposer_node = QueryDecomposerNode()
