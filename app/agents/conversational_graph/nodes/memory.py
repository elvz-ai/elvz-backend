"""
Memory Retriever Node.

Retrieves relevant context from all memory layers.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.services.memory_manager import memory_manager
from app.services.rag_retriever import rag_retriever

logger = structlog.get_logger(__name__)


class MemoryRetrieverNode:
    """
    Retrieves relevant context from multi-layer memory.

    Layers accessed:
    1. Working memory (Redis) - Current conversation context
    2. Short-term memory (PostgreSQL) - Recent messages
    3. Long-term memory (Vector Store) - Semantic search
    4. User profile memory (PostgreSQL + Cache) - Brand voice
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Retrieve memory context for the current turn.

        Args:
            state: Current conversation state

        Returns:
            Updated state with retrieved memory
        """
        start_time = time.time()
        state["current_node"] = "memory_retriever"

        add_stream_event(state, "node_started", node="memory_retriever")

        try:
            user_id = state["user_id"]
            conversation_id = state["conversation_id"]
            current_input = state["current_input"]

            # Layer 1: Working memory
            working_memory = await memory_manager.get_working_memory(conversation_id)
            state["working_memory"].update(working_memory)

            # Layer 2: User profile
            user_profile = await memory_manager.get_user_profile(user_id)
            state["user_profile"] = user_profile

            # Layer 3 & 4: RAG retrieval based on intent
            intent_type = (state.get("current_intent") or {}).get("type", "artifact")

            if intent_type in ["artifact", "multi_platform"]:
                # For artifact generation, get specialized context
                platforms = self._get_platforms(state)

                rag_context = await rag_retriever.retrieve(
                    query=current_input,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_types=["user_history", "knowledge", "examples"],
                    platforms=platforms,
                )

                state["retrieved_memory"] = self._flatten_rag_results(rag_context)

            else:
                # For Q&A or clarification, lighter retrieval
                rag_context = await rag_retriever.retrieve(
                    query=current_input,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_types=["knowledge"],
                )

                state["retrieved_memory"] = self._flatten_rag_results(rag_context)

            # Format RAG context for prompt injection
            state["rag_context"] = rag_retriever.format_context_for_prompt({
                "user_content": rag_context.get("user_history", []),
                "examples": rag_context.get("examples", []),
                "knowledge": rag_context.get("knowledge", []),
            })

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "memory_retriever",
                "completed",
                execution_time,
                metadata={
                    "retrieved_count": len(state["retrieved_memory"]),
                    "has_user_profile": state["user_profile"] is not None,
                },
            )
            add_stream_event(
                state,
                "node_completed",
                content={"retrieved_count": len(state["retrieved_memory"])},
                node="memory_retriever",
            )

            logger.info(
                "Memory retrieved",
                retrieved_count=len(state["retrieved_memory"]),
                has_profile=state["user_profile"] is not None,
            )

        except Exception as e:
            logger.error("Memory retrieval failed", error=str(e))
            state["errors"].append(f"Memory retrieval error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "memory_retriever", "failed", execution_time, str(e))

        return state

    def _get_platforms(self, state: ConversationState) -> list[str]:
        """Extract platforms from state."""
        # From decomposed queries
        queries = state.get("decomposed_queries", [])
        if queries:
            return [q.get("platform") for q in queries if q.get("platform")]

        # From intent entities
        intent = state.get("current_intent", {})
        entities = intent.get("entities", {})

        if entities.get("platforms"):
            return entities["platforms"]
        if entities.get("platform"):
            return [entities["platform"]]

        return []

    def _flatten_rag_results(self, rag_context: dict) -> list[dict]:
        """Flatten RAG results into a single list."""
        results = []

        for source_type in ["knowledge", "examples", "user_history"]:
            items = rag_context.get(source_type, [])
            for item in items:
                item["source_type"] = source_type
                results.append(item)

        return results


# Create node instance
memory_retriever_node = MemoryRetrieverNode()
