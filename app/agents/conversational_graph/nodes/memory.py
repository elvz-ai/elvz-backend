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

            # Restore last_artifact from working memory for cross-turn modification support
            if not state.get("last_artifact") and working_memory.get("last_artifact"):
                state["last_artifact"] = working_memory["last_artifact"]

            # Restore artifact_history from working memory
            if not state.get("artifact_history") and working_memory.get("artifact_history"):
                state["artifact_history"] = working_memory["artifact_history"]

            # Restore pending_modification context (user was asked "which artifact?")
            if working_memory.get("pending_modification"):
                state["pending_modification"] = working_memory["pending_modification"]

            logger.info(
                "Memory Layer 1 - Working Memory loaded",
                conversation_id=conversation_id,
                working_memory_keys=list(working_memory.keys()),
                working_memory_size=len(working_memory),
            )

            # Layer 2: User profile
            user_profile = await memory_manager.get_user_profile(user_id)
            state["user_profile"] = user_profile

            logger.info(
                "Memory Layer 2 - User Profile loaded",
                user_id=user_id,
                has_profile=user_profile is not None,
                profile_keys=list(user_profile.keys()) if user_profile else [],
            )

            # Layer 3 & 4: RAG retrieval based on intent
            intent = state.get("current_intent") or {}
            intent_type = intent.get("type", "artifact")
            search_modalities = intent.get("search_modalities", ["text"])

            if intent_type in ["artifact", "multi_platform"]:
                # For artifact generation, get specialized context + past conversations for style
                platforms = self._get_platforms(state)

                rag_context = await rag_retriever.retrieve(
                    query=current_input,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_types=["social_history", "conversation"],
                    platforms=platforms,
                    modalities=search_modalities,
                )

                state["retrieved_memory"] = self._flatten_rag_results(rag_context)

            else:
                # For Q&A, clarification, modification — fetch knowledge + past conversations
                rag_context = await rag_retriever.retrieve(
                    query=current_input,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_types=["knowledge", "conversation"],
                    modalities=search_modalities,
                )

                state["retrieved_memory"] = self._flatten_rag_results(rag_context)

            # Format RAG context for prompt injection
            state["rag_context"] = rag_retriever.format_context_for_prompt({
                "social_history": rag_context.get("social_history", []),
                "user_content": rag_context.get("user_history", []),
                "examples": rag_context.get("examples", []),
                "knowledge": rag_context.get("knowledge", []),
                "conversation": rag_context.get("conversation", []),
            })

            logger.info(
                "Memory Layer 3 & 4 - RAG Context retrieved",
                intent_type=intent_type,
                user_history_count=len(rag_context.get("user_history", [])),
                examples_count=len(rag_context.get("examples", [])),
                knowledge_count=len(rag_context.get("knowledge", [])),
                rag_context_length=len(state["rag_context"]),
                total_retrieved=len(state["retrieved_memory"]),
            )

            # Log conversation messages from state
            messages = state.get("messages", [])
            logger.info(
                "Conversation Messages in state",
                message_count=len(messages),
                message_types=[type(m).__name__ for m in messages] if messages else [],
            )

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

        for source_type in ["social_history", "knowledge", "examples", "user_history", "conversation"]:
            items = rag_context.get(source_type, [])
            for item in items:
                item["source_type"] = source_type
                results.append(item)

        return results


# Create node instance
memory_retriever_node = MemoryRetrieverNode()
