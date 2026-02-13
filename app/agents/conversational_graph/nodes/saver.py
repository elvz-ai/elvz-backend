"""
Memory Saver Node.

Persists conversation state to memory layers.
"""

import time
from datetime import datetime

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.services.memory_manager import memory_manager

logger = structlog.get_logger(__name__)


class MemorySaverNode:
    """
    Persists conversation state to memory.

    Saves:
    - Working memory updates (Redis)
    - Message to short-term memory (PostgreSQL)
    - Conversation metadata updates
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Save state to memory layers.

        Args:
            state: Current conversation state

        Returns:
            Updated state
        """
        start_time = time.time()
        state["current_node"] = "memory_saver"

        add_stream_event(state, "node_started", node="memory_saver")

        try:
            conversation_id = state["conversation_id"]
            user_id = state["user_id"]

            # 1. Update working memory
            await self._save_working_memory(state)

            # 2. Save assistant message (if we have a response)
            if state.get("final_response"):
                await self._save_message(state)

            # 3. Update token usage
            state["total_tokens_used"] += state.get("working_memory", {}).get(
                "context_tokens", 0
            )

            # 4. Calculate total execution time
            if state.get("execution_start_time"):
                total_time = datetime.utcnow() - state["execution_start_time"]
                state["working_memory"]["total_execution_ms"] = int(
                    total_time.total_seconds() * 1000
                )

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "memory_saver", "completed", execution_time)
            add_stream_event(
                state,
                "node_completed",
                content={"saved": True},
                node="memory_saver",
            )

            # Final completion event
            add_stream_event(
                state,
                "generation_complete",
                content={
                    "response": state.get("final_response"),
                    "artifacts": len(state.get("artifacts", [])),
                    "suggestions": state.get("suggestions", []),
                },
                metadata={
                    "conversation_id": conversation_id,
                    "batch_id": state.get("artifact_batch_id"),
                },
            )

            logger.info("Memory saved", conversation_id=conversation_id)

        except Exception as e:
            logger.error("Memory save failed", error=str(e))
            state["errors"].append(f"Memory save error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "memory_saver", "failed", execution_time, str(e))

        return state

    async def _save_working_memory(self, state: ConversationState) -> None:
        """Save working memory to Redis."""
        conversation_id = state["conversation_id"]

        # Prepare working memory update
        memory_update = {
            # Intent tracking
            "last_intent": state.get("current_intent", {}).get("type"),
            "intent_history": state.get("intent_history", []),
            # Platform tracking
            "last_platforms": [
                q.get("platform")
                for q in state.get("decomposed_queries", [])
            ],
            # Artifact tracking
            "last_batch_id": state.get("artifact_batch_id"),
            "artifact_count": len(state.get("artifacts", [])),
            # Context
            "last_topic": state.get("working_memory", {}).get("shared_topic"),
            # Timestamps
            "last_updated": datetime.utcnow().isoformat(),
        }

        await memory_manager.update_working_memory(
            conversation_id,
            memory_update,
            merge=True,
        )

    async def _save_message(self, state: ConversationState) -> None:
        """Save assistant message to database."""
        conversation_id = state["conversation_id"]
        response = state.get("final_response", "")

        # Build metadata
        metadata = {
            "intent": state.get("current_intent", {}).get("type"),
            "platforms": [
                q.get("platform")
                for q in state.get("decomposed_queries", [])
            ],
            "artifact_count": len(state.get("artifacts", [])),
            "batch_id": state.get("artifact_batch_id"),
            "tokens_used": state.get("total_tokens_used", 0),
            "execution_trace": [
                {
                    "node": t.get("node"),
                    "status": t.get("status"),
                    "time_ms": t.get("time_ms"),
                }
                for t in state.get("execution_trace", [])
            ],
        }

        # Add errors if any
        if state.get("errors"):
            metadata["errors"] = state["errors"]

        await memory_manager.save_message_to_memory(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
            metadata=metadata,
        )


# Create node instance
memory_saver_node = MemorySaverNode()
