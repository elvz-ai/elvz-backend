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

            # 0. Append new artifacts to history ring buffer (before persisting)
            self._append_to_artifact_history(state)

            # 1. Update working memory
            await self._save_working_memory(state)

            # 2. Save assistant message (if we have a response)
            if state.get("final_response"):
                await self._save_message(state)

            # 3. Save artifacts to long-term vector memory
            if state.get("artifacts"):
                await self._save_to_long_term_memory(state)

            # 3. Update token usage
            state["total_tokens_used"] += (state.get("working_memory") or {}).get(
                "context_tokens", 0
            )

            # 4. Calculate total execution time
            if state.get("execution_start_time"):
                total_time = datetime.utcnow() - state["execution_start_time"]
                state["working_memory"]["total_execution_ms"] = int(
                    total_time.total_seconds() * 1000
                )

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "memory_saver",
                "completed",
                execution_time,
                metadata={
                    "saved": True,
                    "total_messages": len(state.get("messages", [])),
                    "total_execution_ms": (state.get("working_memory") or {}).get("total_execution_ms", 0),
                }
            )
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
            "last_intent": (state.get("current_intent") or {}).get("type"),
            "intent_history": state.get("intent_history", []),
            # Platform tracking
            "last_platforms": [
                q.get("platform")
                for q in state.get("decomposed_queries", [])
            ],
            # Artifact tracking
            "last_batch_id": state.get("artifact_batch_id"),
            "artifact_count": len(state.get("artifacts", [])),
            "last_artifact": state.get("last_artifact"),  # Persist for cross-turn modification
            # Artifact history ring buffer (for multi-artifact modification resolution)
            "artifact_history": state.get("artifact_history", []),
            # Pending modification context (survives across turns for follow-up flow)
            "pending_modification": state.get("pending_modification"),
            # Context
            "last_topic": (state.get("working_memory") or {}).get("shared_topic"),
            # Timestamps
            "last_updated": datetime.utcnow().isoformat(),
        }

        await memory_manager.update_working_memory(
            conversation_id,
            memory_update,
            merge=True,
        )

    def _append_to_artifact_history(self, state: ConversationState) -> None:
        """Append newly generated artifacts to the history ring buffer."""
        from app.core.config import settings

        artifacts = state.get("artifacts", [])
        if not artifacts:
            return

        history = list(state.get("artifact_history") or [])

        for artifact in artifacts:
            content = artifact.get("content") or {}
            text = content.get("text") or ""
            entry = {
                **artifact,
                "topic_summary": text[:80] if text else "untitled",
                "created_at": datetime.utcnow().isoformat(),
            }
            history.append(entry)

        # Cap at max size (keep most recent)
        max_size = settings.artifact_history_max_size
        if len(history) > max_size:
            history = history[-max_size:]

        state["artifact_history"] = history

    async def _save_message(self, state: ConversationState) -> None:
        """Save assistant message to database and Qdrant for future retrieval."""
        conversation_id = state["conversation_id"]
        response = state.get("final_response", "")
        intent_type = (state.get("current_intent") or {}).get("type", "")

        # Build metadata
        metadata = {
            "intent": intent_type,
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

        # Save conversation turns to Qdrant for semantic search in future turns
        # All intents save the user query so context is always retrievable.
        # For artifacts, we save a short summary instead of the full response
        # (the full artifact content is saved separately via _save_to_long_term_memory).
        await self._save_conversation_turn_to_vector_store(state, response, intent_type=intent_type)

    async def _save_to_long_term_memory(self, state: ConversationState) -> None:
        """Persist generated artifacts to vector store for future retrieval."""
        from uuid import uuid4
        from app.core.vector_store import VectorDocument, vector_store

        artifacts = state.get("artifacts", [])
        if not artifacts:
            return

        documents = []
        for artifact in artifacts:
            content = artifact.get("content") or {}
            text = content.get("text", "")
            if not text:
                continue

            doc = VectorDocument(
                id=artifact.get("id", str(uuid4())),
                content=text,
                metadata={
                    "modality": "text",
                    "content_type": "user_history",
                    "user_id": state["user_id"],
                    "platform": artifact.get("platform", "unknown"),
                    "category": "generated_content",
                    "conversation_id": state["conversation_id"],
                    "created_at": datetime.utcnow().isoformat(),
                    "engagement": {},  # To be updated later
                    "tags": content.get("hashtags", [])[:5] if content.get("hashtags") else [],
                },
            )
            documents.append(doc)

        if documents:
            try:
                await vector_store.add_user_content(state["user_id"], documents)
                logger.info(
                    "Artifacts saved to long-term memory",
                    user_id=state["user_id"],
                    count=len(documents),
                )
            except Exception as e:
                logger.error("Failed to save artifacts to vector store", error=str(e))

    async def _save_conversation_turn_to_vector_store(
        self, state: ConversationState, response: str, intent_type: str = "qa"
    ) -> None:
        """
        Embed and upsert the user query and assistant response to Qdrant.

        This enables semantic search over past conversations so future turns
        can retrieve relevant context (e.g. "what did we discuss about LinkedIn?").

        Called for ALL intent types so users can always retrieve what was discussed.
        For artifact intents, saves a short summary instead of the full response
        (full artifact content is already saved separately via _save_to_long_term_memory).

        Short messages ("ok", "yes") are skipped.
        """
        from uuid import uuid4
        from app.core.config import settings
        from app.core.vector_store import VectorDocument, vector_store

        user_id = state["user_id"]
        conversation_id = state["conversation_id"]
        user_input = state.get("current_input", "")
        now = datetime.utcnow().isoformat()
        min_query_length = settings.rag_min_save_length_query
        min_response_length = settings.rag_min_save_length_response

        # For artifact intents, build a concise summary instead of saving the full response
        # (which may be very long). The actual artifact content is stored separately.
        is_artifact_intent = intent_type in ("artifact", "multi_platform", "modification")
        if is_artifact_intent:
            artifacts = state.get("artifacts", [])
            platforms = list({
                q.get("platform", "unknown")
                for q in state.get("decomposed_queries", [])
                if q.get("platform")
            })
            if artifacts:
                platform_str = ", ".join(platforms) if platforms else "unknown platform"
                save_response = (
                    f"Generated {len(artifacts)} content piece(s) for {platform_str}. "
                    f"User requested: {user_input[:200]}"
                )
            else:
                save_response = response
        else:
            save_response = response

        documents = []

        # Save user query if it's meaningful (short questions are still searchable)
        if len(user_input) >= min_query_length:
            documents.append(
                VectorDocument(
                    id=str(uuid4()),
                    content=user_input,
                    metadata={
                        "modality": "text",
                        "content_type": "user_query",
                        "intent_type": intent_type,
                        "user_id": user_id,
                        "platform": "general",
                        "category": "conversation",
                        "conversation_id": conversation_id,
                        "created_at": now,
                    },
                )
            )

        # Save assistant response (or summary for artifacts) if meaningful
        if len(save_response) >= min_response_length:
            documents.append(
                VectorDocument(
                    id=str(uuid4()),
                    content=save_response,
                    metadata={
                        "modality": "text",
                        "content_type": "qa_response",
                        "intent_type": intent_type,
                        "user_id": user_id,
                        "platform": "general",
                        "category": "conversation",
                        "conversation_id": conversation_id,
                        "created_at": now,
                    },
                )
            )

        if not documents:
            return

        try:
            # Use search_knowledge namespace so conversation content is found
            # by the "conversation" context type in rag_retriever
            await vector_store.add_knowledge(documents, category="conversation")
            logger.info(
                "Conversation turn saved to vector store",
                user_id=user_id,
                conversation_id=conversation_id,
                docs_saved=len(documents),
            )
        except Exception as e:
            logger.error("Failed to save conversation turn to vector store", error=str(e))


# Create node instance
memory_saver_node = MemorySaverNode()
