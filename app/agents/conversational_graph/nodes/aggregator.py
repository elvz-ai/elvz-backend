"""
Stream Aggregator Node.

Collects and organizes streaming events for the response.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


class StreamAggregatorNode:
    """
    Aggregates streaming events and prepares final response.

    Handles:
    - Collecting events from previous nodes
    - Formatting final response
    - Preparing streaming output
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Aggregate events and prepare response.

        Args:
            state: Current conversation state

        Returns:
            Updated state with aggregated response
        """
        start_time = time.time()
        state["current_node"] = "stream_aggregator"

        add_stream_event(state, "node_started", node="stream_aggregator")

        try:
            # If no final response set yet, generate one
            if not state.get("final_response"):
                state["final_response"] = await self._generate_response(state)

            # Add metadata to response
            metadata = {
                "conversation_id": state["conversation_id"],
                "execution_trace": state.get("execution_trace", []),
                "tokens_used": state.get("total_tokens_used", 0),
                "errors": state.get("errors", []),
            }

            # Add artifact info if present
            artifacts = state.get("artifacts", [])
            if artifacts:
                metadata["artifact_count"] = len(artifacts)
                metadata["batch_id"] = state.get("artifact_batch_id")
                metadata["platforms"] = [a.get("platform") for a in artifacts]

            state["working_memory"]["response_metadata"] = metadata

            # Prepare streaming events summary
            events = state.get("stream_events", [])
            state["working_memory"]["event_count"] = len(events)

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "stream_aggregator",
                "completed",
                execution_time,
                metadata={
                    "has_response": bool(state.get("final_response")),
                    "response_preview": ((state.get("final_response") or "")[:100] + "..."
                                       if state.get("final_response") and len(state.get("final_response") or "") > 100
                                       else (state.get("final_response") or "")),
                    "artifact_count": len(state.get("artifacts", [])),
                    "event_count": len(events),
                }
            )
            add_stream_event(
                state,
                "node_completed",
                content={"has_response": bool(state.get("final_response"))},
                node="stream_aggregator",
            )

            logger.info(
                "Stream aggregation complete",
                event_count=len(events),
                has_response=bool(state.get("final_response")),
            )

        except Exception as e:
            logger.error("Stream aggregation failed", error=str(e))
            state["errors"].append(f"Aggregation error: {str(e)}")

            if not state.get("final_response"):
                state["final_response"] = "I encountered an error. Please try again."

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "stream_aggregator", "failed", execution_time, str(e))

        return state

    async def _generate_response(self, state: ConversationState) -> str:
        """Generate response based on state."""
        intent = state.get("current_intent", {})
        intent_type = intent.get("type", "artifact")

        # Check for errors
        errors = state.get("errors", [])
        if errors and not state.get("artifacts"):
            return (
                "I encountered some issues while processing your request. "
                f"Error: {errors[0]}"
            )

        # Follow-up response
        if state.get("needs_follow_up"):
            questions = state.get("follow_up_questions", [])
            if questions:
                return questions[0]
            return "Could you provide more details about what you'd like?"

        # Guardrail blocked
        if not state.get("guardrail_passed", True):
            return (
                "I'm sorry, but I can't process that request due to content policy violations. "
                "Please rephrase your message."
            )

        # Q&A response (includes clarification/memory questions)
        if intent_type in ["qa", "clarification"]:
            return await self._generate_qa_response(state)

        # Artifact response
        if state.get("artifacts"):
            return self._format_artifact_response(state)

        # Default
        return (
            "I've processed your request. Is there anything else you'd like me to help with?"
        )

    async def _generate_qa_response(self, state: ConversationState) -> str:
        """Generate Q&A response using LLM."""
        query = state["current_input"]
        rag_context = state.get("rag_context", "")
        conversation_history = state.get("messages", [])

        # Log memory being passed
        logger.info(
            "QA response generation - Memory context",
            conversation_history_count=len(conversation_history),
            has_rag_context=bool(rag_context),
            rag_context_length=len(rag_context) if rag_context else 0,
            working_memory_keys=list(state.get("working_memory", {}).keys()),
            user_profile_exists=state.get("user_profile") is not None,
        )

        system_prompt = (
            "You are Elvz, an AI assistant specializing in social media content strategy "
            "and creation. Answer the user's question helpfully and concisely. "
            "If the user seems to want content created, suggest they ask you to generate it."
        )

        # Build messages with conversation history
        messages = [LLMMessage(role="system", content=system_prompt)]

        # Add conversation history (previous messages)
        for msg in conversation_history:
            # LangGraph BaseMessage format
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "user" if msg.type == "human" else "assistant"
                messages.append(LLMMessage(role=role, content=msg.content))

        # Add current query with context
        user_message = query
        if rag_context:
            user_message = f"Context:\n{rag_context}\n\nUser question: {query}"

        messages.append(LLMMessage(role="user", content=user_message))

        logger.info(
            "QA response - Total messages being sent to LLM",
            total_messages=len(messages),
            message_roles=[m.role for m in messages],
        )

        try:
            response = await llm_client.generate_for_task(
                task="response_aggregation",
                messages=messages,
            )
            return response.content
        except Exception as e:
            logger.error("LLM Q&A generation failed", error=str(e))
            return "I'm having trouble generating a response right now. Please try again."

    def _format_artifact_response(self, state: ConversationState) -> str:
        """Format response with artifact information."""
        artifacts = state.get("artifacts", [])

        if not artifacts:
            return "No content was generated."

        if len(artifacts) == 1:
            artifact = artifacts[0]
            platform = (artifact.get("platform") or "").title()
            content = (artifact.get("content") or {})
            text = (content.get("text") or "")

            response_parts = [f"Here's your {platform} post:\n"]
            response_parts.append(f"\n{text}\n")

            hashtags = content.get("hashtags", [])
            if hashtags:
                response_parts.append(f"\n**Hashtags:** {' '.join(hashtags[:10])}")

            schedule = content.get("schedule", {})
            if schedule:
                response_parts.append(f"\n**Best time to post:** {schedule.get('datetime', 'N/A')}")

            return "\n".join(response_parts)

        else:
            # Multiple artifacts
            response_parts = [f"Generated content for {len(artifacts)} platforms:\n"]

            for artifact in artifacts:
                platform = (artifact.get("platform") or "unknown").title()
                content = (artifact.get("content") or {})
                text = (content.get("text") or "")
                preview = text[:150] + "..." if len(text) > 150 else text

                response_parts.append(f"\n**{platform}:**")
                response_parts.append(preview)

            response_parts.append("\n\nWould you like to see the full content for any platform?")
            return "\n".join(response_parts)


# Create node instance
stream_aggregator_node = StreamAggregatorNode()
