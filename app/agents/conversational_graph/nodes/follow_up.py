"""
Follow-up Handler Nodes.

Detects when follow-up questions are needed and generates them.
"""

import json
import time
from typing import Optional

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


class FollowUpDetectorNode:
    """
    Detects when follow-up questions are needed.

    Triggers for:
    - Missing platform specification
    - Missing topic
    - Ambiguous references
    - Insufficient context
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Check if follow-up questions are needed.

        Args:
            state: Current conversation state

        Returns:
            Updated state with follow-up detection
        """
        start_time = time.time()
        state["current_node"] = "follow_up_detector"

        add_stream_event(state, "node_started", node="follow_up_detector")

        try:
            intent = state.get("current_intent", {})
            entities = intent.get("entities", {})
            intent_type = intent.get("type", "artifact")

            # If router already flagged ambiguous_artifact, preserve that decision
            if state.get("follow_up_type") == "ambiguous_artifact":
                state["needs_follow_up"] = True
                execution_time = int((time.time() - start_time) * 1000)
                add_execution_trace(
                    state, "follow_up_detector", "completed", execution_time,
                    metadata={"needs_follow_up": True, "type": "ambiguous_artifact"},
                )
                add_stream_event(
                    state, "node_completed",
                    content={"needs_follow_up": True}, node="follow_up_detector",
                )
                logger.info("Follow-up preserved: ambiguous_artifact from router")
                return state

            # Check for missing required information
            needs_follow_up = False
            follow_up_type = None
            missing_info = []

            if intent_type in ["artifact", "multi_platform"]:
                # Check for platform
                if not entities.get("platform") and not entities.get("platforms"):
                    queries = state.get("decomposed_queries", [])
                    if not queries or not any(q.get("platform") for q in queries):
                        needs_follow_up = True
                        follow_up_type = "missing_platform"
                        missing_info.append("platform")

                # Check for topic
                if not entities.get("topic"):
                    queries = state.get("decomposed_queries", [])
                    if not queries or not any(q.get("topic") for q in queries):
                        # Only require topic if message is very short
                        if len(state["current_input"]) < 20:
                            needs_follow_up = True
                            follow_up_type = follow_up_type or "missing_topic"
                            missing_info.append("topic")

            elif intent_type == "modification":
                # Check if we have a previous artifact to modify
                if not state.get("artifacts"):
                    needs_follow_up = True
                    follow_up_type = "missing_reference"
                    missing_info.append("artifact_reference")

            state["needs_follow_up"] = needs_follow_up
            state["follow_up_type"] = follow_up_type
            state["follow_up_context"] = {
                "missing_info": missing_info,
                "original_query": state["current_input"],
            }

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "follow_up_detector",
                "completed",
                execution_time,
                metadata={"needs_follow_up": needs_follow_up, "type": follow_up_type},
            )
            add_stream_event(
                state,
                "node_completed",
                content={"needs_follow_up": needs_follow_up},
                node="follow_up_detector",
            )

            logger.info(
                "Follow-up detection complete",
                needs_follow_up=needs_follow_up,
                follow_up_type=follow_up_type,
            )

        except Exception as e:
            logger.error("Follow-up detection failed", error=str(e))
            state["needs_follow_up"] = False
            state["errors"].append(f"Follow-up detection error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "follow_up_detector", "failed", execution_time, str(e))

        return state


class FollowUpGeneratorNode:
    """
    Generates natural follow-up questions when needed.
    """

    AMBIGUOUS_ARTIFACT_PROMPT = """The user wants to modify a previous post, but it's unclear which one.
They said: "{original_query}"

Available posts:
{artifact_list}

Generate a friendly, concise question asking which post they want to modify.
List the options with numbers. Keep it to 2-3 sentences.

Respond in JSON:
{{
    "question": "the follow-up question",
    "options": ["option1", "option2"]
}}"""

    FOLLOW_UP_PROMPT = """Generate a friendly, concise follow-up question to get the missing information.

User's original request: "{original_query}"
Missing information: {missing_info}
Detected platforms: {platforms}

Generate a natural question that:
1. Is friendly and conversational
2. Clearly asks for the specific missing information
3. Provides helpful examples if relevant
4. If platforms include LinkedIn, Instagram, Facebook, or Twitter, ALSO ask for their social media handle
5. Is concise (2-3 sentences max)

Respond in JSON:
{{
    "question": "the follow-up question",
    "options": ["option1", "option2", "option3"]  // if applicable, otherwise empty
}}
"""

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Generate follow-up questions.

        Args:
            state: Current conversation state

        Returns:
            Updated state with follow-up questions
        """
        start_time = time.time()
        state["current_node"] = "follow_up_generator"

        add_stream_event(state, "node_started", node="follow_up_generator")

        try:
            if not state.get("needs_follow_up"):
                # No follow-up needed, pass through
                return state

            follow_up_type = state.get("follow_up_type")
            context = state.get("follow_up_context", {})

            # Get platforms from decomposed queries
            queries = state.get("decomposed_queries", [])
            platforms = [q.get("platform", "") for q in queries if q.get("platform")]

            # Extract artifact options for ambiguous_artifact follow-ups
            artifact_options = context.get("artifact_options", [])

            # Generate follow-up question
            question_data = await self._generate_question(
                context.get("original_query", ""),
                context.get("missing_info", []),
                follow_up_type,
                platforms,
                artifact_options=artifact_options,
            )

            state["follow_up_questions"] = [question_data.get("question", "")]

            # Set final response to the follow-up question
            state["final_response"] = question_data.get("question", "")
            state["suggestions"] = question_data.get("options", [])

            # Mark that we're waiting for user input
            state["requires_approval"] = True

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "follow_up_generator",
                "completed",
                execution_time,
            )
            add_stream_event(
                state,
                "node_completed",
                content={"question": state["follow_up_questions"][0] if state["follow_up_questions"] else None},
                node="follow_up_generator",
            )

            logger.info("Follow-up question generated")

        except Exception as e:
            logger.error("Follow-up generation failed", error=str(e))
            # Generate default question
            state["follow_up_questions"] = [self._get_default_question(state.get("follow_up_type"))]
            state["final_response"] = state["follow_up_questions"][0]
            state["errors"].append(f"Follow-up generation error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "follow_up_generator", "failed", execution_time, str(e))

        return state

    async def _generate_question(
        self,
        original_query: str,
        missing_info: list,
        follow_up_type: Optional[str],
        platforms: list = None,
        artifact_options: list = None,
    ) -> dict:
        """Generate follow-up question using LLM."""
        platforms = platforms or []

        if follow_up_type == "ambiguous_artifact" and artifact_options:
            artifact_list = "\n".join(
                f"{i + 1}. {opt}" for i, opt in enumerate(artifact_options)
            )
            prompt = self.AMBIGUOUS_ARTIFACT_PROMPT.format(
                original_query=original_query,
                artifact_list=artifact_list,
            )
        else:
            prompt = self.FOLLOW_UP_PROMPT.format(
                original_query=original_query,
                missing_info=", ".join(missing_info),
                platforms=", ".join(platforms) if platforms else "none specified",
            )

        messages = [
            LLMMessage(
                role="system",
                content="You are a helpful assistant asking clarifying questions. Respond with valid JSON.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await llm_client.generate_fast(messages, json_mode=True)
            return json.loads(response.content)

        except Exception:
            return {"question": self._get_default_question(follow_up_type), "options": []}

    def _get_default_question(self, follow_up_type: Optional[str]) -> str:
        """Get default follow-up question based on type."""
        defaults = {
            "missing_platform": (
                "Which platform would you like me to create content for? (LinkedIn, Instagram, Facebook, Twitter)\n"
                "Also, what's your social media handle? (e.g., @yourhandle)"
            ),
            "missing_topic": "What topic would you like the content to be about?",
            "missing_reference": "Which previous post would you like me to modify?",
            "ambiguous_artifact": "I've generated several posts recently. Which one would you like me to modify?",
            "clarification": "Could you provide more details about what you'd like?",
        }
        return defaults.get(follow_up_type, "Could you provide more details?")


def should_generate_follow_up(state: ConversationState) -> str:
    """
    Conditional edge function for follow-up routing.

    Args:
        state: Current conversation state

    Returns:
        Name of next node
    """
    if state.get("needs_follow_up"):
        return "follow_up_generator"
    return "data_checker"


# Create node instances
follow_up_detector_node = FollowUpDetectorNode()
follow_up_generator_node = FollowUpGeneratorNode()
