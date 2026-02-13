"""
Intent Classifier Node.

Classifies user intent to route the conversation appropriately.
"""

import json
import time
from typing import Optional

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    IntentClassification,
    add_execution_trace,
    add_stream_event,
)
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a social media content generation assistant.

Analyze the user's message and classify the intent into one of these categories:

1. **artifact** - User wants to generate content (post, image, hashtags, schedule)
   - Examples: "Create a LinkedIn post", "Generate content about AI", "Write a post for Instagram"

2. **qa** - User is asking a question or seeking information
   - Examples: "What are best practices for LinkedIn?", "How do I increase engagement?"

3. **clarification** - User is responding to a follow-up question or providing more context
   - Examples: Responses after being asked for topic/platform

4. **multi_platform** - User wants content for multiple platforms at once
   - Examples: "Generate posts for LinkedIn and Facebook", "Create content for all my platforms"

5. **modification** - User wants to modify previously generated content
   - Examples: "Make it shorter", "Change the tone", "Add more hashtags"

Also extract any relevant entities:
- platform: linkedin, instagram, facebook, twitter, tiktok
- topic: what the content should be about
- tone: professional, casual, humorous, inspirational
- action: create, modify, analyze, schedule

User Message: {user_message}

Recent Conversation Context:
{conversation_context}

Respond in JSON format:
{{
    "type": "artifact|qa|clarification|multi_platform|modification",
    "confidence": 0.0-1.0,
    "entities": {{
        "platform": "platform or null",
        "platforms": ["list of platforms if multi_platform"],
        "topic": "topic or null",
        "tone": "tone or null",
        "action": "action or null"
    }},
    "reasoning": "brief explanation"
}}
"""


class IntentClassifierNode:
    """
    Classifies user intent for routing decisions.

    Uses LLM to understand user's goal and extract entities.
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Classify user intent.

        Args:
            state: Current conversation state

        Returns:
            Updated state with intent classification
        """
        start_time = time.time()
        state["current_node"] = "intent_classifier"

        add_stream_event(state, "node_started", node="intent_classifier")

        try:
            # Build conversation context from recent messages
            context = self._build_context(state)

            # Classify intent
            intent = await self._classify_intent(
                state["current_input"],
                context,
            )

            state["current_intent"] = intent
            state["intent_history"].append(intent.get("type", "unknown"))

            # Set multi-platform flag
            if intent.get("type") == "multi_platform":
                state["is_multi_platform"] = True

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "intent_classifier", "completed", execution_time)
            add_stream_event(
                state,
                "node_completed",
                content={"intent": intent.get("type"), "confidence": intent.get("confidence")},
                node="intent_classifier",
            )

            logger.info(
                "Intent classified",
                intent_type=intent.get("type"),
                confidence=intent.get("confidence"),
            )

        except Exception as e:
            logger.error("Intent classification failed", error=str(e))
            # Default to artifact intent on error
            state["current_intent"] = IntentClassification(
                type="artifact",
                confidence=0.5,
                entities={},
                reasoning="Default due to classification error",
            )
            state["errors"].append(f"Intent classification error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "intent_classifier", "failed", execution_time, str(e))

        return state

    def _build_context(self, state: ConversationState) -> str:
        """Build conversation context string from recent messages."""
        messages = state.get("messages", [])

        if not messages:
            return "No previous conversation context."

        context_parts = []
        for msg in messages[-5:]:  # Last 5 messages
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))[:200]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts) if context_parts else "No previous context."

    async def _classify_intent(
        self,
        user_message: str,
        conversation_context: str,
    ) -> IntentClassification:
        """
        Use LLM to classify intent.

        Args:
            user_message: User's current message
            conversation_context: Recent conversation history

        Returns:
            IntentClassification dict
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_message=user_message,
            conversation_context=conversation_context,
        )

        messages = [
            LLMMessage(
                role="system",
                content="You are an intent classifier. Respond only with valid JSON.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await llm_client.generate_fast(messages, json_mode=True)

            # Parse JSON response
            result = json.loads(response.content)

            return IntentClassification(
                type=result.get("type", "artifact"),
                confidence=float(result.get("confidence", 0.8)),
                entities=result.get("entities", {}),
                reasoning=result.get("reasoning", ""),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse intent JSON: {e}")
            # Fallback to simple keyword matching
            return self._fallback_classify(user_message)

    def _fallback_classify(self, user_message: str) -> IntentClassification:
        """Fallback classification using keyword matching."""
        message_lower = user_message.lower()

        # Multi-platform detection
        multi_indicators = [" and ", "all platforms", "each platform", "both"]
        platforms = ["linkedin", "instagram", "facebook", "twitter", "tiktok"]
        platform_count = sum(1 for p in platforms if p in message_lower)

        if platform_count >= 2 or any(ind in message_lower for ind in multi_indicators):
            return IntentClassification(
                type="multi_platform",
                confidence=0.7,
                entities={"platforms": [p for p in platforms if p in message_lower]},
                reasoning="Multiple platforms detected",
            )

        # Artifact generation
        artifact_keywords = ["create", "generate", "write", "make", "post", "content"]
        if any(kw in message_lower for kw in artifact_keywords):
            return IntentClassification(
                type="artifact",
                confidence=0.7,
                entities={},
                reasoning="Artifact keywords detected",
            )

        # Question
        if message_lower.startswith(("what", "how", "why", "when", "where", "can", "?")):
            return IntentClassification(
                type="qa",
                confidence=0.7,
                entities={},
                reasoning="Question format detected",
            )

        # Default to artifact
        return IntentClassification(
            type="artifact",
            confidence=0.5,
            entities={},
            reasoning="Default classification",
        )


# Create node instance
intent_classifier_node = IntentClassifierNode()
