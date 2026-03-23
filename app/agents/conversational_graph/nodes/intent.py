"""
Intent Classifier Node.

Classifies user intent to route the conversation appropriately.
"""

import json
import time
import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    IntentClassification,
    add_execution_trace,
    add_stream_event,
)
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for an AI content generation assistant called Elvz.

Classify the user's message into one of these types:

- **artifact** — User wants to generate content (social post, blog, image, hashtags, schedule)
  Examples: "Create a LinkedIn post", "Write a post about AI", "Generate Instagram content"

- **qa** — User is asking a question or seeking information about a specific topic
  Examples: "What are best practices for LinkedIn?", "How do I increase engagement?", "Explain SEO strategies"

- **clarification** — User is ANSWERING a question that the assistant just asked in the previous message.
  The assistant asked for missing information, and the user is providing it.
  Examples:
    - Assistant asked "Which platform?" → User says "LinkedIn" → clarification
    - Assistant asked "What topic?" → User says "About AI trends" → clarification
    - Assistant asked "Which post to modify?" → User says "The first one" → clarification
  KEY RULE: If the MOST RECENT assistant message is a QUESTION, and the user's current message is a SHORT ANSWER to that question, this is **clarification**.

- **multi_platform** — User wants content for multiple platforms at once
  Examples: "Generate posts for LinkedIn and Facebook", "Create content for all my platforms"

- **modification** — User wants to CHANGE, UPDATE, or MODIFY content that was ALREADY GENERATED.
  The content must already exist. The user is requesting a change to it.
  Examples:
    - "Make it shorter" — changing existing text
    - "Change the tone to casual" — changing existing text style
    - "Add more hashtags" — updating existing hashtags
    - "Add a watermark to the image" — updating existing image
    - "Change the schedule to 3pm" — updating existing schedule
    - "Can you update the post you just created" — referencing previous artifact
  KEY RULE: If previously generated content EXISTS in the conversation, and the user is asking to ALTER it, this is **modification** — NOT artifact, NOT clarification.

- **greeting** — User is sending a simple greeting or pleasantry with no actionable request
  Examples: "hey", "hi", "hello", "good morning", "thanks", "thank you", "bye"
  NOT greeting: "what can you do", "what are your capabilities" — these are **qa**

Extract entities:
- platform: linkedin, instagram, facebook, twitter, tiktok (or null)
- platforms: list (for multi_platform)
- topic: what the content should be about (or null)
- tone: professional, casual, humorous, inspirational (or null)
- action: create, modify, analyze, schedule (or null)

Determine relevant search modalities:
- "text" — always include
- "image" — when user mentions images, visuals, photos, graphics
- "video" — when user mentions video, reels, stories, clips

---

## CURRENT QUERY (classify THIS message):
"{user_message}"

## PREVIOUS CONVERSATION HISTORY (numbered, higher number = more recent):
{conversation_context}

---

INSTRUCTIONS:
- Classify the CURRENT QUERY based on what the user is asking RIGHT NOW.
- Use the PREVIOUS CONVERSATION HISTORY to understand context.

HOW TO DISTINGUISH clarification vs modification:
1. Look at the MOST RECENT assistant message in the history.
2. If that assistant message is a QUESTION (asking for platform, topic, which post, etc.)
   AND the current query is ANSWERING that question → **clarification**
3. If the assistant previously GENERATED content (a post, image, etc.)
   AND the current query asks to CHANGE that content → **modification**

Examples with history:
  - History: Assistant asked "Which platform?" → User says "for linkedin" → **clarification**
  - History: Assistant generated a post → User says "make it shorter" → **modification**
  - History: Assistant generated a post → User says "add watermark to the image" → **modification**
  - History: Assistant asked "What topic?" → User says "wildlife protection" → **clarification**
  - History: Any → User says "create a new post about AI" → **artifact** (new request)

Respond in JSON:
{{
    "type": "artifact|qa|clarification|multi_platform|modification|greeting",
    "confidence": 0.0-1.0,
    "entities": {{
        "platform": "platform or null",
        "platforms": ["list if multi_platform"],
        "topic": "topic or null",
        "tone": "tone or null",
        "action": "action or null"
    }},
    "reasoning": "brief explanation",
    "search_modalities": ["text"]
}}
"""


class IntentClassifierNode:
    """
    Classifies user intent for routing decisions.

    Uses LLM to understand user's goal and extract entities.
    Context evaluation for HITL is handled separately by FollowUpDetectorNode.
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

            # Classify intent via LLM
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
            add_execution_trace(
                state,
                "intent_classifier",
                "completed",
                execution_time,
                metadata={
                    "intent_type": intent.get("type"),
                    "confidence": intent.get("confidence"),
                    "entities": (intent.get("entities") or {}),
                    "reasoning": (intent.get("reasoning") or "")[:100],  # Truncate reasoning
                }
            )
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
            # Default to artifact intent on error — never block on failure
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
        """Build numbered conversation context from recent messages.

        Uses last 10 turns with full content (no truncation).
        Higher number = more recent message.
        """
        messages = state.get("messages", [])

        if not messages:
            return "No previous conversation history."

        recent = messages[-10:]
        context_parts = []
        for i, msg in enumerate(recent, start=1):
            role = "USER" if getattr(msg, "type", "unknown") == "human" else "ASSISTANT"
            content = getattr(msg, "content", str(msg))
            context_parts.append(f"[{i}] {role}: {content}")

        return "\n".join(context_parts) if context_parts else "No previous conversation history."

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

        print("\n" + "=" * 80)
        print("INTENT CLASSIFICATION — FULL PROMPT SENT TO LLM")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")

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
                search_modalities=result.get("search_modalities", ["text"]),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse intent JSON: {e}")
            # Fallback to simple keyword matching
            return self._fallback_classify(user_message)

    def _fallback_classify(self, user_message: str) -> IntentClassification:
        """Fallback classification using keyword matching."""
        message_lower = user_message.lower()

        # Detect modalities
        search_modalities = ["text"]  # Always include text
        if any(word in message_lower for word in ["image", "visual", "photo", "graphic", "design", "picture"]):
            search_modalities.append("image")
        if any(word in message_lower for word in ["audio", "podcast", "voice", "sound"]):
            search_modalities.append("audio")
        if any(word in message_lower for word in ["video", "reel", "story", "clip"]):
            search_modalities.append("video")

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
                search_modalities=search_modalities,
            )

        # Artifact generation
        artifact_keywords = ["create", "generate", "write", "make", "post", "content"]
        if any(kw in message_lower for kw in artifact_keywords):
            return IntentClassification(
                type="artifact",
                confidence=0.7,
                entities={},
                reasoning="Artifact keywords detected",
                search_modalities=search_modalities,
            )

        # Question
        if message_lower.startswith(("what", "how", "why", "when", "where", "can", "?")):
            return IntentClassification(
                type="qa",
                confidence=0.7,
                entities={},
                reasoning="Question format detected",
                search_modalities=search_modalities,
            )

        # Default to artifact
        return IntentClassification(
            type="artifact",
            confidence=0.5,
            entities={},
            reasoning="Default classification",
            search_modalities=search_modalities,
        )


# Create node instance
intent_classifier_node = IntentClassifierNode()
