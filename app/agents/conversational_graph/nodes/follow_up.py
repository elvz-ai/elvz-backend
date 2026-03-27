"""
Follow-up Handler Nodes.

Detects when follow-up questions are needed and generates them.

Uses a two-call HITL architecture:
- Call 1 (intent classifier, upstream): Classifies intent type
- Call 2 (this node): Intent-specific context evaluation
  - artifact/multi_platform: predefined critical fields + LLM dynamic fields
  - qa: LLM sufficiency score (no predefined fields)
  - clarification/modification/greeting: skip evaluation entirely
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


# ── Intent-specific evaluation prompts ──────────────────────────────────

ARTIFACT_EVAL_PROMPT = """Evaluate if this content creation request has enough context to generate useful content.

User message: "{user_message}"
Recent conversation context: {conversation_context}
Detected entities: {entities}

IMPORTANT: Check BOTH the current message AND the conversation context for information.
The user may have provided platform or topic in a previous message.

Required fields for content generation:
1. **platform** (CRITICAL) — which social media platform. Current value: {platform}
2. **topic** (CRITICAL) — what the content should be about. Current value: {topic}

Also identify any ADDITIONAL fields specific to THIS query that are critical.
For example:
- "competitor name" for comparison posts
- "event name" for event recaps
- "product features" for product posts
- "data points" for statistics-based posts

For each field, determine if it's present and whether it's CRITICAL or NICE-TO-HAVE.
- CRITICAL: Output would be meaningless without it.
- NICE-TO-HAVE: System can use sensible defaults (e.g., tone defaults to professional).

If critical fields are missing, generate a natural, friendly follow-up question that asks ONLY about the critical missing fields. Include 2-3 suggestion options.

Respond in JSON:
{{
    "required_context": [
        {{"field": "name", "present": true/false, "value": "value or null", "critical": true/false}}
    ],
    "completeness_score": 0.0-1.0,
    "critical_missing": ["field1"],
    "follow_up_question": "question or null if sufficient",
    "follow_up_suggestions": ["suggestion1", "suggestion2"]
}}"""


class FollowUpDetectorNode:
    """
    Detects when follow-up questions are needed using intent-specific
    LLM evaluation (Call 2 in the two-call HITL architecture).

    Evaluation strategy per intent type:
    - artifact/multi_platform: Predefined critical fields + LLM dynamic fields (Call 2)
    - qa/greeting/clarification/modification: Skip entirely (no evaluation, just proceed)

    Includes 5 loop-fix mechanisms:
    1. Partial answer → re-evaluate, ask for remaining critical fields
    2. Max attempts (>=2) → force proceed with defaults
    3. User changes subject → clear pending, handle fresh
    4. Malformed LLM eval → default to sufficient
    5. Exception → never block, proceed
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        start_time = time.time()
        state["current_node"] = "follow_up_detector"

        add_stream_event(state, "node_started", node="follow_up_detector")

        try:
            # ─── 1. Preserve ambiguous_artifact (set by router) ───
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

            intent = (state.get("current_intent") or {})
            intent_type = intent.get("type", "artifact")

            # ─── 2. Handle pending requirements (resumption) ───
            pending = (state.get("working_memory") or {}).get("pending_requirements")

            if pending:
                # Fix #3: User changed subject — abandon follow-up
                if intent_type != "clarification":
                    state["working_memory"]["pending_requirements"] = None
                    logger.info(
                        "Pending follow-up abandoned — user changed subject",
                        new_intent=intent_type,
                    )
                    # Fall through to evaluate the new request fresh

                # Fix #2: Max attempts — force proceed
                elif pending.get("attempt", 0) >= 2:
                    state["needs_follow_up"] = False
                    state["working_memory"]["pending_requirements"] = None
                    state["working_memory"]["route"] = pending.get(
                        "original_route", "artifact_generation"
                    )
                    self._trace_complete(state, start_time, False, "max_attempts_reached")
                    logger.info("Max follow-up attempts reached — proceeding with defaults")
                    return state

                # Clarification response — re-evaluate
                else:
                    original_intent = pending.get("original_intent", "artifact")
                    eval_result = await self._evaluate_for_intent(state, original_intent)

                    if self._is_sufficient(eval_result, original_intent):
                        state["needs_follow_up"] = False
                        state["working_memory"]["pending_requirements"] = None
                        state["working_memory"]["route"] = pending.get(
                            "original_route", "artifact_generation"
                        )
                        self._trace_complete(state, start_time, False, "requirements_satisfied")
                        logger.info("Pending requirements satisfied — proceeding")
                        return state
                    else:
                        # Fix #1: Partial answer — still missing, ask again
                        pending["attempt"] = pending.get("attempt", 0) + 1
                        self._set_hitl_response(state, eval_result, original_intent)
                        self._trace_complete(state, start_time, True, "partial_answer", extra={
                            "attempt": pending["attempt"],
                        })
                        logger.info("Partial answer — re-asking", attempt=pending["attempt"])
                        return state

            # ─── 3. Fresh evaluation — intent-specific ───

            # Also check route — router may have set modification even if intent says artifact
            route = (state.get("working_memory") or {}).get("route", "")

            # Skip evaluation entirely for these types
            if intent_type in ["clarification", "modification", "greeting", "qa"] or route == "modification":
                state["needs_follow_up"] = False
                self._trace_complete(state, start_time, False, f"skip_{intent_type}")
                logger.info("Evaluation skipped", intent_type=intent_type)
                return state

            # Run intent-specific evaluation (Call 2)
            eval_result = await self._evaluate_for_intent(state, intent_type)

            if not self._is_sufficient(eval_result, intent_type):
                self._set_hitl_response(state, eval_result, intent_type)
                state["working_memory"]["pending_requirements"] = {
                    "original_route": (state.get("working_memory") or {}).get(
                        "route", "artifact_generation"
                    ),
                    "original_intent": intent_type,
                    "attempt": 1,
                }
                self._trace_complete(state, start_time, True, "missing_requirements", extra={
                    "intent_type": intent_type,
                    "eval_result": eval_result,
                })
                logger.info("HITL triggered", intent_type=intent_type, eval_result=eval_result)
            else:
                state["needs_follow_up"] = False
                self._trace_complete(state, start_time, False, "sufficient")
                logger.info("Context sufficient — no follow-up needed")

        except Exception as e:
            # Fix #5: Never block on error — proceed with generation
            logger.error("Follow-up detection failed", error=str(e))
            state["needs_follow_up"] = False
            state["errors"].append(f"Follow-up detection error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "follow_up_detector", "failed", execution_time, str(e))

        return state

    # ── Intent-specific evaluation methods ──────────────────────────

    async def _evaluate_for_intent(self, state: ConversationState, intent_type: str) -> dict:
        """Route to the correct evaluator based on intent type. Only artifact/multi_platform get evaluated."""
        if intent_type in ["artifact", "multi_platform"]:
            return await self._evaluate_artifact(state)
        return {}  # No evaluation needed

    async def _evaluate_artifact(self, state: ConversationState) -> dict:
        """Evaluate artifact request: predefined critical fields + LLM dynamic fields."""
        intent = (state.get("current_intent") or {})
        entities = (intent.get("entities") or {})

        # Build conversation context from recent messages
        messages = state.get("messages", [])
        conversation_context = "No prior context"
        if messages:
            context_parts = []
            for msg in messages[-4:]:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", str(msg))[:200]
                context_parts.append(f"{role}: {content}")
            conversation_context = "\n".join(context_parts)

        prompt = ARTIFACT_EVAL_PROMPT.format(
            user_message=state.get("current_input", ""),
            conversation_context=conversation_context,
            entities=json.dumps(entities),
            platform=entities.get("platform") or "not specified",
            topic=entities.get("topic") or "not specified",
        )

        print("\n" + "=" * 80)
        print("HITL ARTIFACT EVALUATION — FULL PROMPT SENT TO LLM")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")

        try:
            eval_messages = [
                LLMMessage(
                    role="system",
                    content="You evaluate content creation requests for completeness. Respond with valid JSON only.",
                ),
                LLMMessage(role="user", content=prompt),
            ]
            response = await llm_client.generate_fast(eval_messages, json_mode=True)
            result = json.loads(response.content)

            print("\n" + "=" * 80)
            print("HITL ARTIFACT EVALUATION — LLM RESPONSE")
            print("=" * 80)
            print(json.dumps(result, indent=2))
            print("=" * 80 + "\n")

            return result
        except Exception as e:
            # Fix #4: Malformed eval → proceed (never block)
            logger.warning("Artifact evaluation failed — proceeding", error=str(e))
            return {"critical_missing": [], "completeness_score": 1.0}

    # ── Decision helpers ────────────────────────────────────────────

    def _is_sufficient(self, eval_result: dict, intent_type: str) -> bool:
        """Check if the evaluation result indicates sufficient context."""
        if intent_type in ["artifact", "multi_platform"]:
            return not eval_result.get("critical_missing", [])
        return True

    def _set_hitl_response(self, state: ConversationState, eval_result: dict, intent_type: str) -> None:
        """Set the HITL follow-up response on state from evaluation result."""
        state["needs_follow_up"] = True
        state["follow_up_type"] = "missing_requirements"

        follow_up_q = eval_result.get("follow_up_question") or "Could you provide more details?"
        state["follow_up_questions"] = [follow_up_q]
        suggestions = eval_result.get("follow_up_suggestions", [])
        state["suggestions"] = suggestions
        if suggestions:
            suggestion_list = "\n".join(f"- {s}" for s in suggestions)
            state["final_response"] = f"{follow_up_q}\n\nHere are some ideas to get you started:\n{suggestion_list}"
        else:
            state["final_response"] = follow_up_q
        state["requires_approval"] = True
        state["follow_up_context"] = {
            "original_query": state.get("current_input", ""),
            "intent_type": intent_type,
            "eval_result": eval_result,
        }

    # ── Tracing helper ──────────────────────────────────────────────

    def _trace_complete(
        self, state: ConversationState, start_time: float,
        needs_follow_up: bool, reason: str, extra: dict | None = None,
    ) -> None:
        """Log execution trace and stream event."""
        execution_time = int((time.time() - start_time) * 1000)
        metadata = {"needs_follow_up": needs_follow_up, "reason": reason}
        if extra:
            metadata.update(extra)
        add_execution_trace(
            state, "follow_up_detector", "completed", execution_time, metadata=metadata,
        )
        add_stream_event(
            state, "node_completed",
            content={"needs_follow_up": needs_follow_up}, node="follow_up_detector",
        )


class FollowUpGeneratorNode:
    """
    Generates natural follow-up questions when needed.

    For missing_requirements type, the follow_up_detector already generated
    the question via the evaluation LLM call — skip the extra call.
    Only uses LLM for ambiguous_artifact and legacy follow-up types.
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
4. Is concise (2-3 sentences max)

Respond in JSON:
{{
    "question": "the follow-up question",
    "options": ["option1", "option2", "option3"]
}}
"""

    async def __call__(self, state: ConversationState) -> ConversationState:
        """Generate follow-up questions."""
        start_time = time.time()
        state["current_node"] = "follow_up_generator"

        add_stream_event(state, "node_started", node="follow_up_generator")

        try:
            if not state.get("needs_follow_up"):
                return state

            # Detector already generated the question via eval LLM — pass through
            if state.get("final_response") and state.get("follow_up_type") == "missing_requirements":
                execution_time = int((time.time() - start_time) * 1000)
                add_execution_trace(
                    state, "follow_up_generator", "completed", execution_time,
                    metadata={"skipped": True, "reason": "question_from_evaluator"},
                )
                add_stream_event(
                    state, "node_completed",
                    content={"question": state["final_response"]},
                    node="follow_up_generator",
                )
                logger.info("Follow-up generator skipped — question from evaluator")
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
            state["final_response"] = question_data.get("question", "")
            state["suggestions"] = question_data.get("options", [])
            state["requires_approval"] = True

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "follow_up_generator", "completed", execution_time)
            add_stream_event(
                state, "node_completed",
                content={"question": state["follow_up_questions"][0] if state["follow_up_questions"] else None},
                node="follow_up_generator",
            )
            logger.info("Follow-up question generated")

        except Exception as e:
            logger.error("Follow-up generation failed", error=str(e))
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
                missing_info=", ".join(missing_info) if missing_info else "unspecified",
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
                "Which platform would you like me to create content for? "
                "(LinkedIn, Instagram, Facebook, Twitter)"
            ),
            "missing_topic": "What topic would you like the content to be about?",
            "missing_reference": "Which previous post would you like me to modify?",
            "ambiguous_artifact": "I've generated several posts recently. Which one would you like me to modify?",
            "missing_requirements": "Could you provide more details about what you'd like?",
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
