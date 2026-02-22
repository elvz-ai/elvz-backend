"""
Router Node.

Routes the conversation to the appropriate handler based on intent.
"""

import json
import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.config import settings
from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM-based artifact resolution
# ---------------------------------------------------------------------------

ARTIFACT_RESOLVE_PROMPT = """You are resolving which post the user wants to modify.

User said: "{user_input}"

Available posts (numbered):
{artifact_list}

Which post does the user want to modify? Consider:
- Platform mentions (e.g. "the linkedin one")
- Topic references (e.g. "the one about AI")
- Ordinal references (e.g. "the second one", "the first post")
- Recency references (e.g. "the latest", "the last one")
- Tone/style references (e.g. "the professional one", "the casual post")
- If no clear reference, pick the most likely match

Respond in JSON:
{{
    "index": <0-based index of the matching post>,
    "confidence": <0.0 to 1.0 — how sure you are>,
    "reasoning": "<brief explanation>"
}}"""


async def _resolve_artifact_llm(
    user_input: str,
    artifact_history: list[dict],
) -> tuple[dict | None, float, str]:
    """
    Use LLM to semantically resolve which artifact the user wants to modify.
    Returns (artifact_or_None, confidence, reasoning).

    Fast path: single artifact skips LLM entirely.
    """
    if not artifact_history:
        return None, 0.0, "No artifacts in history"

    if len(artifact_history) == 1:
        return artifact_history[0], 1.0, "Only one artifact in history"

    # Build numbered artifact list for LLM
    artifact_list_str = ""
    for i, entry in enumerate(artifact_history):
        platform = (entry.get("platform") or "unknown").title()
        summary = entry.get("topic_summary") or "untitled"
        created = entry.get("created_at", "")
        artifact_list_str += f"{i}. [{platform}] {summary} (created: {created})\n"

    prompt = ARTIFACT_RESOLVE_PROMPT.format(
        user_input=user_input,
        artifact_list=artifact_list_str.strip(),
    )

    try:
        messages = [
            LLMMessage(
                role="system",
                content="You resolve artifact references. Respond with valid JSON only.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = await llm_client.generate_fast(messages, json_mode=True)
        result = json.loads(response.content)

        idx = result.get("index", 0)
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")

        # Bounds check
        if 0 <= idx < len(artifact_history):
            return artifact_history[idx], confidence, reasoning

        # Invalid index — fallback to latest
        return artifact_history[-1], 0.3, f"LLM returned invalid index {idx}, falling back to latest"

    except Exception as e:
        logger.warning("LLM artifact resolution failed, falling back to latest", error=str(e))
        return artifact_history[-1], 0.4, f"LLM call failed: {e}"


class RouterNode:
    """
    Routes conversation to appropriate handler.

    Routing decisions:
    - artifact/multi_platform -> data_checker -> orchestrator
    - qa -> qa_handler (direct response)
    - clarification -> follow_up_processor
    - modification -> modification_handler (with LLM-based artifact resolution)
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Determine routing based on intent.

        Args:
            state: Current conversation state

        Returns:
            Updated state with routing decision
        """
        start_time = time.time()
        state["current_node"] = "router"

        add_stream_event(state, "node_started", node="router")

        try:
            intent = state.get("current_intent") or {}
            intent_type = intent.get("type", "artifact")

            # ------------------------------------------------------------------
            # Check for pending_modification (user was asked "which artifact?")
            # Runs FIRST before any other routing.
            # ------------------------------------------------------------------
            pending = state.get("pending_modification")
            if pending and pending.get("pending"):
                artifact_history = state.get("artifact_history") or []
                resolved, confidence, reasoning = await _resolve_artifact_llm(
                    state.get("current_input", ""), artifact_history
                )

                if resolved and confidence >= settings.modification_confidence_threshold:
                    # User answered — resolve and proceed with original feedback
                    state["target_artifact"] = resolved
                    state["last_artifact"] = resolved
                    state["modification_feedback"] = pending.get("original_feedback", "")
                    state["pending_modification"] = None
                    route = "modification"
                    state["working_memory"]["route"] = route
                    state["working_memory"]["needs_data_check"] = False
                    if state.get("current_intent") is not None:
                        state["current_intent"]["type"] = "artifact"
                    logger.info(
                        "Pending modification resolved via LLM",
                        artifact_id=resolved.get("id"),
                        confidence=confidence,
                        reasoning=reasoning,
                    )

                    execution_time = int((time.time() - start_time) * 1000)
                    add_execution_trace(
                        state, "router", "completed", execution_time,
                        metadata={"route": route, "resolved_pending": True,
                                  "confidence": confidence, "reasoning": reasoning},
                    )
                    add_stream_event(
                        state, "node_completed",
                        content={"route": route}, node="router",
                    )
                    return state
                else:
                    # Check if user ignored follow-up and sent a new request
                    _modify_kw = {
                        "make", "change", "update", "rewrite", "revise", "edit",
                        "shorter", "longer", "casual", "formal", "professional",
                        "more", "less", "tone", "add", "remove",
                    }
                    if not (_modify_kw & set((state.get("current_input") or "").lower().split())):
                        state["pending_modification"] = None  # Clear, fall through

            # ------------------------------------------------------------------
            # Universal modification pre-check.
            # Fires BEFORE intent branches so it catches misclassified intents
            # (LLM may return "qa" or "clarification" for "make it more casual").
            # Only skipped for explicit artifact creation intents.
            # ------------------------------------------------------------------
            _modify_keywords = {
                "make", "change", "update", "rewrite", "revise", "edit",
                "shorter", "longer", "casual", "formal", "professional",
                "more", "less", "tone", "add", "remove",
            }
            _words = set((state.get("current_input") or "").lower().split())
            has_modify_keywords = bool(_modify_keywords & _words)

            entities = intent.get("entities") or {}
            is_modify_action = (entities.get("action") or "").lower() in ("modify", "edit", "change")

            # artifacts resets each turn; last_artifact is restored from working memory
            has_prior_artifacts = bool(state.get("artifacts")) or bool(state.get("last_artifact"))

            is_modification_override = (
                intent_type not in ("artifact", "multi_platform")
                and has_prior_artifacts
                and (is_modify_action or has_modify_keywords)
            )

            # Determine route
            if is_modification_override or intent_type == "modification":
                # --- LLM-based artifact resolution ---
                artifact_history = state.get("artifact_history") or []
                resolved, confidence, reasoning = await _resolve_artifact_llm(
                    state.get("current_input", ""), artifact_history
                )

                if confidence >= settings.modification_confidence_threshold and resolved:
                    # High confidence — proceed directly
                    state["target_artifact"] = resolved
                    state["last_artifact"] = resolved
                    route = "modification"
                    state["modification_feedback"] = state.get("current_input", "")
                    state["working_memory"]["route"] = route
                    state["working_memory"]["needs_data_check"] = False
                    if state.get("current_intent") is not None:
                        state["current_intent"]["type"] = "artifact"
                    logger.info(
                        "Artifact resolved via LLM",
                        confidence=confidence,
                        reasoning=reasoning,
                    )

                elif len(artifact_history) > 1:
                    # Low confidence + multiple artifacts — ask follow-up
                    state["pending_modification"] = {
                        "pending": True,
                        "original_feedback": state.get("current_input", ""),
                    }
                    state["needs_follow_up"] = True
                    state["follow_up_type"] = "ambiguous_artifact"
                    state["follow_up_context"] = {
                        "missing_info": ["target_artifact"],
                        "original_query": state.get("current_input", ""),
                        "artifact_options": [
                            f"{a.get('platform', '').title()}: {a.get('topic_summary', 'untitled')}"
                            for a in artifact_history
                        ],
                    }
                    route = "modification"
                    state["working_memory"]["route"] = route
                    state["working_memory"]["needs_data_check"] = False
                    logger.info(
                        "LLM uncertain — asking follow-up",
                        artifact_count=len(artifact_history),
                        confidence=confidence,
                        reasoning=reasoning,
                    )

                else:
                    # Single or no artifact — original behavior (use last_artifact)
                    route = "modification"
                    state["modification_feedback"] = state.get("current_input", "")
                    state["working_memory"]["route"] = route
                    state["working_memory"]["needs_data_check"] = False
                    if state.get("current_intent") is not None:
                        state["current_intent"]["type"] = "artifact"

                if is_modification_override and intent_type != "modification":
                    logger.info(
                        "Intent overridden to modification",
                        original_intent=intent_type,
                        has_modify_keywords=has_modify_keywords,
                        is_modify_action=is_modify_action,
                    )

            elif intent_type in ["artifact", "multi_platform"]:
                route = "artifact_generation"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = True

            elif intent_type == "qa":
                route = "qa_response"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = False

            elif intent_type == "clarification":
                route = "qa_response"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = False

            else:
                route = "artifact_generation"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = True

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "router",
                "completed",
                execution_time,
                metadata={"route": route},
            )
            add_stream_event(
                state,
                "node_completed",
                content={"route": route},
                node="router",
            )

            logger.info("Route determined", route=route, intent_type=intent_type)

        except Exception as e:
            logger.error("Routing failed", error=str(e))
            # Default to artifact generation on error
            state["working_memory"]["route"] = "artifact_generation"
            state["working_memory"]["needs_data_check"] = True
            state["errors"].append(f"Routing error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "router", "failed", execution_time, str(e))

        return state


def get_route(state: ConversationState) -> str:
    """
    Conditional edge function for router.

    Used by LangGraph to determine next node.

    Args:
        state: Current conversation state

    Returns:
        Name of next node
    """
    route = (state.get("working_memory") or {}).get("route", "artifact_generation")
    needs_data_check = (state.get("working_memory") or {}).get("needs_data_check", False)

    if route == "qa_response":
        # For Q&A, go directly to aggregator (which will generate response)
        return "stream_aggregator"

    elif route == "artifact_generation":
        if needs_data_check:
            return "data_checker"
        return "multi_platform_orchestrator"

    elif route == "process_clarification":
        return "follow_up_detector"

    elif route == "modification":
        # For modifications, go to orchestrator to regenerate
        return "multi_platform_orchestrator"

    # Default
    return "data_checker"


# Create node instance
router_node = RouterNode()
