"""
Router Node.

Routes the conversation to the appropriate handler based on intent.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)

logger = structlog.get_logger(__name__)


class RouterNode:
    """
    Routes conversation to appropriate handler.

    Routing decisions:
    - artifact/multi_platform -> data_checker -> orchestrator
    - qa -> qa_handler (direct response)
    - clarification -> follow_up_processor
    - modification -> modification_handler
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
            intent = state.get("current_intent", {})
            intent_type = intent.get("type", "artifact")

            # Determine route
            if intent_type in ["artifact", "multi_platform"]:
                route = "artifact_generation"
                # Check if we need to validate data availability
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = True

            elif intent_type == "qa":
                route = "qa_response"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = False

            elif intent_type == "clarification":
                route = "process_clarification"
                state["working_memory"]["route"] = route
                state["working_memory"]["needs_data_check"] = False

            elif intent_type == "modification":
                route = "modification"
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
    route = state.get("working_memory", {}).get("route", "artifact_generation")
    needs_data_check = state.get("working_memory", {}).get("needs_data_check", False)

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
