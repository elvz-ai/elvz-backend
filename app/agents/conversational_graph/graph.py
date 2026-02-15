"""
Master Conversational Graph.

The main LangGraph that orchestrates the entire conversation flow,
wrapping the existing content generation pipeline.
"""

from typing import Literal, Optional

import structlog
from langgraph.graph import END, StateGraph

from app.agents.conversational_graph.state import ConversationState
from app.agents.conversational_graph.nodes import (
    guardrail_node,
    intent_classifier_node,
    query_decomposer_node,
    memory_retriever_node,
    context_builder_node,
    router_node,
    follow_up_detector_node,
    follow_up_generator_node,
    data_checker_node,
    multi_platform_orchestrator_node,
    stream_aggregator_node,
    memory_saver_node,
)
from app.agents.conversational_graph.nodes.router import get_route
from app.agents.conversational_graph.nodes.follow_up import should_generate_follow_up

logger = structlog.get_logger(__name__)

# Global graph instance
_graph = None


def create_conversational_graph(checkpointer=None) -> StateGraph:
    """
    Create the master conversational graph.

    Graph Structure:
    ```
    Entry -> guardrail_check
        |
        ├─[blocked]─> memory_saver -> END
        |
        └─[passed]─> intent_classifier -> query_decomposer -> memory_retriever
                                                                    |
                                                                    v
                                                              context_builder
                                                                    |
                                                                    v
                                                                 router
                                                                    |
                    ┌───────────────────────────────────────────────┼───────────────┐
                    |                                               |               |
                    v                                               v               v
            follow_up_detector                               data_checker     stream_aggregator
                    |                                               |            (for Q&A)
                    ├─[needs_follow_up]─> follow_up_generator       |
                    |                           |                   v
                    └─[no_follow_up]────────────┼──> multi_platform_orchestrator
                                                |                   |
                                                └───────────────────┘
                                                          |
                                                          v
                                                   stream_aggregator
                                                          |
                                                          v
                                                     memory_saver -> END
    ```

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence

    Returns:
        Compiled StateGraph
    """
    # Create workflow
    workflow = StateGraph(ConversationState)

    # ==================== Add Nodes ====================
    workflow.add_node("guardrail_check", guardrail_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("query_decomposer", query_decomposer_node)
    workflow.add_node("memory_retriever", memory_retriever_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("router", router_node)
    workflow.add_node("follow_up_detector", follow_up_detector_node)
    workflow.add_node("follow_up_generator", follow_up_generator_node)
    workflow.add_node("data_checker", data_checker_node)
    workflow.add_node("multi_platform_orchestrator", multi_platform_orchestrator_node)
    workflow.add_node("stream_aggregator", stream_aggregator_node)
    workflow.add_node("memory_saver", memory_saver_node)

    # ==================== Set Entry Point ====================
    workflow.set_entry_point("guardrail_check")

    # ==================== Guardrail Routing ====================
    def guardrail_router(state: ConversationState) -> Literal["intent_classifier", "memory_saver"]:
        """Route based on guardrail check result."""
        if state.get("guardrail_passed", True):
            return "intent_classifier"
        # Blocked content goes directly to save and end
        return "memory_saver"

    workflow.add_conditional_edges(
        "guardrail_check",
        guardrail_router,
        {
            "intent_classifier": "intent_classifier",
            "memory_saver": "memory_saver",
        },
    )

    # ==================== Main Flow Edges ====================
    # Intent -> Decomposer -> Memory -> Context -> Router
    workflow.add_edge("intent_classifier", "query_decomposer")
    workflow.add_edge("query_decomposer", "memory_retriever")
    workflow.add_edge("memory_retriever", "context_builder")
    workflow.add_edge("context_builder", "router")

    # ==================== Router Conditional Edges ====================
    def router_decision(state: ConversationState) -> Literal[
        "follow_up_detector", "data_checker", "stream_aggregator"
    ]:
        """Route based on intent and context."""
        route = state.get("working_memory", {}).get("route", "artifact_generation")
        needs_data_check = state.get("working_memory", {}).get("needs_data_check", False)

        if route == "qa_response":
            # Q&A goes directly to aggregator
            return "stream_aggregator"

        if route == "process_clarification":
            return "follow_up_detector"

        if route in ["artifact_generation", "modification"]:
            if needs_data_check:
                return "data_checker"
            return "follow_up_detector"

        return "data_checker"

    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            "follow_up_detector": "follow_up_detector",
            "data_checker": "data_checker",
            "stream_aggregator": "stream_aggregator",
        },
    )

    # ==================== Follow-up Path ====================
    def follow_up_router(state: ConversationState) -> Literal[
        "follow_up_generator", "data_checker"
    ]:
        """Route based on whether follow-up is needed."""
        if state.get("needs_follow_up", False):
            return "follow_up_generator"
        return "data_checker"

    workflow.add_conditional_edges(
        "follow_up_detector",
        follow_up_router,
        {
            "follow_up_generator": "follow_up_generator",
            "data_checker": "data_checker",
        },
    )

    # Follow-up generator goes to aggregator (to return question to user)
    workflow.add_edge("follow_up_generator", "stream_aggregator")

    # ==================== Data Check -> Orchestrator ====================
    workflow.add_edge("data_checker", "multi_platform_orchestrator")

    # ==================== Orchestrator -> Aggregator ====================
    workflow.add_edge("multi_platform_orchestrator", "stream_aggregator")

    # ==================== Final Edges ====================
    # Aggregator -> Saver -> END
    workflow.add_edge("stream_aggregator", "memory_saver")
    workflow.add_edge("memory_saver", END)

    # ==================== Compile Graph ====================
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


async def get_conversational_graph(checkpointer=None):
    """
    Get or create the conversational graph.

    Uses a cached instance for performance.

    Args:
        checkpointer: Optional checkpointer (will use default if not provided)

    Returns:
        Compiled StateGraph
    """
    global _graph

    if _graph is None:
        # Get checkpointer if not provided
        if checkpointer is None:
            try:
                from app.core.checkpointer import get_checkpointer
                checkpointer = await get_checkpointer()
            except Exception as e:
                logger.warning(f"Could not initialize checkpointer: {e}")
                checkpointer = None

        _graph = create_conversational_graph(checkpointer)
        logger.info("Conversational graph initialized")

    return _graph


async def invoke_conversation(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    config: Optional[dict] = None,
) -> ConversationState:
    """
    Invoke the conversational graph with user input.

    Args:
        conversation_id: Conversation identifier
        user_id: User identifier
        thread_id: LangGraph thread identifier
        user_input: User's message
        config: Optional additional config

    Returns:
        Final conversation state
    """
    import time
    from uuid import uuid4
    from app.agents.conversational_graph.state import create_initial_state
    from app.services.execution_monitor import execution_logger, ExecutionStatus

    # Generate execution ID for monitoring
    execution_id = str(uuid4())
    start_time = time.time()

    # Log execution start (non-blocking)
    execution_logger.log_execution_started(
        execution_id=execution_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_message=user_input,
    )

    # Get graph
    graph = await get_conversational_graph()

    # Create initial state
    initial_state = create_initial_state(
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,
        user_input=user_input,
    )

    # Store execution_id in state for tracking
    initial_state["execution_id"] = execution_id

    # Build config with thread_id for checkpointing
    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    if config:
        invoke_config.update(config)

    try:
        # Invoke graph
        result = await graph.ainvoke(initial_state, invoke_config)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Determine status based on errors
        errors = result.get("errors", [])
        failed_nodes = [
            trace["node"]
            for trace in result.get("execution_trace", [])
            if trace.get("status") == "failed"
        ]

        if errors or failed_nodes:
            status = ExecutionStatus.PARTIAL if result.get("final_response") else ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.COMPLETED

        # Log execution completion (non-blocking)
        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=status,
            duration_ms=duration_ms,
            response_message=result.get("final_response", ""),
            execution_trace=result.get("execution_trace", []),
            error_summary="; ".join(errors) if errors else None,
            failed_nodes=failed_nodes,
            start_time=start_time,
        )

        return result

    except Exception as e:
        # Log failed execution
        duration_ms = int((time.time() - start_time) * 1000)
        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            duration_ms=duration_ms,
            error_summary=str(e),
            failed_nodes=["unknown"],
        )
        raise


async def stream_conversation(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    config: Optional[dict] = None,
):
    """
    Stream the conversational graph execution.

    Yields events as the graph executes.

    Args:
        conversation_id: Conversation identifier
        user_id: User identifier
        thread_id: LangGraph thread identifier
        user_input: User's message
        config: Optional additional config

    Yields:
        Streaming events from graph execution
    """
    from app.agents.conversational_graph.state import create_initial_state

    # Get graph
    graph = await get_conversational_graph()

    # Create initial state
    initial_state = create_initial_state(
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,
        user_input=user_input,
    )

    # Build config
    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    if config:
        invoke_config.update(config)

    # Stream graph execution
    async for event in graph.astream(initial_state, invoke_config):
        yield event
