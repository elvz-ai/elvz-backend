"""
Master Conversational Graph.

The main LangGraph that orchestrates the entire conversation flow,
wrapping the existing content generation pipeline.
"""

import asyncio
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
_graph_lock = asyncio.Lock()


def create_conversational_graph(checkpointer=None) -> StateGraph:
    """
    Create the master conversational graph.

    Graph Structure:
    ```
    Entry -> guardrail_check
        |
        ├─[blocked]─> stream_aggregator -> END
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
                                                   stream_aggregator -> END

    NOTE: memory_saver runs as a fire-and-forget background task
    after invoke_conversation() returns, to avoid blocking the HTTP response.
    ```

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence

    Returns:
        Compiled StateGraph
    """
    # Create workflow
    workflow = StateGraph(ConversationState)

    # ==================== Add Nodes ====================
    # NOTE: memory_saver is NOT in the graph — it runs as a background task
    # after invoke_conversation() returns, to avoid blocking the HTTP response.
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

    # ==================== Set Entry Point ====================
    workflow.set_entry_point("guardrail_check")

    # ==================== Guardrail Routing ====================
    def guardrail_router(state: ConversationState) -> Literal["intent_classifier", "stream_aggregator"]:
        """Route based on guardrail check result."""
        if state.get("guardrail_passed", True):
            return "intent_classifier"
        # Blocked content — final_response already set, skip to aggregator → END
        return "stream_aggregator"

    workflow.add_conditional_edges(
        "guardrail_check",
        guardrail_router,
        {
            "intent_classifier": "intent_classifier",
            "stream_aggregator": "stream_aggregator",
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

    # ==================== Data Check -> Conditional ====================
    def data_checker_router(state: ConversationState) -> Literal[
        "multi_platform_orchestrator", "stream_aggregator"
    ]:
        """Route based on social data availability."""
        if state.get("social_not_connected", False):
            # Skip generation — final_response already set by data_checker
            return "stream_aggregator"
        return "multi_platform_orchestrator"

    workflow.add_conditional_edges(
        "data_checker",
        data_checker_router,
        {
            "multi_platform_orchestrator": "multi_platform_orchestrator",
            "stream_aggregator": "stream_aggregator",
        },
    )

    # ==================== Orchestrator -> Aggregator ====================
    workflow.add_edge("multi_platform_orchestrator", "stream_aggregator")

    # ==================== Final Edges ====================
    # Aggregator -> END (memory_saver runs as background task)
    workflow.add_edge("stream_aggregator", END)

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

    if _graph is not None:
        return _graph

    async with _graph_lock:
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
    from langchain_core.messages import HumanMessage, AIMessage
    from app.agents.conversational_graph.state import create_initial_state
    from app.services.execution_monitor import execution_logger, ExecutionStatus
    from app.services.memory_manager import memory_manager
    from app.core.config import settings

    # Dev user_id override
    if settings.skip_user_id and settings.dev_user_id:
        logger.info(
            "Dev override: using dev_user_id",
            original_user_id=user_id,
            dev_user_id=settings.dev_user_id,
        )
        user_id = settings.dev_user_id

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

    # Load previous messages from PostgreSQL and prepend to state
    # This gives the LLM full conversation history without relying on the checkpointer
    try:
        recent_messages = await memory_manager.get_recent_messages(
            conversation_id, limit=10
        )
        history_messages = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        if history_messages:
            # Prepend history before the current HumanMessage
            initial_state["messages"] = history_messages + initial_state["messages"]
            logger.info(
                "Conversation history loaded",
                conversation_id=conversation_id,
                history_count=len(history_messages),
            )
    except Exception as e:
        logger.warning("Failed to load conversation history", error=str(e))

    # Save user message to PostgreSQL (fire-and-forget — not needed before graph runs)
    asyncio.create_task(_background_save_user_message(
        conversation_id, user_input
    ))

    # Log initial state messages
    logger.info(
        "Initial state created - Memory tracking",
        conversation_id=conversation_id,
        thread_id=thread_id,
        initial_messages_count=len(initial_state.get("messages", [])),
        current_input_length=len(user_input),
        message_types=[type(m).__name__ for m in initial_state.get("messages", [])],
    )

    # Build config with thread_id for checkpointing
    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    if config:
        invoke_config.update(config)

    try:
        # Invoke graph (memory_saver is NOT in the graph — runs as background task)
        result = await graph.ainvoke(initial_state, invoke_config)

        # Fire memory_saver + execution logging as background tasks
        # so the HTTP response returns immediately.
        asyncio.create_task(_background_post_response(
            result=result,
            execution_id=execution_id,
            start_time=start_time,
            execution_logger=execution_logger,
            ExecutionStatus=ExecutionStatus,
        ))

        return result

    except Exception as e:
        # Log failed execution (still background)
        duration_ms = int((time.time() - start_time) * 1000)
        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            duration_ms=duration_ms,
            error_summary=str(e),
            failed_nodes=["unknown"],
        )
        raise


async def _background_save_user_message(conversation_id: str, user_input: str) -> None:
    """Fire-and-forget: save user message to PostgreSQL."""
    from app.services.memory_manager import memory_manager

    try:
        await memory_manager.save_message_to_memory(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
        )
    except Exception as e:
        logger.warning("Background: failed to save user message", error=str(e))


async def _background_post_response(
    result: dict,
    execution_id: str,
    start_time: float,
    execution_logger,
    ExecutionStatus,
) -> None:
    """Fire-and-forget: run memory_saver + log execution completion."""
    import time

    try:
        # 1. Run memory_saver on the final state
        await memory_saver_node(result)
    except Exception as e:
        logger.error("Background memory save failed", error=str(e))

    try:
        # 2. Log execution completion
        duration_ms = int((time.time() - start_time) * 1000)
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
    except Exception as e:
        logger.error("Background execution log failed", error=str(e))


# ---------------------------------------------------------------------------
# 4-stage mapping for SSE streaming
# ---------------------------------------------------------------------------
NODE_TO_STAGE: dict[str, tuple[int, str]] = {
    "guardrail_check":              (1, "Understanding your request..."),
    "intent_classifier":            (1, "Understanding your request..."),
    "query_decomposer":             (1, "Understanding your request..."),
    "memory_retriever":             (2, "Preparing context..."),
    "context_builder":              (2, "Preparing context..."),
    "router":                       (2, "Preparing context..."),
    "follow_up_detector":           (2, "Preparing context..."),
    "data_checker":                 (2, "Preparing context..."),
    "multi_platform_orchestrator":  (3, "Generating content..."),
    "follow_up_generator":          (3, "Generating content..."),
    "stream_aggregator":            (4, "Finalizing response..."),
}

TOTAL_STAGES = 4


async def stream_conversation_sse(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    event_bus: "EventBus",
    config: Optional[dict] = None,
) -> ConversationState:
    """
    Run the conversational graph with SSE streaming via event_bus.

    Combines graph.astream() (node-level events) with event_bus
    (token-level events pushed by nodes mid-execution).

    The event_bus is injected into LangGraph config so nodes can access it.
    """
    import time
    from uuid import uuid4
    from langchain_core.messages import HumanMessage, AIMessage
    from app.agents.conversational_graph.state import create_initial_state
    from app.services.execution_monitor import execution_logger, ExecutionStatus
    from app.services.memory_manager import memory_manager
    from app.core.config import settings

    # Dev user_id override
    if settings.skip_user_id and settings.dev_user_id:
        user_id = settings.dev_user_id

    execution_id = str(uuid4())
    start_time = time.time()

    execution_logger.log_execution_started(
        execution_id=execution_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_message=user_input,
    )

    graph = await get_conversational_graph()

    initial_state = create_initial_state(
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,
        user_input=user_input,
    )
    initial_state["execution_id"] = execution_id

    # Load conversation history (same as invoke_conversation)
    try:
        recent_messages = await memory_manager.get_recent_messages(
            conversation_id, limit=10
        )
        history_messages = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))
        if history_messages:
            initial_state["messages"] = history_messages + initial_state["messages"]
    except Exception as e:
        logger.warning("Failed to load conversation history", error=str(e))

    asyncio.create_task(_background_save_user_message(conversation_id, user_input))

    # Inject event_bus into LangGraph config so nodes can access it
    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
            "event_bus": event_bus,
        }
    }
    if config:
        invoke_config.update(config)

    final_state = None
    last_stage = 0

    try:
        async for event in graph.astream(initial_state, invoke_config):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                # Emit step event only when stage changes
                stage_info = NODE_TO_STAGE.get(node_name)
                if stage_info:
                    stage_num, stage_label = stage_info
                    if stage_num > last_stage:
                        event_bus.push_step(
                            node=node_name,
                            status="completed",
                            stage=stage_num,
                            label=stage_label,
                            progress=round(stage_num / TOTAL_STAGES, 2),
                        )
                        last_stage = stage_num

                # Forward artifacts immediately
                if isinstance(node_output, dict):
                    for artifact in node_output.get("artifacts", []):
                        event_bus.push("artifact", **artifact)

                    final_state = node_output

    except Exception as e:
        event_bus.push("error", message=str(e))
        logger.error("Streaming graph execution failed", error=str(e))
        duration_ms = int((time.time() - start_time) * 1000)
        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            duration_ms=duration_ms,
            error_summary=str(e),
            failed_nodes=["unknown"],
        )
        raise
    finally:
        event_bus.done()  # Always signal completion

    # Background: memory save + execution logging
    if final_state:
        asyncio.create_task(_background_post_response(
            result=final_state,
            execution_id=execution_id,
            start_time=start_time,
            execution_logger=execution_logger,
            ExecutionStatus=ExecutionStatus,
        ))

    return final_state


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
