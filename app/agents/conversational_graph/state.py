"""
Conversation State Schema for the Master Graph.

Defines the TypedDict that flows through all nodes in the
conversational graph, maintaining context across turns.
"""

from datetime import datetime
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class IntentClassification(TypedDict, total=False):
    """Intent classification result."""
    type: str  # qa, artifact, clarification, multi_platform
    confidence: float
    entities: dict  # {platform, topic, action, etc.}
    reasoning: str
    search_modalities: list[str]  # ["text", "image", "audio", "video"]


class DecomposedQuery(TypedDict, total=False):
    """Single decomposed query for multi-platform requests."""
    platform: str
    query: str
    topic: str
    priority: int
    status: str  # pending, in_progress, completed, failed


class GeneratedArtifact(TypedDict, total=False):
    """Generated content artifact."""
    id: str
    platform: str
    artifact_type: str
    content: dict
    status: str
    generation_time_ms: int
    tokens_used: int


class StreamEvent(TypedDict, total=False):
    """Event for streaming updates to client."""
    type: str  # node_started, node_completed, text_chunk, artifact_ready, etc.
    node: str
    content: Any
    timestamp: str
    metadata: dict


class ConversationState(TypedDict, total=False):
    """
    Master state for the conversational graph.

    This state flows through all nodes and maintains
    context across the conversation turn.
    """

    # ==================== Core Identifiers ====================
    conversation_id: str
    user_id: str
    thread_id: str  # LangGraph thread for checkpointing

    # ==================== Message History ====================
    # Using LangGraph's add_messages reducer for automatic message handling
    messages: Annotated[list[BaseMessage], add_messages]

    # Current turn input
    current_input: str
    current_message_id: Optional[str]

    # ==================== Intent & Classification ====================
    current_intent: Optional[IntentClassification]
    intent_history: list[str]  # Track intent types across turns

    # ==================== Query Decomposition ====================
    is_multi_platform: bool
    decomposed_queries: list[DecomposedQuery]
    active_query_index: int
    decomposition_complete: bool

    # ==================== Memory & Context ====================
    working_memory: dict  # Current conversation context
    retrieved_memory: list[dict]  # From vector store
    user_profile: Optional[dict]  # Brand voice, preferences
    rag_context: str  # Formatted context for LLM

    # ==================== Content Safety ====================
    guardrail_passed: bool
    guardrail_violations: list[str]
    guardrail_action: Optional[str]  # pass, block, warn

    # ==================== Follow-up / HITL ====================
    needs_follow_up: bool
    follow_up_type: Optional[str]  # missing_platform, missing_topic, clarification
    follow_up_questions: list[str]
    follow_up_context: dict
    hitl_request_id: Optional[str]
    requires_approval: bool

    # ==================== Data Availability ====================
    data_available: dict  # {platform: bool}
    missing_data_platforms: list[str]
    needs_scraping: bool

    # ==================== Artifact Generation ====================
    artifacts: list[GeneratedArtifact]
    artifact_batch_id: Optional[str]
    artifacts_by_platform: dict  # {platform: artifact}
    generation_progress: dict  # {platform: progress_pct}

    # ==================== Streaming & Events ====================
    stream_events: list[StreamEvent]
    current_node: str

    # ==================== Response ====================
    final_response: Optional[str]
    suggestions: list[str]

    # ==================== Execution Metadata ====================
    execution_trace: list[dict]  # [{node, status, time_ms, error}]
    errors: list[str]
    total_tokens_used: int
    total_cost: float
    execution_start_time: Optional[datetime]


def create_initial_state(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
) -> ConversationState:
    """
    Create initial state for a new conversation turn.

    Args:
        conversation_id: Conversation identifier
        user_id: User identifier
        thread_id: LangGraph thread identifier
        user_input: User's message

    Returns:
        Initial conversation state
    """
    from langchain_core.messages import HumanMessage

    return ConversationState(
        # Core identifiers
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,

        # Messages - Include current user message
        # LangGraph's add_messages reducer will merge with previous messages from checkpoint
        messages=[HumanMessage(content=user_input)],
        current_input=user_input,
        current_message_id=None,

        # Intent
        current_intent=None,
        intent_history=[],

        # Query decomposition
        is_multi_platform=False,
        decomposed_queries=[],
        active_query_index=0,
        decomposition_complete=False,

        # Memory
        working_memory={},
        retrieved_memory=[],
        user_profile=None,
        rag_context="",

        # Safety
        guardrail_passed=True,
        guardrail_violations=[],
        guardrail_action=None,

        # Follow-up
        needs_follow_up=False,
        follow_up_type=None,
        follow_up_questions=[],
        follow_up_context={},
        hitl_request_id=None,
        requires_approval=False,

        # Data availability
        data_available={},
        missing_data_platforms=[],
        needs_scraping=False,

        # Artifacts
        artifacts=[],
        artifact_batch_id=None,
        artifacts_by_platform={},
        generation_progress={},

        # Streaming
        stream_events=[],
        current_node="start",

        # Response
        final_response=None,
        suggestions=[],

        # Execution
        execution_trace=[],
        errors=[],
        total_tokens_used=0,
        total_cost=0.0,
        execution_start_time=datetime.utcnow(),
    )


def add_execution_trace(
    state: ConversationState,
    node: str,
    status: str,
    time_ms: int = 0,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Add an execution trace entry to the state.

    Args:
        state: Current conversation state
        node: Node name
        status: Execution status (started, completed, failed)
        time_ms: Execution time in milliseconds
        error: Error message if failed
        metadata: Additional metadata
    """
    trace_entry = {
        "node": node,
        "status": status,
        "time_ms": time_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if error:
        trace_entry["error"] = error

    if metadata:
        trace_entry["metadata"] = metadata

    state["execution_trace"].append(trace_entry)


def add_stream_event(
    state: ConversationState,
    event_type: str,
    content: Any = None,
    node: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Add a streaming event to the state.

    Args:
        state: Current conversation state
        event_type: Type of event
        content: Event content
        node: Node that generated the event
        metadata: Additional metadata
    """
    event = StreamEvent(
        type=event_type,
        node=node or state.get("current_node", "unknown"),
        content=content,
        timestamp=datetime.utcnow().isoformat(),
        metadata=metadata or {},
    )

    state["stream_events"].append(event)
