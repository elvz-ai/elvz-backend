"""
WebSocket endpoints for real-time agent execution updates.

Supports both legacy platform orchestrator and new conversational graph.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from app.agents.platform_orchestrator import orchestrator
from app.core.config import settings

logger = structlog.get_logger(__name__)


# Event types for the conversational graph
class EventTypes:
    """WebSocket event types."""
    # Connection events
    PONG = "pong"
    ERROR = "error"

    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"

    # Node progress (conversational graph)
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"

    # Platform generation
    PLATFORM_STARTED = "platform_started"
    PLATFORM_PROGRESS = "platform_progress"
    PLATFORM_COMPLETED = "platform_completed"

    # Content streaming
    TEXT_CHUNK = "text_chunk"
    ARTIFACT_READY = "artifact_ready"

    # HITL
    HITL_REQUEST = "hitl_request"
    HITL_RESPONSE = "hitl_response"

    # Final result
    RESULT = "result"

    # Legacy events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info("WebSocket connected", client_id=client_id)
    
    def disconnect(self, client_id: str) -> None:
        """Remove a connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info("WebSocket disconnected", client_id=client_id)
    
    async def send_message(self, client_id: str, message: dict) -> None:
        """Send message to specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)
    
    async def broadcast(self, message: dict) -> None:
        """Send message to all connected clients."""
        for websocket in self.active_connections.values():
            try:
                await websocket.send_json(message)
            except Exception:
                pass


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Events:
    - agent_started: An agent has started execution
    - agent_completed: An agent has finished
    - agent_failed: An agent encountered an error
    - task_progress: Progress update during execution
    - result: Final result
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "ping":
                await manager.send_message(client_id, {"type": "pong"})
            
            elif message_type == "chat":
                # Handle chat request with streaming updates
                await handle_streaming_chat(client_id, data)
            
            elif message_type == "execute":
                # Handle direct Elf execution
                await handle_streaming_execute(client_id, data)

            elif message_type == "conversational_chat":
                # Handle conversational chat with new LangGraph pipeline
                await handle_conversational_chat(client_id, data)

            elif message_type == "hitl_response":
                # Handle human-in-the-loop response
                await handle_hitl_response(client_id, data)

            else:
                await manager.send_message(
                    client_id,
                    {"type": "error", "message": f"Unknown message type: {message_type}"}
                )
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), client_id=client_id)
        manager.disconnect(client_id)


async def handle_streaming_chat(client_id: str, data: dict) -> None:
    """Handle chat request with streaming progress updates."""
    
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")
    session_id = data.get("session_id")
    
    # Send start event
    await manager.send_message(client_id, {
        "type": "task_started",
        "task": "chat",
        "message": "Processing your request...",
    })
    
    try:
        from app.agents.platform_orchestrator.orchestrator import ChatRequest
        
        # Execute chat
        result = await orchestrator.chat(
            ChatRequest(
                user_id=user_id,
                message=message,
                session_id=session_id,
            )
        )
        
        # Send result
        await manager.send_message(client_id, {
            "type": "result",
            "response": result.response,
            "session_id": result.session_id,
            "elf_used": result.elf_used,
            "execution_time_ms": result.execution_time_ms,
            "suggestions": result.suggestions,
        })
        
    except Exception as e:
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e) if settings.environment == "development" else "Internal server error",
        })


async def handle_streaming_execute(client_id: str, data: dict) -> None:
    """Handle direct Elf execution with streaming updates."""

    elf_type = data.get("elf_type")
    user_id = data.get("user_id", "anonymous")
    request_data = data.get("request", {})

    if not elf_type:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "elf_type is required",
        })
        return

    # Send start event
    await manager.send_message(client_id, {
        "type": "task_started",
        "elf_type": elf_type,
        "message": f"Starting {elf_type} execution...",
    })

    try:
        result = await orchestrator.execute_elf(
            elf_type=elf_type,
            request_data=request_data,
            user_id=user_id,
        )

        # Send progress updates if available
        trace = result.get("result", {}).get("execution_trace", [])
        for step in trace:
            await manager.send_message(client_id, {
                "type": "agent_completed",
                "agent": step.get("agent"),
                "status": step.get("status"),
            })
            await asyncio.sleep(0.1)  # Small delay for visibility

        # Send final result
        await manager.send_message(client_id, {
            "type": "result",
            **result,
        })

    except Exception as e:
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e) if settings.environment == "development" else "Internal server error",
        })


async def handle_conversational_chat(client_id: str, data: dict) -> None:
    """
    Handle chat request using the conversational graph with full streaming.

    Uses EventBus for real-time step + token streaming via WebSocket.
    """
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")
    conversation_id = data.get("conversation_id")

    # Send start event
    await manager.send_message(client_id, {
        "type": EventTypes.TASK_STARTED,
        "task": "conversational_chat",
        "message": "Processing your request...",
    })

    try:
        from app.services.conversation_service import conversation_service
        from app.agents.conversational_graph.graph import stream_conversation_sse
        from app.agents.conversational_graph.event_bus import EventBus

        # Get or create conversation
        conversation = await conversation_service.get_or_create_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        # Save user message
        await conversation_service.add_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        event_bus = EventBus()

        # Run graph in background, pushing events to event_bus
        graph_task = asyncio.create_task(
            stream_conversation_sse(
                conversation_id=conversation.id,
                user_id=user_id,
                thread_id=conversation.thread_id,
                user_input=message,
                event_bus=event_bus,
            )
        )

        # Forward all events from bus to WebSocket client
        async for evt in event_bus:
            await manager.send_message(client_id, {
                "type": evt.event,
                **evt.data,
            })

        # Wait for graph to finish and get final state
        final_state = await graph_task

        # Send final result
        if final_state:
            await manager.send_message(client_id, {
                "type": EventTypes.RESULT,
                "response": final_state.get("final_response", ""),
                "conversation_id": conversation.id,
                "thread_id": conversation.thread_id,
                "artifacts": final_state.get("artifacts", []),
                "suggestions": final_state.get("suggestions", []),
                "requires_approval": final_state.get("requires_approval", False),
                "execution_trace": final_state.get("execution_trace", []),
            })

    except Exception as e:
        logger.error("Conversational chat error", error=str(e), client_id=client_id)
        await manager.send_message(client_id, {
            "type": EventTypes.ERROR,
            "message": str(e) if settings.environment == "development" else "Internal server error",
        })


async def handle_hitl_response(client_id: str, data: dict) -> None:
    """Handle HITL (human-in-the-loop) response from user."""

    conversation_id = data.get("conversation_id")
    hitl_request_id = data.get("hitl_request_id")
    response = data.get("response", "")
    selected_options = data.get("selected_options", [])

    if not conversation_id or not hitl_request_id:
        await manager.send_message(client_id, {
            "type": EventTypes.ERROR,
            "message": "conversation_id and hitl_request_id are required",
        })
        return

    try:
        # TODO: Implement HITL response handling
        # This would resume the graph from the checkpoint
        await manager.send_message(client_id, {
            "type": EventTypes.HITL_RESPONSE,
            "status": "received",
            "message": "Processing your response...",
        })

    except Exception as e:
        await manager.send_message(client_id, {
            "type": EventTypes.ERROR,
            "message": str(e) if settings.environment == "development" else "Internal server error",
        })

