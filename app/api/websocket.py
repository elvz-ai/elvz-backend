"""
WebSocket endpoints for real-time agent execution updates.
"""

import asyncio
import json
from typing import Optional

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from app.agents.platform_orchestrator import orchestrator
from app.core.config import settings

logger = structlog.get_logger(__name__)


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
            "message": str(e),
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
            "message": str(e),
        })

