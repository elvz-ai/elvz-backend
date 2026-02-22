"""Background execution logger with minimal latency impact."""

import asyncio
from datetime import datetime
from typing import Optional
import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.execution_log import ExecutionLog, ExecutionStatus
from app.core.database import AsyncSessionLocal

logger = structlog.get_logger(__name__)


class ExecutionLogger:
    """
    Background execution logger that writes to PostgreSQL asynchronously.

    Uses an async queue to avoid blocking the main request flow.
    Latency impact: ~1-2ms (just adding to queue).
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background log writer."""
        self._task = asyncio.create_task(self._process_queue())
        logger.info("ExecutionLogger started")

    async def stop(self):
        """Stop background writer gracefully."""
        if self._task:
            self._task.cancel()
            logger.info("ExecutionLogger stopped")

    async def _process_queue(self):
        """Process log events from queue in background."""
        while True:
            try:
                event = await self._queue.get()
                await self._write_event(event)
            except asyncio.CancelledError:
                logger.info("ExecutionLogger cancelled")
                break
            except Exception as e:
                logger.error("Failed to write log event", error=str(e), event_type=event.get("type"))

    async def _write_event(self, event: dict):
        """Write single event to database."""
        async with AsyncSessionLocal() as session:
            try:
                if event["type"] == "execution_started":
                    # Create new execution log
                    log = ExecutionLog(
                        id=event["execution_id"],
                        conversation_id=event["conversation_id"],
                        user_id=event["user_id"],
                        request_message=event["request_message"][:5000],  # Truncate if too long
                        status=ExecutionStatus.RUNNING,
                        started_at=event["timestamp"],
                    )
                    session.add(log)

                elif event["type"] == "execution_completed":
                    # Build node outputs summary with elapsed time
                    node_outputs = self._build_node_outputs(
                        event.get("execution_trace", [])
                    )

                    # Update execution log with final results
                    await session.execute(
                        update(ExecutionLog)
                        .where(ExecutionLog.id == event["execution_id"])
                        .values(
                            status=event["status"],
                            completed_at=event["timestamp"],
                            total_duration_ms=event["duration_ms"],
                            response_message=event.get("response_message", "")[:5000],
                            execution_trace=event.get("execution_trace"),
                            node_outputs=node_outputs,
                            error_summary=event.get("error_summary"),
                            failed_nodes=event.get("failed_nodes", []),
                        )
                    )

                await session.commit()
                logger.debug("Log event written", event_type=event["type"], execution_id=event.get("execution_id"))

            except Exception as e:
                logger.error("Database write failed", error=str(e), event_type=event["type"])
                await session.rollback()

    # Public API - non-blocking (just adds to queue)
    def log_execution_started(
        self,
        execution_id: str,
        conversation_id: str,
        user_id: str,
        request_message: str,
    ):
        """Log the start of an execution. Non-blocking."""
        self._queue.put_nowait({
            "type": "execution_started",
            "execution_id": execution_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "request_message": request_message,
            "timestamp": datetime.utcnow(),
        })

    def log_execution_completed(
        self,
        execution_id: str,
        status: ExecutionStatus,
        duration_ms: int,
        response_message: str = None,
        execution_trace: list = None,
        error_summary: str = None,
        failed_nodes: list = None,
        start_time: float = None,
    ):
        """Log the completion of an execution. Non-blocking."""
        self._queue.put_nowait({
            "type": "execution_completed",
            "execution_id": execution_id,
            "status": status,
            "duration_ms": duration_ms,
            "response_message": response_message,
            "execution_trace": execution_trace,
            "error_summary": error_summary,
            "failed_nodes": failed_nodes or [],
            "start_time": start_time,
            "timestamp": datetime.utcnow(),
        })

    def _build_node_outputs(self, execution_trace: list) -> dict:
        """
        Build a summary of node outputs with elapsed time.

        Returns dict like:
        {
            "guardrail_check": {
                "elapsed_ms": 120,
                "duration_ms": 120,
                "status": "completed",
                "output": {"guardrail_passed": True}
            },
            "intent_classifier": {
                "elapsed_ms": 570,
                "duration_ms": 450,
                "status": "completed",
                "output": {"intent_type": "artifact", "confidence": 0.9}
            }
        }
        """
        if not execution_trace:
            return {}

        node_outputs = {}
        cumulative_time_ms = 0  # Track cumulative elapsed time

        for trace in execution_trace:
            node_name = trace.get("node")
            if not node_name:
                continue

            # Get node duration
            node_duration_ms = trace.get("time_ms", 0)

            # Add to cumulative time
            cumulative_time_ms += node_duration_ms

            # Extract metadata if available
            metadata = trace.get("metadata", {})
            output_summary = {}

            # Extract ALL metadata fields
            if metadata:
                for key, value in metadata.items():
                    # Skip internal/large fields
                    if key not in ['__internal__', 'raw_response', 'full_context']:
                        output_summary[key] = value

            # Extract errors from trace itself
            if trace.get("error"):
                output_summary["error"] = trace["error"][:200]  # Truncate errors

            node_outputs[node_name] = {
                "elapsed_ms": cumulative_time_ms,  # Cumulative time from start
                "duration_ms": node_duration_ms,    # Individual node duration
                "status": trace.get("status", "unknown"),
                "output": output_summary if output_summary else None,
            }

        return node_outputs


# Global singleton
execution_logger = ExecutionLogger()
