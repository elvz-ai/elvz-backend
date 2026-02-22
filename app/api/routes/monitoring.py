"""Monitoring API endpoints for viewing execution logs."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional

from app.core.database import get_db
from app.models.execution_log import ExecutionLog, ExecutionStatus

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/logs")
async def list_execution_logs(
    limit: int = Query(50, le=200, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    status: Optional[ExecutionStatus] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    hours: int = Query(24, le=168, description="Time range in hours"),
    session: AsyncSession = Depends(get_db),
):
    """
    List execution logs with filters.

    Frontend will query this endpoint to display logs table.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    # Build query
    query = select(ExecutionLog).where(ExecutionLog.started_at >= since)

    if status:
        query = query.where(ExecutionLog.status == status)
    if user_id:
        query = query.where(ExecutionLog.user_id == user_id)
    if conversation_id:
        query = query.where(ExecutionLog.conversation_id == conversation_id)

    # Order and paginate
    query = query.order_by(desc(ExecutionLog.started_at)).offset(offset).limit(limit)

    result = await session.execute(query)
    logs = result.scalars().all()

    return {
        "total": len(logs),
        "offset": offset,
        "limit": limit,
        "logs": [
            {
                "id": log.id,
                "conversation_id": log.conversation_id,
                "user_id": log.user_id,
                "request_message": log.request_message[:200] + "..." if len(log.request_message) > 200 else log.request_message,
                "response_message": log.response_message[:200] + "..." if log.response_message and len(log.response_message) > 200 else log.response_message,
                "status": log.status.value,
                "started_at": log.started_at.isoformat() if log.started_at else None,
                "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                "total_duration_ms": log.total_duration_ms,
                "failed_nodes": log.failed_nodes or [],
                "error_summary": log.error_summary,
            }
            for log in logs
        ],
    }


@router.get("/logs/{execution_id}")
async def get_execution_log(
    execution_id: str,
    session: AsyncSession = Depends(get_db),
):
    """
    Get detailed execution log by ID.

    Returns full execution trace for tree generation in frontend.
    """
    result = await session.execute(
        select(ExecutionLog).where(ExecutionLog.id == execution_id)
    )
    log = result.scalar_one_or_none()

    if not log:
        raise HTTPException(status_code=404, detail="Execution log not found")

    return {
        "id": log.id,
        "conversation_id": log.conversation_id,
        "user_id": log.user_id,
        "request_message": log.request_message,
        "response_message": log.response_message,
        "status": log.status.value,
        "started_at": log.started_at.isoformat() if log.started_at else None,
        "completed_at": log.completed_at.isoformat() if log.completed_at else None,
        "total_duration_ms": log.total_duration_ms,
        "execution_trace": log.execution_trace or [],
        "node_outputs": log.node_outputs or {},  # Node outputs with elapsed time
        "failed_nodes": log.failed_nodes or [],
        "error_summary": log.error_summary,
    }


@router.get("/conversations/{conversation_id}/logs")
async def get_conversation_logs(
    conversation_id: str,
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_db),
):
    """Get all execution logs for a specific conversation."""
    result = await session.execute(
        select(ExecutionLog)
        .where(ExecutionLog.conversation_id == conversation_id)
        .order_by(desc(ExecutionLog.started_at))
        .limit(limit)
    )
    logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "status": log.status.value,
            "started_at": log.started_at.isoformat() if log.started_at else None,
            "total_duration_ms": log.total_duration_ms,
            "failed_nodes": log.failed_nodes or [],
        }
        for log in logs
    ]


@router.get("/stats")
async def get_monitoring_stats(
    hours: int = Query(24, le=168, description="Time range in hours"),
    session: AsyncSession = Depends(get_db),
):
    """
    Get monitoring statistics.

    Returns counts by status, average duration, etc.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    result = await session.execute(
        select(ExecutionLog).where(ExecutionLog.started_at >= since)
    )
    logs = result.scalars().all()

    # Calculate stats
    total = len(logs)
    completed = sum(1 for log in logs if log.status == ExecutionStatus.COMPLETED)
    failed = sum(1 for log in logs if log.status == ExecutionStatus.FAILED)
    partial = sum(1 for log in logs if log.status == ExecutionStatus.PARTIAL)
    running = sum(1 for log in logs if log.status == ExecutionStatus.RUNNING)

    durations = [log.total_duration_ms for log in logs if log.total_duration_ms]
    avg_duration = sum(durations) // len(durations) if durations else 0

    # Failed nodes breakdown
    failed_nodes_count = {}
    for log in logs:
        if log.failed_nodes:
            for node in log.failed_nodes:
                failed_nodes_count[node] = failed_nodes_count.get(node, 0) + 1

    return {
        "time_range_hours": hours,
        "total_executions": total,
        "by_status": {
            "completed": completed,
            "failed": failed,
            "partial": partial,
            "running": running,
        },
        "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
        "avg_duration_ms": avg_duration,
        "failed_nodes_breakdown": failed_nodes_count,
    }
