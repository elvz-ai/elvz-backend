"""Execution monitoring models for storing graph execution logs."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ExecutionStatus(str, Enum):
    """Status of overall execution."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some nodes failed but execution completed


class ExecutionLog(Base):
    """
    Main execution log table - stores one row per conversation turn.
    Frontend will query this table to display logs and generate trees.
    """
    __tablename__ = "execution_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="SET NULL"), index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("user.id", ondelete="SET NULL"), index=True
    )

    # Request/Response
    request_message: Mapped[str] = mapped_column(Text, nullable=False)
    response_message: Mapped[Optional[str]] = mapped_column(Text)

    # Status — plain string, not PG enum (Next.js uses text())
    status: Mapped[Optional[str]] = mapped_column(
        String(50), default=ExecutionStatus.RUNNING.value
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    total_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Execution details - stores all node executions as JSON
    execution_trace: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Node outputs - stores key outputs from each node for debugging
    node_outputs: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Error tracking
    error_summary: Mapped[Optional[str]] = mapped_column(Text)
    failed_nodes: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_execution_status_date', 'status', 'started_at'),
        Index('idx_execution_user_date', 'user_id', 'started_at'),
    )
