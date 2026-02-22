"""Execution monitoring models for storing graph execution logs."""

from sqlalchemy import Column, String, Integer, DateTime, Text, Enum, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime
import enum


class ExecutionStatus(str, enum.Enum):
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

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)

    # Request/Response
    request_message = Column(Text, nullable=False)  # User input
    response_message = Column(Text)  # Final response to user

    # Status
    status = Column(Enum(ExecutionStatus), default=ExecutionStatus.RUNNING, index=True)

    # Timing
    started_at = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_duration_ms = Column(Integer)

    # Execution details - stores all node executions as JSON
    execution_trace = Column(JSON)  # Full trace from state["execution_trace"]

    # Node outputs - stores key outputs from each node for debugging
    node_outputs = Column(JSON)  # {node_name: {output_summary, elapsed_ms}}

    # Error tracking
    error_summary = Column(Text)  # Summary of errors if failed
    failed_nodes = Column(JSON)  # List of node names that failed

    # Indexes for common queries
    __table_args__ = (
        Index('idx_execution_status_date', 'status', 'started_at'),
        Index('idx_execution_user_date', 'user_id', 'started_at'),
    )
