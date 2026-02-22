"""
Conversation and Message database models.
Supports multi-turn conversations with LangGraph checkpointing.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.orm.exc import DetachedInstanceError

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.hitl import HITLRequest
    from app.models.user import User


class ConversationStatus(str, Enum):
    """Conversation status states."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """
    Conversation model for multi-turn chat sessions.
    Linked to LangGraph thread for state persistence.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # LangGraph integration
    thread_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)

    # Conversation metadata
    title: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(50), default=ConversationStatus.ACTIVE.value, nullable=False
    )

    # Context and settings
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    # {
    #     "platforms": ["linkedin", "instagram"],
    #     "topic": "AI trends",
    #     "intent_history": ["artifact", "qa", "clarification"],
    #     "total_tokens": 15000
    # }

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    user: Mapped["User"] = relationship("User", backref="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", back_populates="conversation", cascade="all, delete-orphan"
    )
    hitl_requests: Mapped[list["HITLRequest"]] = relationship(
        "HITLRequest", back_populates="conversation", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_conversations_user_status", "user_id", "status"),
        Index("idx_conversations_last_message", "last_message_at"),
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id}, status={self.status})>"

    @property
    def message_count(self) -> int:
        """Get total message count."""
        try:
            return len(self.messages) if self.messages else 0
        except DetachedInstanceError:
            return 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "title": self.title,
            "status": self.status,
            "message_count": self.message_count,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }


class Message(Base):
    """
    Message model for conversation history.
    Stores individual messages with metadata.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Message metadata
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    # {
    #     "intent": {"type": "artifact", "confidence": 0.95},
    #     "entities": {"platform": "linkedin", "topic": "AI"},
    #     "tokens_used": 500,
    #     "model": "claude-3-sonnet",
    #     "execution_time_ms": 1500,
    #     "node_trace": ["guardrail", "intent", "router", "orchestrator"]
    # }

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    # Indexes
    __table_args__ = (
        Index("idx_messages_conversation_created", "conversation_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, conv_id={self.conversation_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryDecomposition(Base):
    """
    Query decomposition records for multi-platform requests.
    Tracks how complex queries are broken down.
    """

    __tablename__ = "query_decompositions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="SET NULL")
    )

    # Original query
    original_query: Mapped[str] = mapped_column(Text, nullable=False)

    # Decomposition results
    is_multi_platform: Mapped[bool] = mapped_column(default=False)
    decomposed_queries: Mapped[list] = mapped_column(JSONB, default=list)
    # [
    #     {"platform": "linkedin", "query": "Generate LinkedIn post about AI", "priority": 1},
    #     {"platform": "facebook", "query": "Generate Facebook post about AI", "priority": 2}
    # ]

    # Execution tracking
    execution_strategy: Mapped[str] = mapped_column(String(50), default="sequential")  # sequential, parallel

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<QueryDecomposition(id={self.id}, multi_platform={self.is_multi_platform})>"
