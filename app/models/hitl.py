"""
Human-in-the-Loop (HITL) database models.
Manages approval workflows and user interactions during generation.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.conversation import Conversation


class HITLRequestType(str, Enum):
    """Types of HITL requests."""
    CLARIFICATION = "clarification"  # Need more info from user
    APPROVAL = "approval"  # Content needs approval before proceeding
    EDIT = "edit"  # User wants to modify content
    PLATFORM_SELECTION = "platform_selection"  # User needs to select platforms
    DATA_MISSING = "data_missing"  # Missing social media handle or data
    CONFIRMATION = "confirmation"  # Confirm action before proceeding


class HITLStatus(str, Enum):
    """HITL request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class HITLRequest(Base):
    """
    Human-in-the-Loop request for user interaction.
    Used when the system needs user input to proceed.
    """

    __tablename__ = "hitl_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("artifacts.id", ondelete="SET NULL")
    )

    # Request details
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default=HITLStatus.PENDING.value, nullable=False
    )

    # Question/prompt for user
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Options for user (if applicable)
    options: Mapped[Optional[list]] = mapped_column(JSONB)
    # [
    #     {"id": "opt1", "label": "LinkedIn", "description": "Generate for LinkedIn"},
    #     {"id": "opt2", "label": "Facebook", "description": "Generate for Facebook"}
    # ]

    # User response
    response: Mapped[Optional[str]] = mapped_column(Text)
    selected_options: Mapped[Optional[list]] = mapped_column(JSONB)  # IDs of selected options

    # Context for resumption
    context: Mapped[dict] = mapped_column(JSONB, default=dict)
    # {
    #     "graph_state_snapshot": {...},
    #     "pending_node": "multi_platform_orchestrator",
    #     "platforms_pending": ["linkedin"],
    #     "original_query": "Generate posts for all platforms"
    # }

    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Timestamps
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Notes
    requester_notes: Mapped[Optional[str]] = mapped_column(Text)  # Why this was triggered
    responder_notes: Mapped[Optional[str]] = mapped_column(Text)  # User's additional notes

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="hitl_requests")
    artifact: Mapped[Optional["Artifact"]] = relationship("Artifact")

    # Indexes
    __table_args__ = (
        Index("idx_hitl_status", "status"),
        Index("idx_hitl_conversation_status", "conversation_id", "status"),
        Index("idx_hitl_expires", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<HITLRequest(id={self.id}, type={self.request_type}, status={self.status})>"

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == HITLStatus.PENDING.value

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at and self.status == HITLStatus.PENDING.value:
            return datetime.now(self.expires_at.tzinfo) > self.expires_at
        return self.status == HITLStatus.EXPIRED.value

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "artifact_id": self.artifact_id,
            "request_type": self.request_type,
            "status": self.status,
            "prompt": self.prompt,
            "options": self.options,
            "response": self.response,
            "selected_options": self.selected_options,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
        }

    def approve(self, notes: Optional[str] = None) -> None:
        """Approve the HITL request."""
        self.status = HITLStatus.APPROVED.value
        self.responded_at = datetime.utcnow()
        if notes:
            self.responder_notes = notes

    def reject(self, notes: Optional[str] = None) -> None:
        """Reject the HITL request."""
        self.status = HITLStatus.REJECTED.value
        self.responded_at = datetime.utcnow()
        if notes:
            self.responder_notes = notes

    def respond(self, response: str, selected_options: Optional[list] = None, notes: Optional[str] = None) -> None:
        """Provide a response to the HITL request."""
        self.response = response
        self.selected_options = selected_options
        self.status = HITLStatus.APPROVED.value
        self.responded_at = datetime.utcnow()
        if notes:
            self.responder_notes = notes
