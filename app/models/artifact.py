"""
Artifact and ArtifactBatch database models.
Stores generated content from the conversational chatbot.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation, Message


class ArtifactType(str, Enum):
    """Types of generated artifacts."""
    SOCIAL_POST = "social_post"
    IMAGE = "image"
    VIDEO_SCRIPT = "video_script"
    HASHTAGS = "hashtags"
    SCHEDULE = "schedule"
    BLOG_POST = "blog_post"
    AD_COPY = "ad_copy"


class ArtifactStatus(str, Enum):
    """Artifact lifecycle status."""
    DRAFT = "draft"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class Platform(str, Enum):
    """Supported social media platforms."""
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    TIKTOK = "tiktok"


class ArtifactBatch(Base):
    """
    Groups artifacts generated from a single multi-platform request.
    """

    __tablename__ = "artifact_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Query that generated this batch
    query_decomposition_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("query_decompositions.id", ondelete="SET NULL")
    )

    # Batch metadata
    platforms: Mapped[list] = mapped_column(ARRAY(String), nullable=False)
    topic: Mapped[Optional[str]] = mapped_column(String(500))

    # Execution info
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )  # pending, in_progress, complete, partial, failed
    execution_strategy: Mapped[str] = mapped_column(String(50), default="sequential")

    # Performance tracking
    total_tokens_used: Mapped[int] = mapped_column(default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    execution_time_ms: Mapped[int] = mapped_column(default=0)

    # Metadata
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation")
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", back_populates="batch", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ArtifactBatch(id={self.id}, platforms={self.platforms}, status={self.status})>"

    @property
    def artifact_count(self) -> int:
        """Get total artifact count in batch."""
        return len(self.artifacts) if self.artifacts else 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "platforms": self.platforms,
            "topic": self.topic,
            "status": self.status,
            "artifact_count": self.artifact_count,
            "total_tokens_used": self.total_tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Artifact(Base):
    """
    Generated content artifact from the chatbot.
    Can be a social post, image, video script, etc.
    """

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="SET NULL")
    )
    batch_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("artifact_batches.id", ondelete="SET NULL"), index=True
    )

    # Artifact classification
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    platform: Mapped[Optional[str]] = mapped_column(String(50), index=True)

    # Content - structure depends on artifact_type
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # For social_post:
    # {
    #     "text": "Post content here...",
    #     "hook": "Attention-grabbing opener",
    #     "cta": "Call to action",
    #     "hashtags": ["#AI", "#Innovation"],
    #     "image_url": "https://...",
    #     "image_description": "...",
    #     "schedule": {"datetime": "2024-02-15T10:00:00Z", "timezone": "UTC"},
    #     "engagement_estimate": {"reach": 1000, "engagement_rate": 0.05}
    # }

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50), default=ArtifactStatus.DRAFT.value, nullable=False
    )

    # User feedback
    user_rating: Mapped[Optional[int]] = mapped_column()  # 1-5 stars
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)
    was_edited: Mapped[bool] = mapped_column(default=False)
    was_published: Mapped[bool] = mapped_column(default=False)

    # Generation metadata
    generation_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    # {
    #     "model": "claude-3-opus",
    #     "tokens_used": 1500,
    #     "cost": 0.05,
    #     "generation_time_ms": 3000,
    #     "prompt_template": "social_post_v2",
    #     "rag_sources": ["post_123", "post_456"],
    #     "brand_voice_confidence": 0.85
    # }

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="artifacts")
    message: Mapped[Optional["Message"]] = relationship("Message")
    batch: Mapped[Optional["ArtifactBatch"]] = relationship("ArtifactBatch", back_populates="artifacts")

    # Indexes
    __table_args__ = (
        Index("idx_artifacts_type_platform", "artifact_type", "platform"),
        Index("idx_artifacts_status", "status"),
        Index("idx_artifacts_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Artifact(id={self.id}, type={self.artifact_type}, platform={self.platform})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "batch_id": self.batch_id,
            "artifact_type": self.artifact_type,
            "platform": self.platform,
            "content": self.content,
            "status": self.status,
            "user_rating": self.user_rating,
            "was_published": self.was_published,
            "generation_metadata": self.generation_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @property
    def text_content(self) -> Optional[str]:
        """Extract text content from artifact."""
        if isinstance(self.content, dict):
            return self.content.get("text")
        return None

    @property
    def image_url(self) -> Optional[str]:
        """Extract image URL from artifact."""
        if isinstance(self.content, dict):
            return self.content.get("image_url")
        return None
