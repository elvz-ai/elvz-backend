"""
Content and task database models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ElfType(str, Enum):
    """Types of Elf agents."""
    SOCIAL_MEDIA = "social_media"
    SEO = "seo"
    COPYWRITER = "copywriter"
    ASSISTANT = "assistant"


class ContentType(str, Enum):
    """Types of generated content."""
    SOCIAL_POST = "social_post"
    BLOG_POST = "blog_post"
    AD_COPY = "ad_copy"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"
    SEO_META = "seo_meta"
    DOCUMENT = "document"


class Platform(str, Enum):
    """Social media and content platforms."""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    GOOGLE = "google"
    BLOG = "blog"
    EMAIL = "email"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentHistory(Base):
    """
    History of generated content for analytics and RAFT learning.
    """
    
    __tablename__ = "content_history"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # Content Classification
    elf_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    platform: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    
    # Content Data
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    # Metadata includes: hashtags, timing, visual recommendations, etc.
    
    # Input Context (for learning)
    input_request: Mapped[dict] = mapped_column(JSON, nullable=False)
    # What the user asked for
    
    # Performance Metrics (updated after publication)
    performance_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    # {
    #     "impressions": 1000,
    #     "engagement_rate": 0.05,
    #     "clicks": 50,
    #     "shares": 10,
    #     "comments": 5
    # }
    performance_score: Mapped[Optional[float]] = mapped_column(Float)  # Normalized 0-1
    
    # User Feedback
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 stars
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)
    was_published: Mapped[bool] = mapped_column(default=False)
    was_edited: Mapped[bool] = mapped_column(default=False)
    
    # Generation Metadata
    model_used: Mapped[Optional[str]] = mapped_column(String(100))
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    generation_cost: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    def __repr__(self) -> str:
        return f"<ContentHistory(id={self.id}, type={self.content_type})>"


class Task(Base):
    """
    Task execution record for tracking Elf operations.
    """
    
    __tablename__ = "tasks"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    
    # Task Classification
    elf_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # e.g., "create_post", "audit_site", "write_blog"
    
    # Request/Response
    request_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_data: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Execution Details
    status: Mapped[str] = mapped_column(
        String(50), default=TaskStatus.PENDING.value, nullable=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance Metrics
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float)
    
    # Agent Execution Trace
    execution_trace: Mapped[Optional[list]] = mapped_column(JSON)
    # [
    #     {"agent": "strategy", "status": "completed", "time_ms": 500},
    #     {"agent": "content", "status": "completed", "time_ms": 1200},
    #     ...
    # ]
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    def __repr__(self) -> str:
        return f"<Task(id={self.id}, elf={self.elf_type}, status={self.status})>"
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate task duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

