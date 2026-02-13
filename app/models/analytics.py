"""
Analytics and usage tracking models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MetricType(str, Enum):
    """Types of analytics metrics."""
    # Social Media Metrics
    IMPRESSIONS = "impressions"
    ENGAGEMENT_RATE = "engagement_rate"
    CLICKS = "clicks"
    SHARES = "shares"
    COMMENTS = "comments"
    FOLLOWERS_GAINED = "followers_gained"
    
    # SEO Metrics
    ORGANIC_TRAFFIC = "organic_traffic"
    KEYWORD_RANKINGS = "keyword_rankings"
    BACKLINKS = "backlinks"
    DOMAIN_AUTHORITY = "domain_authority"
    
    # Content Metrics
    PAGE_VIEWS = "page_views"
    TIME_ON_PAGE = "time_on_page"
    BOUNCE_RATE = "bounce_rate"
    CONVERSION_RATE = "conversion_rate"
    
    # Email Metrics
    OPEN_RATE = "open_rate"
    CLICK_RATE = "click_rate"
    UNSUBSCRIBE_RATE = "unsubscribe_rate"


class Analytics(Base):
    """
    User analytics data from various platforms.
    Used for optimizing content timing and strategy.
    """
    
    __tablename__ = "analytics"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # Metric Classification
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    platform: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    
    # Metric Value
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Context
    content_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    # Reference to ContentHistory if applicable
    
    # Time Dimension
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    hour_of_day: Mapped[Optional[int]] = mapped_column(Integer)  # 0-23
    day_of_week: Mapped[Optional[int]] = mapped_column(Integer)  # 0-6
    
    def __repr__(self) -> str:
        return f"<Analytics(user_id={self.user_id}, type={self.metric_type}, value={self.metric_value})>"


class APIUsage(Base):
    """
    API usage tracking for billing and rate limiting.
    """
    
    __tablename__ = "api_usage"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # Usage Classification
    elf_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Token Usage
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Cost
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Request Details
    model_used: Mapped[str] = mapped_column(String(100))
    request_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status
    success: Mapped[bool] = mapped_column(default=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    
    def __repr__(self) -> str:
        return f"<APIUsage(user_id={self.user_id}, elf={self.elf_type}, tokens={self.total_tokens})>"


class UsageSummary(Base):
    """
    Daily/monthly usage summaries for quick billing lookups.
    """
    
    __tablename__ = "usage_summaries"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # Period
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "daily", "monthly"
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Aggregate Usage
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Per-Elf Breakdown
    requests_by_elf: Mapped[dict] = mapped_column(JSON, default=dict)
    # {"social_media": 50, "seo": 30, "copywriter": 20, "assistant": 100}

    tokens_by_elf: Mapped[dict] = mapped_column(JSON, default=dict)
    cost_by_elf: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<UsageSummary(user_id={self.user_id}, period={self.period_type})>"

