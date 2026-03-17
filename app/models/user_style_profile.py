"""
UserStyleProfile model — stores pre-computed writing style features per user.

Computed at webhook time from scraped social posts and used by the RAG retriever
to inject style context into artifact generation prompts without re-scanning Qdrant.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class UserStyleProfile(Base):
    """
    Pre-computed writing style profile for a user.

    Written at webhook time (once per extraction job) and read at chat time.
    FK to user.id with CASCADE — user must exist before upserting a style profile.
    """

    __tablename__ = "user_style_profiles"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True
    )
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    posts_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False, default="VERY LOW")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
