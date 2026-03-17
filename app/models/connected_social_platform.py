"""
Read-only model for the connected_social_platform table (owned by Next.js / Drizzle).
Python never creates or alters this table — only reads it to check connection status.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConnectedSocialPlatform(Base):
    """Mirrors the Next.js-managed connected_social_platform table (read-only)."""

    __tablename__ = "connected_social_platform"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String, nullable=False)
    platform_username: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
