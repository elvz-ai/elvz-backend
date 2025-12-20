"""
User-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SubscriptionTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class User(Base):
    """User account model."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    subscription_tier: Mapped[str] = mapped_column(
        String(50), default=SubscriptionTier.FREE.value, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    # Relationships
    profile: Mapped[Optional["UserProfile"]] = relationship(
        "UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    brand_voice: Mapped[Optional["BrandVoiceProfile"]] = relationship(
        "BrandVoiceProfile", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class UserProfile(Base):
    """User profile with brand and business information."""
    
    __tablename__ = "user_profiles"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    
    # Brand Information
    brand_name: Mapped[Optional[str]] = mapped_column(String(255))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    company_size: Mapped[Optional[str]] = mapped_column(String(50))
    website_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Brand Voice Settings
    brand_voice: Mapped[Optional[str]] = mapped_column(Text)  # Description of brand voice
    tone_preferences: Mapped[Optional[dict]] = mapped_column(JSON)  # {formal: 0.7, friendly: 0.8}
    
    # Target Audience
    target_audience: Mapped[Optional[dict]] = mapped_column(JSON)  # Demographics, interests
    buyer_personas: Mapped[Optional[list]] = mapped_column(JSON)
    
    # Goals & Preferences
    business_goals: Mapped[Optional[list]] = mapped_column(JSON)  # ["brand awareness", "leads"]
    content_preferences: Mapped[Optional[dict]] = mapped_column(JSON)
    social_platforms: Mapped[Optional[list]] = mapped_column(JSON)  # ["linkedin", "twitter"]
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="profile")
    
    def __repr__(self) -> str:
        return f"<UserProfile(user_id={self.user_id}, brand={self.brand_name})>"


class BrandVoiceProfile(Base):
    """
    Analyzed brand voice profile for RAFT-based content generation.
    Extracted from user's past content samples.
    """
    
    __tablename__ = "brand_voice_profiles"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    
    # Tone Characteristics (0-1 scale)
    tone_characteristics: Mapped[dict] = mapped_column(JSON, default=dict)
    # {
    #     "formal": 0.6,
    #     "friendly": 0.8,
    #     "authoritative": 0.7,
    #     "humorous": 0.3,
    #     "inspirational": 0.5
    # }
    
    # Vocabulary Patterns
    vocabulary_patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    # {
    #     "common_words": ["innovative", "transform", "empower"],
    #     "avoided_words": ["cheap", "basic"],
    #     "jargon_level": "moderate",
    #     "complexity": "accessible"
    # }
    
    # Sentence Structure
    sentence_structure: Mapped[dict] = mapped_column(JSON, default=dict)
    # {
    #     "avg_sentence_length": 15,
    #     "uses_questions": true,
    #     "uses_lists": true,
    #     "paragraph_style": "short"
    # }
    
    # Personality Traits
    personality_traits: Mapped[list] = mapped_column(JSON, default=list)
    # ["innovative", "customer-centric", "data-driven"]
    
    # Content Patterns
    content_patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    # {
    #     "typical_cta_style": "action-oriented",
    #     "storytelling_approach": "problem-solution",
    #     "emoji_usage": "minimal"
    # }
    
    # Sample Phrases (for few-shot)
    sample_phrases: Mapped[list] = mapped_column(JSON, default=list)
    
    # Analysis Metadata
    samples_analyzed: Mapped[int] = mapped_column(default=0)
    confidence_score: Mapped[float] = mapped_column(default=0.0)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="brand_voice")
    
    def __repr__(self) -> str:
        return f"<BrandVoiceProfile(user_id={self.user_id}, confidence={self.confidence_score})>"
    
    def to_prompt_context(self) -> str:
        """Convert profile to prompt-injectable context."""
        parts = []
        
        if self.tone_characteristics:
            tones = [f"{k}: {v:.0%}" for k, v in self.tone_characteristics.items() if v > 0.5]
            if tones:
                parts.append(f"Tone: {', '.join(tones)}")
        
        if self.personality_traits:
            parts.append(f"Personality: {', '.join(self.personality_traits)}")
        
        if self.vocabulary_patterns:
            common = self.vocabulary_patterns.get("common_words", [])
            if common:
                parts.append(f"Key vocabulary: {', '.join(common[:10])}")
        
        if self.content_patterns:
            cta = self.content_patterns.get("typical_cta_style")
            if cta:
                parts.append(f"CTA style: {cta}")
        
        if self.sample_phrases:
            parts.append(f"Example phrases: {'; '.join(self.sample_phrases[:3])}")
        
        return "\n".join(parts) if parts else "No brand voice profile available."

