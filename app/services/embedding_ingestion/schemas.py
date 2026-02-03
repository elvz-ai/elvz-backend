"""
Pydantic schemas for Embedding Ingestion API.

Derived from PRD 1 (Frontend Payload Specification).
"""

from typing import Optional, Literal, Union, Annotated
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class Modality(str, Enum):
    """Content modality types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class Platform(str, Enum):
    """Source platform types."""
    INSTAGRAM = "instagram"
    NONE = "none"


# BlockType is now a flexible string (not an enum)
# This allows frontend to send any block_type value
# Common values include:
#   - company_profile, company_positioning, company_voice, company_services
#   - market_analysis, competitive_landscape
#   - competitor_profile, competitor_swot, competitor_strategy
#   - ig_profile_text, ig_post_text, ig_post_image, ig_post_video_segment
#   - document_text_chunk
# The block_type is stored as metadata for filtering during retrieval


class SourceType(str, Enum):
    """Source of the ingestion request."""
    MANUAL = "manual"
    INSTAGRAM = "instagram"
    COMPETITOR = "competitor"
    DOCUMENT = "document"


# ============================================================================
# Content by Modality
# ============================================================================

class TextContent(BaseModel):
    """Text content payload."""
    text: str = Field(..., min_length=1, description="Text content to embed")


class ImageContent(BaseModel):
    """Image content payload."""
    image_url: str = Field(
        ..., 
        description="Permanent Firebase URL for the image"
    )


class VideoContent(BaseModel):
    """Video content payload with optional segment boundaries."""
    video_url: str = Field(
        ..., 
        description="Permanent Firebase URL for the video"
    )
    start_ms: Optional[int] = Field(
        None, 
        ge=0,
        description="Start time in milliseconds (for video segments)"
    )
    end_ms: Optional[int] = Field(
        None, 
        ge=0,
        description="End time in milliseconds (for video segments)"
    )


# Union type for content
ContentType = Union[TextContent, ImageContent, VideoContent]


# ============================================================================
# Block Metadata
# ============================================================================

class BlockMetadata(BaseModel):
    """
    Metadata attached to every embedding block.
    All fields stored in vector database for filtering.
    
    Note: modality and platform are OPTIONAL here because they
    already exist at the block level. The service will copy them
    from the block when storing.
    """
    company_id: str = Field(..., description="Treasury / company ID")
    user_id: str = Field(..., description="Influencer / owner ID")
    source_collection: str = Field(
        ..., 
        description="Firestore source collection name"
    )
    doc_id: Optional[str] = Field(
        None, 
        description="Firestore document ID"
    )
    post_type: Optional[Literal["Image", "Video", "Carousel"]] = Field(
        None,
        description="Instagram post type (if applicable)"
    )
    modality: Optional[Modality] = Field(
        None, 
        description="Content modality (optional - copied from block level)"
    )
    platform: Optional[Platform] = Field(
        None, 
        description="Source platform (optional - copied from block level)"
    )
    region: str = Field(..., description="Geographic region")
    language: str = Field(default="en", description="Content language")
    topic_tags: Optional[list[str]] = Field(
        default=None, 
        alias="topicTags",
        description="Optional topic tags for categorization"
    )
    updated_at: int = Field(
        ..., 
        description="Timestamp (Firestore updated_at or current)"
    )
    
    class Config:
        populate_by_name = True


# ============================================================================
# Embedding Block
# ============================================================================

class EmbeddingBlock(BaseModel):
    """
    Single embedding block - the core unit.
    Each block becomes one vector in the database.
    """
    block_id: str = Field(
        ..., 
        description="Stable unique ID (e.g., company:mission, ig_post:123:text)"
    )
    block_type: str = Field(
        ..., 
        description="Type of content block (flexible string, e.g., company_profile, ig_post_text)"
    )
    modality: Modality = Field(..., description="Content modality")
    platform: Platform = Field(..., description="Source platform")
    content: ContentType = Field(..., description="Modality-specific content")
    metadata: BlockMetadata = Field(..., description="Filterable metadata")


# ============================================================================
# Request / Response
# ============================================================================

class EmbeddingIngestionRequest(BaseModel):
    """
    Top-level ingestion request from frontend.
    Contains all blocks to be embedded and stored.
    """
    company_id: str = Field(..., description="Treasury / company ID")
    user_id: str = Field(..., description="Influencer / owner ID")
    region: str = Field(..., description="Geographic region")
    language: str = Field(default="en", description="Default language")
    source: SourceType = Field(..., description="Source of ingestion")
    blocks: list[EmbeddingBlock] = Field(
        ..., 
        min_length=1,
        description="Array of embedding blocks to process"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "company_123",
                "user_id": "user_456",
                "region": "us-west",
                "language": "en",
                "source": "manual",
                "blocks": [
                    {
                        "block_id": "company:mission",
                        "block_type": "company_profile",
                        "modality": "text",
                        "platform": "none",
                        "content": {
                            "text": "Our mission is to revolutionize..."
                        },
                        "metadata": {
                            "company_id": "company_123",
                            "user_id": "user_456",
                            "source_collection": "companies",
                            "doc_id": "company_123",
                            "region": "us-west",
                            "language": "en",
                            "updated_at": 1705500000
                        }
                    }
                ]
            }
        }


class BlockFailure(BaseModel):
    """Details of a failed block processing."""
    block_id: str = Field(..., description="ID of the failed block")
    reason: str = Field(..., description="Error message")


class EmbeddingIngestionResponse(BaseModel):
    """
    Response from embedding ingestion.
    Supports partial success - some blocks may fail while others succeed.
    """
    company_id: str = Field(..., description="Company ID from request")
    total_blocks: int = Field(..., description="Total blocks in request")
    embedded_blocks: int = Field(..., description="Successfully embedded blocks")
    failed_blocks: int = Field(..., description="Number of failed blocks")
    failures: Optional[list[BlockFailure]] = Field(
        None, 
        description="Details of failed blocks (if any)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "company_123",
                "total_blocks": 5,
                "embedded_blocks": 4,
                "failed_blocks": 1,
                "failures": [
                    {
                        "block_id": "ig_post:789:image",
                        "reason": "Image URL expired or inaccessible"
                    }
                ]
            }
        }

