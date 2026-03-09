"""
Pydantic schemas for the New Post wizard endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GeneratePostRequest(BaseModel):
    """Request to generate social media content via the wizard flow."""

    platforms: list[str] = Field(
        ...,
        min_length=1,
        description="Target platforms from Step 1",
        json_schema_extra={"examples": [["linkedin", "instagram"]]},
    )
    idea: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Descriptive content idea/brief",
    )
    tone: str = Field(
        default="professional",
        pattern="^(professional|engaging|concise|witty)$",
        description="Tone chip selection",
    )
    length: str = Field(
        default="keep",
        pattern="^(shorter|keep|longer)$",
        description="Length preference or adjustment",
    )
    draft: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="User's draft text or previously generated text for refinement",
    )


class GeneratePostArtifact(BaseModel):
    """Single artifact in the generate response."""

    id: str
    platform: str
    content: dict
    status: str = "draft"


class GeneratePostResponse(BaseModel):
    """Response from the generate endpoint."""

    batch_id: str
    artifacts: list[GeneratePostArtifact]
    execution_time_ms: int


class GenerateImageRequest(BaseModel):
    """Request to generate an AI image for an artifact."""

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Text prompt describing the desired image",
    )


class GenerateImageResponse(BaseModel):
    """Response from the image generation endpoint."""

    image_url: str
    artifact_id: str
    credits_used: int = 5
