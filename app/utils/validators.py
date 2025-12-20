"""
Input validation utilities for API requests.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ContentRequestValidator(BaseModel):
    """Validate content generation requests."""
    
    user_id: str = Field(..., min_length=1)
    platform: str = Field(..., pattern="^(linkedin|twitter|facebook|instagram|blog|email)$")
    content_type: Optional[str] = None
    topic: str = Field(..., min_length=3, max_length=500)
    tone: Optional[str] = Field(
        default="professional",
        pattern="^(professional|casual|formal|friendly|witty|storytelling)$"
    )
    word_count: Optional[int] = Field(default=None, ge=50, le=5000)
    
    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Ensure topic is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        return v.strip()


class SEOAuditRequestValidator(BaseModel):
    """Validate SEO audit requests."""
    
    user_id: str = Field(..., min_length=1)
    website_url: str = Field(..., pattern=r"^https?://.*")
    include_competitors: bool = False
    competitor_urls: Optional[list[str]] = None
    
    @field_validator("website_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is valid."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @field_validator("competitor_urls")
    @classmethod
    def validate_competitor_urls(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate competitor URLs if provided."""
        if v:
            for url in v:
                if not url.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid competitor URL: {url}")
        return v


class BlogRequestValidator(BaseModel):
    """Validate blog writing requests."""
    
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=5, max_length=500)
    target_keywords: Optional[list[str]] = Field(default_factory=list)
    target_audience: Optional[str] = None
    tone: str = Field(
        default="professional",
        pattern="^(professional|casual|storytelling|witty|formal)$"
    )
    word_count: int = Field(default=1000, ge=300, le=5000)
    include_seo: bool = True
    
    @field_validator("target_keywords")
    @classmethod
    def validate_keywords(cls, v: list[str]) -> list[str]:
        """Limit number of keywords."""
        if len(v) > 10:
            raise ValueError("Maximum 10 target keywords allowed")
        return [k.strip().lower() for k in v if k.strip()]


def validate_content_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize content request data.
    
    Args:
        request_data: Raw request data
        
    Returns:
        Validated and normalized request data
        
    Raises:
        ValueError: If validation fails
    """
    # Determine request type and validate accordingly
    if "website_url" in request_data:
        validator = SEOAuditRequestValidator(**request_data)
    elif "word_count" in request_data and request_data.get("word_count", 0) > 500:
        validator = BlogRequestValidator(**request_data)
    else:
        validator = ContentRequestValidator(**request_data)
    
    return validator.model_dump()


def validate_platform(platform: str) -> bool:
    """Check if platform is supported."""
    supported = {"linkedin", "twitter", "facebook", "instagram", "blog", "email", "google"}
    return platform.lower() in supported


def validate_elf_type(elf_type: str) -> bool:
    """Check if Elf type is valid."""
    valid_elves = {"social_media", "seo", "copywriter", "assistant"}
    return elf_type.lower().replace("-", "_") in valid_elves


def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input text for safety.
    
    - Truncates to max length
    - Removes control characters
    - Normalizes whitespace
    """
    import re
    
    # Truncate
    text = text[:max_length]
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

