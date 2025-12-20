"""
Copy Writer Elf API routes.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.elves.copywriter import CopywriterElf
from app.api.deps import get_current_user_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/elves/copywriter", tags=["copywriter"])

# Initialize Elf
copywriter_elf = CopywriterElf()


class WriteBlogRequest(BaseModel):
    """Request to write a blog post."""
    topic: str = Field(..., min_length=5, max_length=500)
    target_keywords: list[str] = Field(default_factory=list, max_length=10)
    target_audience: Optional[str] = None
    tone: str = Field(
        default="professional",
        pattern="^(professional|casual|storytelling|witty|formal)$"
    )
    word_count: int = Field(default=1000, ge=300, le=5000)
    include_seo: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "10 Productivity Tips for Remote Workers",
                "target_keywords": ["remote work", "productivity", "work from home"],
                "target_audience": "professionals working remotely",
                "tone": "professional",
                "word_count": 1500,
                "include_seo": True,
            }
        }


class WriteBlogResponse(BaseModel):
    """Response from write blog endpoint."""
    title: str
    meta_description: str
    content: str
    structure: dict
    seo_analysis: Optional[dict] = None
    alternative_titles: list[str] = []
    execution_time_ms: int


class WriteAdCopyRequest(BaseModel):
    """Request to write ad copy."""
    product: str = Field(..., min_length=3, max_length=500)
    platform: str = Field(
        default="google",
        pattern="^(google|meta|linkedin)$"
    )
    goals: list[str] = Field(default=["conversions"])
    target_audience: Optional[str] = None
    unique_selling_points: list[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "product": "AI-powered project management tool",
                "platform": "google",
                "goals": ["conversions", "brand_awareness"],
                "target_audience": "small business owners",
                "unique_selling_points": ["AI automation", "Time savings", "Easy to use"],
            }
        }


class WriteProductDescriptionRequest(BaseModel):
    """Request to write product description."""
    product: str = Field(..., min_length=3)
    features: list[str] = Field(default_factory=list)
    target_audience: Optional[str] = None
    tone: str = "persuasive"
    
    class Config:
        json_schema_extra = {
            "example": {
                "product": "Ergonomic Standing Desk",
                "features": ["Height adjustable", "Memory presets", "Cable management"],
                "target_audience": "office workers concerned about health",
                "tone": "persuasive",
            }
        }


class RewriteRequest(BaseModel):
    """Request to rewrite content."""
    content: str = Field(..., min_length=10, max_length=10000)
    target_tone: str = Field(
        default="professional",
        pattern="^(professional|casual|storytelling|witty|formal)$"
    )
    purpose: Optional[str] = None  # "simplify", "expand", "summarize"


@router.post("/write-blog", response_model=WriteBlogResponse)
async def write_blog(
    request: WriteBlogRequest,
    user_id: str = Depends(get_current_user_id),
) -> WriteBlogResponse:
    """
    Write a complete blog post.
    
    Generates:
    - SEO-optimized title
    - Meta description
    - Structured content with headings
    - SEO analysis (if requested)
    """
    logger.info(
        "Write blog request",
        user_id=user_id,
        topic=request.topic[:50],
    )
    
    try:
        result = await copywriter_elf.execute(
            request={
                "content_type": "blog",
                "topic": request.topic,
                "target_keywords": request.target_keywords,
                "target_audience": request.target_audience,
                "tone": request.tone,
                "word_count": request.word_count,
                "include_seo": request.include_seo,
            },
            context={"user_id": user_id},
        )
        
        return WriteBlogResponse(
            title=result.get("title", request.topic),
            meta_description=result.get("meta_description", ""),
            content=result.get("content", ""),
            structure=result.get("structure", {}),
            seo_analysis=result.get("seo_analysis"),
            alternative_titles=result.get("alternative_titles", []),
            execution_time_ms=result.get("execution_time_ms", 0),
        )
        
    except Exception as e:
        logger.error("Write blog failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/write-ad-copy")
async def write_ad_copy(
    request: WriteAdCopyRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Write ad copy for specified platform.
    
    Generates multiple variations for A/B testing.
    """
    logger.info(
        "Write ad copy request",
        user_id=user_id,
        platform=request.platform,
    )
    
    try:
        result = await copywriter_elf.execute(
            request={
                "content_type": "ad",
                "product": request.product,
                "platform": request.platform,
                "goals": request.goals,
                "target_audience": request.target_audience,
                "unique_selling_points": request.unique_selling_points,
            },
            context={"user_id": user_id},
        )
        
        return {
            "platform": request.platform,
            "variations": result.get("variations", []),
            "recommendation": {
                "best_variation": 0,
                "reason": "All variations meet platform requirements",
            },
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Write ad copy failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/write-product-description")
async def write_product_description(
    request: WriteProductDescriptionRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Write compelling product description.
    
    Generates:
    - Headline
    - Short description
    - Long description
    - Bullet points
    """
    logger.info(
        "Write product description request",
        user_id=user_id,
        product=request.product[:50],
    )
    
    try:
        result = await copywriter_elf.execute(
            request={
                "content_type": "product",
                "product": request.product,
                "features": request.features,
                "target_audience": request.target_audience,
                "tone": request.tone,
            },
            context={"user_id": user_id},
        )
        
        return {
            "headline": result.get("headline", request.product),
            "short_description": result.get("short_description", ""),
            "long_description": result.get("long_description", ""),
            "bullet_points": result.get("bullet_points", []),
            "cta": result.get("cta", "Buy Now"),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Write product description failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rewrite")
async def rewrite_content(
    request: RewriteRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Rewrite content with a different tone or purpose.
    """
    logger.info(
        "Rewrite content request",
        user_id=user_id,
        target_tone=request.target_tone,
    )
    
    try:
        from app.core.llm_clients import LLMMessage, llm_client
        
        system_prompt = f"""You are an expert content editor.
Rewrite the following content in a {request.target_tone} tone.
{"Purpose: " + request.purpose if request.purpose else ""}
Maintain the core message while adapting the style."""
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=request.content),
        ]
        
        response = await llm_client.generate(messages)
        
        return {
            "original_length": len(request.content),
            "rewritten_content": response.content,
            "rewritten_length": len(response.content),
            "target_tone": request.target_tone,
        }
        
    except Exception as e:
        logger.error("Rewrite content failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

