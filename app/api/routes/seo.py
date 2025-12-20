"""
SEO Optimizer Elf API routes.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.agents.elves.seo_optimizer import SEOOptimizerElf
from app.api.deps import get_current_user_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/elves/seo", tags=["seo"])

# Initialize Elf
seo_elf = SEOOptimizerElf()


class AuditSiteRequest(BaseModel):
    """Request to audit a website."""
    website_url: str = Field(..., min_length=10)
    include_competitors: bool = False
    competitor_urls: Optional[list[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "website_url": "https://example.com",
                "include_competitors": True,
                "competitor_urls": ["https://competitor1.com", "https://competitor2.com"]
            }
        }


class TechnicalIssue(BaseModel):
    """Technical SEO issue."""
    type: str
    severity: str
    page_url: str
    description: str
    fix_suggestion: str


class AuditSiteResponse(BaseModel):
    """Response from site audit."""
    overall_score: float
    technical_issues: list[dict]
    keyword_opportunities: list[dict]
    content_gaps: list[str]
    recommendations: list[dict]
    execution_time_ms: int


class OptimizePageRequest(BaseModel):
    """Request to optimize a page."""
    page_url: str
    target_keywords: list[str] = Field(default_factory=list, max_length=10)


class KeywordResearchRequest(BaseModel):
    """Request for keyword research."""
    seed_keywords: list[str] = Field(..., min_length=1, max_length=10)
    target_url: Optional[str] = None
    location: str = "US"
    language: str = "en"


@router.post("/audit-site", response_model=AuditSiteResponse)
async def audit_site(
    request: AuditSiteRequest,
    user_id: str = Depends(get_current_user_id),
) -> AuditSiteResponse:
    """
    Perform comprehensive SEO audit on a website.
    
    Analyzes:
    - Technical SEO issues (broken links, missing tags, etc.)
    - Keyword opportunities
    - Content gaps
    - Competitor comparison (if requested)
    
    Returns prioritized recommendations.
    """
    logger.info(
        "Audit site request",
        user_id=user_id,
        url=request.website_url,
    )
    
    try:
        result = await seo_elf.execute(
            request={
                "website_url": request.website_url,
                "include_competitors": request.include_competitors,
                "competitor_urls": request.competitor_urls or [],
            },
            context={"user_id": user_id},
        )
        
        return AuditSiteResponse(
            overall_score=result.get("overall_score", 0),
            technical_issues=result.get("technical_issues", []),
            keyword_opportunities=result.get("keyword_opportunities", []),
            content_gaps=result.get("content_gaps", []),
            recommendations=result.get("recommendations", []),
            execution_time_ms=result.get("execution_time_ms", 0),
        )
        
    except Exception as e:
        logger.error("Audit site failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-page")
async def optimize_page(
    request: OptimizePageRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Get optimization suggestions for a specific page.
    
    Returns:
    - Meta tag recommendations
    - Content optimization suggestions
    - Schema markup suggestions
    - Internal linking opportunities
    """
    logger.info(
        "Optimize page request",
        user_id=user_id,
        url=request.page_url,
    )
    
    try:
        # Run focused audit on single page
        result = await seo_elf.execute(
            request={
                "website_url": request.page_url,
                "target_keywords": request.target_keywords,
                "single_page": True,
            },
            context={"user_id": user_id},
        )
        
        return {
            "page_url": request.page_url,
            "target_keywords": request.target_keywords,
            "meta_suggestions": result.get("meta_suggestions", {}),
            "content_suggestions": result.get("content_suggestions", []),
            "schema_suggestions": result.get("schema_suggestions", []),
            "internal_linking": [],  # TODO: Implement
            "execution_time_ms": result.get("execution_time_ms", 0),
        }
        
    except Exception as e:
        logger.error("Optimize page failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyword-research")
async def keyword_research(
    request: KeywordResearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Research keywords related to seed keywords.
    
    Returns:
    - Keyword suggestions with volume and difficulty
    - Related topics
    - Questions people ask
    - Long-tail variations
    """
    logger.info(
        "Keyword research request",
        user_id=user_id,
        keywords=request.seed_keywords,
    )
    
    try:
        from app.tools.registry import tool_registry
        from app.tools.seo_tools import KeywordResearchInput
        
        tool = tool_registry.get("keyword_research")
        if not tool:
            raise HTTPException(status_code=503, detail="Keyword tool not available")
        
        result = await tool.execute(
            KeywordResearchInput(
                seed_keywords=request.seed_keywords,
                target_url=request.target_url,
                location=request.location,
                language=request.language,
            )
        )
        
        if result.success:
            return {
                "seed_keywords": request.seed_keywords,
                "keywords": result.data.get("keywords", []),
                "related_topics": result.data.get("related_topics", []),
                "questions": result.data.get("questions", []),
            }
        
        raise HTTPException(status_code=500, detail="Keyword research failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Keyword research failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-check")
async def quick_check(
    url: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Quick SEO health check for a URL.
    
    Returns basic metrics without full audit.
    """
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            content = response.text.lower()
            
            checks = {
                "https": url.startswith("https"),
                "has_title": "<title>" in content and "</title>" in content,
                "has_meta_description": 'name="description"' in content,
                "has_h1": "<h1" in content,
                "has_viewport": 'name="viewport"' in content,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "status_code": response.status_code,
            }
            
            score = sum([
                checks["https"] * 20,
                checks["has_title"] * 20,
                checks["has_meta_description"] * 20,
                checks["has_h1"] * 15,
                checks["has_viewport"] * 15,
                (checks["response_time_ms"] < 1000) * 10,
            ])
            
            return {
                "url": url,
                "quick_score": score,
                "checks": checks,
                "recommendation": "Run full audit for detailed analysis" if score < 80 else "Good basic SEO",
            }
            
    except Exception as e:
        logger.error("Quick check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

