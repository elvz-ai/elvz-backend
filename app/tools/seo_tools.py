"""
SEO tools for website optimization.
Tools for crawling, keyword research, backlink analysis, and technical audits.
"""

import asyncio
from typing import Optional
from urllib.parse import urlparse

import httpx
import structlog
from pydantic import BaseModel

from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = structlog.get_logger(__name__)


# Input/Output Models

class TechnicalAuditInput(BaseModel):
    """Input for technical SEO audit."""
    website_url: str
    check_sitemap: bool = True
    check_robots: bool = True
    max_pages: int = 50


class TechnicalIssue(BaseModel):
    """Single technical SEO issue."""
    type: str  # "broken_link", "missing_meta", "slow_page", etc.
    severity: str  # "critical", "high", "medium", "low"
    page_url: str
    description: str
    fix_suggestion: str


class TechnicalAuditOutput(BaseModel):
    """Output for technical SEO audit."""
    issues: list[TechnicalIssue]
    pages_crawled: int
    sitemap_found: bool
    robots_found: bool
    https_enabled: bool
    mobile_friendly: bool


class KeywordResearchInput(BaseModel):
    """Input for keyword research."""
    seed_keywords: list[str]
    target_url: Optional[str] = None
    location: str = "US"
    language: str = "en"


class KeywordData(BaseModel):
    """Single keyword data."""
    keyword: str
    search_volume: int
    difficulty: float  # 0-100
    cpc: Optional[float] = None
    current_rank: Optional[int] = None
    trend: str = "stable"  # "rising", "falling", "stable"


class KeywordResearchOutput(BaseModel):
    """Output for keyword research."""
    keywords: list[KeywordData]
    related_topics: list[str]
    questions: list[str]


class BacklinkAnalysisInput(BaseModel):
    """Input for backlink analysis."""
    website_url: str
    compare_with: Optional[list[str]] = None


class Backlink(BaseModel):
    """Single backlink."""
    source_url: str
    target_url: str
    anchor_text: str
    domain_authority: float
    is_dofollow: bool


class BacklinkAnalysisOutput(BaseModel):
    """Output for backlink analysis."""
    total_backlinks: int
    referring_domains: int
    domain_authority: float
    top_backlinks: list[Backlink]
    competitor_gaps: list[str]


class ContentAnalysisInput(BaseModel):
    """Input for content analysis."""
    page_url: str
    target_keywords: list[str]


class ContentAnalysisOutput(BaseModel):
    """Output for content analysis."""
    word_count: int
    keyword_density: dict[str, float]
    readability_score: float
    heading_structure: list[dict]
    internal_links: int
    external_links: int
    suggestions: list[str]


# Tool Implementations

@register_tool(category="seo")
class TechnicalSEOAuditTool(BaseTool[TechnicalAuditInput, TechnicalAuditOutput]):
    """
    Perform technical SEO audit on a website.
    Checks for common technical issues.
    """
    
    name = "technical_seo_audit"
    description = "Audit website for technical SEO issues"
    timeout_seconds = 120  # Longer timeout for crawling
    cache_ttl = 86400  # 24 hour cache
    
    async def _execute(self, input_data: TechnicalAuditInput) -> TechnicalAuditOutput:
        """Execute technical SEO audit."""
        
        issues = []
        parsed_url = urlparse(input_data.website_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check HTTPS
        https_enabled = parsed_url.scheme == "https"
        if not https_enabled:
            issues.append(TechnicalIssue(
                type="no_https",
                severity="critical",
                page_url=input_data.website_url,
                description="Website is not using HTTPS",
                fix_suggestion="Install SSL certificate and redirect HTTP to HTTPS",
            ))
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Check robots.txt
            robots_found = False
            try:
                robots_response = await client.get(f"{base_url}/robots.txt")
                robots_found = robots_response.status_code == 200
            except Exception:
                pass
            
            if not robots_found:
                issues.append(TechnicalIssue(
                    type="missing_robots",
                    severity="medium",
                    page_url=f"{base_url}/robots.txt",
                    description="robots.txt file not found",
                    fix_suggestion="Create a robots.txt file to guide search engine crawlers",
                ))
            
            # Check sitemap
            sitemap_found = False
            try:
                sitemap_response = await client.get(f"{base_url}/sitemap.xml")
                sitemap_found = sitemap_response.status_code == 200
            except Exception:
                pass
            
            if not sitemap_found:
                issues.append(TechnicalIssue(
                    type="missing_sitemap",
                    severity="medium",
                    page_url=f"{base_url}/sitemap.xml",
                    description="XML sitemap not found",
                    fix_suggestion="Create an XML sitemap and submit to search engines",
                ))
            
            # Check homepage
            try:
                homepage_response = await client.get(input_data.website_url)
                
                # Check for meta tags
                content = homepage_response.text.lower()
                
                if "<title>" not in content or "</title>" not in content:
                    issues.append(TechnicalIssue(
                        type="missing_title",
                        severity="critical",
                        page_url=input_data.website_url,
                        description="Missing or empty title tag",
                        fix_suggestion="Add a unique, descriptive title tag (50-60 characters)",
                    ))
                
                if 'name="description"' not in content:
                    issues.append(TechnicalIssue(
                        type="missing_meta_description",
                        severity="high",
                        page_url=input_data.website_url,
                        description="Missing meta description",
                        fix_suggestion="Add a compelling meta description (150-160 characters)",
                    ))
                
                if 'name="viewport"' not in content:
                    issues.append(TechnicalIssue(
                        type="missing_viewport",
                        severity="high",
                        page_url=input_data.website_url,
                        description="Missing viewport meta tag",
                        fix_suggestion="Add viewport meta tag for mobile responsiveness",
                    ))
                
            except Exception as e:
                logger.error("Failed to fetch homepage", error=str(e))
        
        # Check mobile friendliness (simplified)
        mobile_friendly = 'name="viewport"' in content if 'content' in dir() else False
        
        return TechnicalAuditOutput(
            issues=issues,
            pages_crawled=1,  # Simplified for now
            sitemap_found=sitemap_found,
            robots_found=robots_found,
            https_enabled=https_enabled,
            mobile_friendly=mobile_friendly,
        )


@register_tool(category="seo")
class KeywordResearchTool(BaseTool[KeywordResearchInput, KeywordResearchOutput]):
    """
    Research keywords and search volumes.
    """
    
    name = "keyword_research"
    description = "Research keywords with search volumes and difficulty"
    cache_ttl = 86400  # 24 hour cache
    
    async def _execute(self, input_data: KeywordResearchInput) -> KeywordResearchOutput:
        """Execute keyword research."""
        
        # In production, would integrate with:
        # - SEMrush API
        # - Ahrefs API
        # - Google Keyword Planner
        
        keywords = []
        
        for seed in input_data.seed_keywords:
            # Main keyword
            keywords.append(KeywordData(
                keyword=seed,
                search_volume=5000,
                difficulty=45.0,
                cpc=1.50,
                trend="stable",
            ))
            
            # Long-tail variations
            keywords.append(KeywordData(
                keyword=f"how to {seed}",
                search_volume=2000,
                difficulty=30.0,
                cpc=0.80,
                trend="rising",
            ))
            
            keywords.append(KeywordData(
                keyword=f"best {seed}",
                search_volume=3000,
                difficulty=50.0,
                cpc=2.00,
                trend="stable",
            ))
            
            keywords.append(KeywordData(
                keyword=f"{seed} for beginners",
                search_volume=1500,
                difficulty=25.0,
                cpc=0.60,
                trend="rising",
            ))
        
        # Related topics
        related_topics = [
            f"{input_data.seed_keywords[0]} tools",
            f"{input_data.seed_keywords[0]} strategies",
            f"{input_data.seed_keywords[0]} examples",
        ] if input_data.seed_keywords else []
        
        # Common questions
        questions = [
            f"What is {input_data.seed_keywords[0]}?",
            f"How does {input_data.seed_keywords[0]} work?",
            f"Why is {input_data.seed_keywords[0]} important?",
        ] if input_data.seed_keywords else []
        
        return KeywordResearchOutput(
            keywords=keywords,
            related_topics=related_topics,
            questions=questions,
        )


@register_tool(category="seo")
class BacklinkAnalysisTool(BaseTool[BacklinkAnalysisInput, BacklinkAnalysisOutput]):
    """
    Analyze backlink profile of a website.
    """
    
    name = "backlink_analysis"
    description = "Analyze website backlink profile"
    cache_ttl = 86400  # 24 hour cache
    
    async def _execute(self, input_data: BacklinkAnalysisInput) -> BacklinkAnalysisOutput:
        """Execute backlink analysis."""
        
        # In production, would integrate with:
        # - Ahrefs API
        # - Moz API
        # - Majestic API
        
        # Placeholder data
        backlinks = [
            Backlink(
                source_url="https://example-blog.com/article",
                target_url=input_data.website_url,
                anchor_text="great resource",
                domain_authority=45.0,
                is_dofollow=True,
            ),
            Backlink(
                source_url="https://industry-news.com/roundup",
                target_url=input_data.website_url,
                anchor_text="company name",
                domain_authority=60.0,
                is_dofollow=True,
            ),
        ]
        
        competitor_gaps = []
        if input_data.compare_with:
            competitor_gaps = [
                "techcrunch.com (competitor has link, you don't)",
                "forbes.com (competitor has link, you don't)",
            ]
        
        return BacklinkAnalysisOutput(
            total_backlinks=150,
            referring_domains=45,
            domain_authority=35.0,
            top_backlinks=backlinks,
            competitor_gaps=competitor_gaps,
        )


@register_tool(category="seo")
class ContentAnalysisTool(BaseTool[ContentAnalysisInput, ContentAnalysisOutput]):
    """
    Analyze page content for SEO optimization.
    """
    
    name = "content_analysis"
    description = "Analyze page content for SEO"
    cache_ttl = 3600  # 1 hour cache
    
    async def _execute(self, input_data: ContentAnalysisInput) -> ContentAnalysisOutput:
        """Execute content analysis."""
        
        suggestions = []
        keyword_density = {}
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(input_data.page_url)
                content = response.text
                
                # Simple word count (in production, would parse HTML properly)
                text_content = content  # Would extract text from HTML
                words = text_content.split()
                word_count = len(words)
                
                # Calculate keyword density
                for keyword in input_data.target_keywords:
                    count = content.lower().count(keyword.lower())
                    density = (count / max(word_count, 1)) * 100
                    keyword_density[keyword] = round(density, 2)
                    
                    if density < 0.5:
                        suggestions.append(f"Increase usage of '{keyword}' (currently {density}%)")
                    elif density > 3.0:
                        suggestions.append(f"Reduce usage of '{keyword}' to avoid keyword stuffing")
                
                # Check heading structure
                heading_structure = []
                if "<h1" in content.lower():
                    heading_structure.append({"tag": "h1", "count": content.lower().count("<h1")})
                if "<h2" in content.lower():
                    heading_structure.append({"tag": "h2", "count": content.lower().count("<h2")})
                
                # Count links
                internal_links = content.lower().count('href="/')
                external_links = content.lower().count('href="http')
                
                if internal_links < 3:
                    suggestions.append("Add more internal links to improve site structure")
                
            except Exception as e:
                logger.error("Content analysis failed", error=str(e))
                word_count = 0
                heading_structure = []
                internal_links = 0
                external_links = 0
        
        return ContentAnalysisOutput(
            word_count=word_count,
            keyword_density=keyword_density,
            readability_score=70.0,  # Would use Flesch-Kincaid in production
            heading_structure=heading_structure,
            internal_links=internal_links,
            external_links=external_links,
            suggestions=suggestions,
        )

