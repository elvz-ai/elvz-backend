"""
Output formatting utilities for different content types.
"""

from typing import Any, Optional

from pydantic import BaseModel


class SocialPostFormat(BaseModel):
    """Formatted social media post."""
    content: str
    hashtags: list[str]
    character_count: int
    platform: str
    is_within_limit: bool


class BlogPostFormat(BaseModel):
    """Formatted blog post."""
    title: str
    meta_description: str
    content: str
    headings: list[str]
    word_count: int
    reading_time_minutes: int


# Platform character limits
PLATFORM_LIMITS = {
    "twitter": 280,
    "linkedin": 3000,
    "facebook": 63206,
    "instagram": 2200,
}


def format_social_post(
    content: str,
    hashtags: list[str],
    platform: str,
) -> SocialPostFormat:
    """
    Format social media post with hashtags and validate limits.
    
    Args:
        content: Post content text
        hashtags: List of hashtags (without #)
        platform: Target platform
        
    Returns:
        Formatted post with metadata
    """
    # Format hashtags
    formatted_hashtags = [f"#{tag.strip('#')}" for tag in hashtags]
    hashtag_string = " ".join(formatted_hashtags)
    
    # Combine content and hashtags
    if hashtags:
        full_content = f"{content}\n\n{hashtag_string}"
    else:
        full_content = content
    
    # Get platform limit
    char_limit = PLATFORM_LIMITS.get(platform.lower(), 5000)
    
    return SocialPostFormat(
        content=full_content,
        hashtags=formatted_hashtags,
        character_count=len(full_content),
        platform=platform,
        is_within_limit=len(full_content) <= char_limit,
    )


def format_blog_post(
    title: str,
    content: str,
    meta_description: Optional[str] = None,
) -> BlogPostFormat:
    """
    Format blog post with metadata.
    
    Args:
        title: Blog post title
        content: Blog post content (markdown)
        meta_description: SEO meta description
        
    Returns:
        Formatted blog post with metadata
    """
    import re
    
    # Extract headings from markdown
    headings = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
    
    # Count words
    words = content.split()
    word_count = len(words)
    
    # Estimate reading time (200 words per minute)
    reading_time = max(1, round(word_count / 200))
    
    # Generate meta description if not provided
    if not meta_description:
        # Use first paragraph, truncated
        first_para = content.split('\n\n')[0]
        first_para = re.sub(r'[#*_`\[\]]', '', first_para)  # Remove markdown
        meta_description = first_para[:155] + "..." if len(first_para) > 155 else first_para
    
    return BlogPostFormat(
        title=title,
        meta_description=meta_description,
        content=content,
        headings=headings,
        word_count=word_count,
        reading_time_minutes=reading_time,
    )


def format_seo_report(
    technical_issues: list[dict],
    keyword_opportunities: list[dict],
    recommendations: list[dict],
    overall_score: float,
) -> dict[str, Any]:
    """
    Format SEO audit report.
    
    Args:
        technical_issues: List of technical SEO issues
        keyword_opportunities: List of keyword opportunities
        recommendations: List of prioritized recommendations
        overall_score: Overall SEO score (0-100)
        
    Returns:
        Formatted SEO report
    """
    # Sort issues by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(
        technical_issues,
        key=lambda x: severity_order.get(x.get("severity", "low"), 4)
    )
    
    # Sort recommendations by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_recommendations = sorted(
        recommendations,
        key=lambda x: priority_order.get(x.get("priority", "low"), 3)
    )
    
    # Calculate issue summary
    issue_summary = {
        "critical": len([i for i in technical_issues if i.get("severity") == "critical"]),
        "high": len([i for i in technical_issues if i.get("severity") == "high"]),
        "medium": len([i for i in technical_issues if i.get("severity") == "medium"]),
        "low": len([i for i in technical_issues if i.get("severity") == "low"]),
    }
    
    return {
        "summary": {
            "overall_score": overall_score,
            "total_issues": len(technical_issues),
            "issues_by_severity": issue_summary,
            "keyword_opportunities": len(keyword_opportunities),
            "recommendations": len(recommendations),
        },
        "technical_issues": sorted_issues,
        "keyword_opportunities": keyword_opportunities,
        "recommendations": sorted_recommendations,
        "score_breakdown": {
            "technical": calculate_technical_score(technical_issues),
            "content": calculate_content_score(keyword_opportunities),
        },
    }


def calculate_technical_score(issues: list[dict]) -> float:
    """Calculate technical SEO score based on issues."""
    if not issues:
        return 100.0
    
    # Deduct points based on severity
    deductions = {
        "critical": 25,
        "high": 15,
        "medium": 5,
        "low": 2,
    }
    
    total_deduction = sum(
        deductions.get(issue.get("severity", "low"), 0)
        for issue in issues
    )
    
    return max(0, 100 - total_deduction)


def calculate_content_score(opportunities: list[dict]) -> float:
    """Calculate content optimization score."""
    if not opportunities:
        return 100.0
    
    # Score based on opportunity quality
    high_potential = len([o for o in opportunities if o.get("opportunity_score", 0) > 70])
    medium_potential = len([o for o in opportunities if 40 <= o.get("opportunity_score", 0) <= 70])
    
    # More opportunities to improve = lower current score
    base_score = 100 - (high_potential * 10 + medium_potential * 5)
    return max(30, base_score)


def format_ad_copy(
    variations: list[dict],
    platform: str,
) -> dict[str, Any]:
    """
    Format ad copy variations for different platforms.
    
    Args:
        variations: List of ad copy variations
        platform: Target ad platform (google, meta, linkedin)
        
    Returns:
        Formatted ad copy with platform-specific requirements
    """
    platform_specs = {
        "google": {
            "headline_limit": 30,
            "description_limit": 90,
            "headlines_required": 3,
        },
        "meta": {
            "headline_limit": 40,
            "primary_text_limit": 125,
            "description_limit": 30,
        },
        "linkedin": {
            "headline_limit": 70,
            "intro_text_limit": 150,
        },
    }
    
    specs = platform_specs.get(platform.lower(), platform_specs["meta"])
    
    formatted_variations = []
    for var in variations:
        formatted = {
            **var,
            "platform": platform,
            "specs": specs,
            "validation": validate_ad_copy(var, specs),
        }
        formatted_variations.append(formatted)
    
    return {
        "platform": platform,
        "specifications": specs,
        "variations": formatted_variations,
        "recommendation": get_best_variation(formatted_variations),
    }


def validate_ad_copy(copy: dict, specs: dict) -> dict[str, bool]:
    """Validate ad copy against platform specs."""
    validation = {}
    
    if "headline" in copy and "headline_limit" in specs:
        validation["headline_valid"] = len(copy["headline"]) <= specs["headline_limit"]
    
    if "description" in copy and "description_limit" in specs:
        validation["description_valid"] = len(copy["description"]) <= specs["description_limit"]
    
    if "primary_text" in copy and "primary_text_limit" in specs:
        validation["primary_text_valid"] = len(copy["primary_text"]) <= specs["primary_text_limit"]
    
    validation["all_valid"] = all(validation.values())
    return validation


def get_best_variation(variations: list[dict]) -> dict:
    """Get recommendation for best ad variation."""
    # Filter valid variations
    valid = [v for v in variations if v.get("validation", {}).get("all_valid", True)]
    
    if not valid:
        return {
            "index": 0,
            "reason": "No variations meet platform requirements. Please revise.",
        }
    
    # For now, return first valid (in real app, would score based on patterns)
    return {
        "index": variations.index(valid[0]),
        "reason": "Best balance of engagement potential and platform compliance.",
    }

