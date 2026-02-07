"""
Social Media Manager Elf API routes.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.elves.social_media_manager import SocialMediaManagerElf
from app.api.deps import get_current_user_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/elves/social-media", tags=["social-media"])

# Initialize Elf
social_media_elf = SocialMediaManagerElf()


class CreatePostRequest(BaseModel):
    """Request to create social media post."""
    platform: str = Field(..., pattern="^(linkedin|twitter|facebook|instagram)$")
    topic: str = Field(..., min_length=3, max_length=500)
    content_type: str = Field(
        default="thought_leadership",
        pattern="^(announcement|thought_leadership|promotion|educational|engagement)$"
    )
    brand_voice: Optional[str] = None
    goals: list[str] = Field(default=["engagement"])
    additional_context: Optional[str] = None
    image: bool = Field(default=False, description="Whether to generate image content")
    video: bool = Field(default=False, description="Whether to generate video content")

    class Config:
        json_schema_extra = {
            "example": {
                "platform": "linkedin",
                "topic": "The future of AI in business automation",
                "content_type": "thought_leadership",
                "goals": ["awareness", "engagement"],
                "image": True,
                "video": False,
            }
        }


class ContentVariation(BaseModel):
    """Generated content variation."""
    version: str
    content: dict
    hashtags: list[dict]
    posting_schedule: dict
    visual_recommendations: list[dict]
    complete_preview: str
    estimated_engagement: dict


class CreatePostResponse(BaseModel):
    """Response from create post endpoint."""
    post_variations: list[ContentVariation]
    recommendations: dict
    execution_time_ms: int


class AnalyzePerformanceRequest(BaseModel):
    """Request to analyze content performance."""
    platform: str
    timeframe: str = "7d"  # "1d", "7d", "30d", "90d"
    content_ids: Optional[list[str]] = None


class GenerateCalendarRequest(BaseModel):
    """Request to generate content calendar."""
    platforms: list[str]
    duration_weeks: int = Field(default=4, ge=1, le=12)
    posts_per_week: int = Field(default=3, ge=1, le=10)
    topics: list[str] = []
    content_mix: Optional[dict] = None  # {"thought_leadership": 0.4, "promotion": 0.3, "engagement": 0.3}


@router.post("/create-post", response_model=CreatePostResponse)
async def create_post(
    request: CreatePostRequest,
    user_id: str = Depends(get_current_user_id),
) -> CreatePostResponse:
    """
    Create social media post with multiple variations.
    
    Generates:
    - 3 content variations (hook-focused, story-focused, value-focused)
    - Optimized hashtags for the platform
    - Best posting time recommendations
    - Visual content recommendations
    """
    logger.info(
        "Create post request",
        user_id=user_id,
        platform=request.platform,
        topic=request.topic[:50],
    )
    
    try:
        result = await social_media_elf.execute(
            request={
                "platform": request.platform,
                "topic": request.topic,
                "content_type": request.content_type,
                "brand_voice": request.brand_voice,
                "goals": request.goals,
                "additional_context": request.additional_context,
            },
            context={
                "user_id": user_id,
                "image": request.image,
                "video": request.video,
            },
        )
        
        return CreatePostResponse(
            post_variations=result.get("post_variations", []),
            recommendations=result.get("recommendations", {}),
            execution_time_ms=result.get("execution_time_ms", 0),
        )
        
    except Exception as e:
        logger.error("Create post failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-performance")
async def analyze_performance(
    request: AnalyzePerformanceRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Analyze social media content performance.
    
    Returns insights on:
    - Engagement rates
    - Best performing content
    - Audience insights
    - Recommendations for improvement
    """
    logger.info(
        "Analyze performance request",
        user_id=user_id,
        platform=request.platform,
    )
    
    # TODO: Implement with analytics data
    return {
        "platform": request.platform,
        "timeframe": request.timeframe,
        "summary": {
            "total_posts": 15,
            "avg_engagement_rate": 0.045,
            "total_impressions": 25000,
            "top_performing_content_type": "thought_leadership",
        },
        "insights": [
            "Posts with questions get 2x more engagement",
            "Best posting time: Tuesday 10 AM",
            "Carousel posts outperform single images by 40%",
        ],
        "recommendations": [
            "Increase posting frequency to 4x per week",
            "Add more video content",
            "Use trending industry hashtags",
        ],
    }


@router.post("/generate-calendar")
async def generate_calendar(
    request: GenerateCalendarRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Generate a content calendar for specified duration.
    
    Creates a strategic content plan with:
    - Post topics and angles
    - Optimal posting schedule
    - Content mix recommendations
    """
    logger.info(
        "Generate calendar request",
        user_id=user_id,
        platforms=request.platforms,
        weeks=request.duration_weeks,
    )
    
    # TODO: Implement full calendar generation
    from datetime import datetime, timedelta
    
    calendar = []
    current_date = datetime.utcnow()
    
    for week in range(request.duration_weeks):
        week_posts = []
        for i in range(request.posts_per_week):
            post_date = current_date + timedelta(weeks=week, days=i*2)
            week_posts.append({
                "date": post_date.strftime("%Y-%m-%d"),
                "platform": request.platforms[i % len(request.platforms)],
                "content_type": "thought_leadership",
                "topic_suggestion": request.topics[i % len(request.topics)] if request.topics else "Industry insight",
                "status": "planned",
            })
        calendar.append({
            "week": week + 1,
            "posts": week_posts,
        })
    
    return {
        "duration_weeks": request.duration_weeks,
        "platforms": request.platforms,
        "total_posts": request.duration_weeks * request.posts_per_week,
        "calendar": calendar,
    }


@router.get("/hashtag-suggestions/{platform}")
async def get_hashtag_suggestions(
    platform: str,
    topic: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Get hashtag suggestions for a topic and platform."""
    from app.tools.registry import tool_registry
    from app.tools.social_media_tools import HashtagSearchInput
    
    tool = tool_registry.get("hashtag_research")
    if not tool:
        raise HTTPException(status_code=503, detail="Hashtag tool not available")
    
    result = await tool.execute(
        HashtagSearchInput(
            keywords=[topic],
            platform=platform,
            max_results=10,
        )
    )
    
    if result.success:
        return {
            "platform": platform,
            "topic": topic,
            "hashtags": result.data.get("hashtags", []),
        }
    
    raise HTTPException(status_code=500, detail="Hashtag research failed")

