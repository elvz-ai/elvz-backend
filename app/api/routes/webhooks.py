"""
Webhook routes — receives notifications from the Next.js app.

Flow (see docs/plans/PYTHON_ENGINE_INTEGRATION.md):
  1. User connects a social platform on elvz.ai and triggers extraction.
  2. Extraction completes → Next.js POSTs to /api/v1/webhook/extraction-complete
     with extractionJobId + userId + platform + connectedSocialPlatformId.
  3. This engine calls GET /api/internal/posts?extractionJobId={id} on the
     Next.js app to fetch the actual post data.
  4. Posts are embedded and upserted into Qdrant for RAG retrieval.

Auth: shared `x-api-key` header (ELVZ_PYTHON_ENGINE_API_KEY in .env on both sides).
"""

import hashlib
import hmac
from typing import Optional

import httpx
import structlog
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.vector_store import VectorDocument, vector_store

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


# ---------------------------------------------------------------------------
# Incoming webhook models (payload sent by Next.js)
# ---------------------------------------------------------------------------

class ExtractionCompletePayload(BaseModel):
    """
    Payload posted by the Next.js app when an extraction job finishes.

    Example:
    {
        "extractionJobId": "BQVZ9ugprx6o7ExsmQDBM",
        "userId": "Z2jpSL9wl94RHeotn64UZCh0DU5kkA7j",
        "platform": "linkedin",
        "connectedSocialPlatformId": "MW_XBSYIC7KxhiWVj9Frh"
    }
    """
    extractionJobId: str
    userId: str
    platform: str
    connectedSocialPlatformId: str


class WebhookResponse(BaseModel):
    received: bool = True
    indexed: int = 0
    skipped: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# Internal API response models (data fetched from Next.js)
# ---------------------------------------------------------------------------

class MediaItem(BaseModel):
    id: str
    type: str  # "image" | "video"
    url: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    sortOrder: int = 0


class PostEngagement(BaseModel):
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    views: int = 0
    engagementRate: Optional[float] = None


class ExtractedPost(BaseModel):
    id: str
    extractionJobId: str
    connectedSocialPlatformId: str
    platformPostId: str
    type: str  # "text" | "image" | "video" | "carousel"
    caption: Optional[str] = None
    url: Optional[str] = None
    postedAt: Optional[str] = None
    hashtags: list[str] = []
    mentions: list[str] = []
    media: list[MediaItem] = []
    engagement: PostEngagement = PostEngagement()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _verify_api_key(x_api_key: Optional[str]) -> None:
    """
    Validate the shared API key sent by the Next.js app.

    Uses constant-time comparison to prevent timing attacks.
    If elvz_api_key is not set (empty), validation is skipped in development
    but enforced in production.
    """
    if not settings.elvz_api_key:
        if settings.environment == "production":
            logger.warning("elvz_api_key not configured in production — rejecting request")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Webhook authentication is not configured on this server",
            )
        return  # dev/staging: allow through without a key

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header",
        )

    if not hmac.compare_digest(
        settings.elvz_api_key.encode(),
        x_api_key.encode(),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/extraction-complete", response_model=WebhookResponse)
async def extraction_complete(
    payload: ExtractionCompletePayload,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
) -> WebhookResponse:
    """
    Called by the Next.js app when a social media extraction job finishes.

    Fetches the extracted posts from the Next.js internal API, then embeds
    and indexes them into Qdrant so the RAG pipeline can use them for
    personalised content generation.

    Auth: x-api-key header matching ELVZ_PYTHON_ENGINE_API_KEY in .env
    """
    _verify_api_key(x_api_key)

    logger.info(
        "Extraction webhook received",
        extraction_job_id=payload.extractionJobId,
        user_id=payload.userId,
        platform=payload.platform,
        connected_social_platform_id=payload.connectedSocialPlatformId,
    )

    # Fetch posts from the Next.js internal API
    posts = await _fetch_posts(payload.extractionJobId)

    if not posts:
        logger.warning(
            "No posts returned for extraction job",
            extraction_job_id=payload.extractionJobId,
        )
        return WebhookResponse(
            indexed=0,
            skipped=0,
            message="No posts found for this extraction job",
        )

    # Index into Qdrant
    indexed, skipped = await _index_posts(
        user_id=payload.userId,
        platform=payload.platform,
        extraction_job_id=payload.extractionJobId,
        posts=posts,
    )

    logger.info(
        "Extraction webhook processed",
        extraction_job_id=payload.extractionJobId,
        user_id=payload.userId,
        platform=payload.platform,
        indexed=indexed,
        skipped=skipped,
    )

    return WebhookResponse(
        indexed=indexed,
        skipped=skipped,
        message=f"Indexed {indexed} posts from {payload.platform}",
    )


# ---------------------------------------------------------------------------
# Internal API call — fetch posts from Next.js
# ---------------------------------------------------------------------------

async def _fetch_posts(extraction_job_id: str) -> list[ExtractedPost]:
    """
    Call GET /api/internal/posts?extractionJobId={id} on the Next.js app
    and return the list of extracted posts.
    """
    url = f"{settings.elvz_nextjs_base_url}/api/internal/posts"
    headers = {"x-api-key": settings.elvz_api_key}
    params = {"extractionJobId": extraction_job_id}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(
            "Next.js internal API returned an error",
            extraction_job_id=extraction_job_id,
            status_code=e.response.status_code,
            body=e.response.text[:500],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch posts from Next.js API: HTTP {e.response.status_code}",
        )
    except httpx.RequestError as e:
        logger.error(
            "Failed to connect to Next.js app",
            extraction_job_id=extraction_job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach Next.js app: {str(e)}",
        )

    body = response.json()
    raw_posts = (body.get("data") or [])

    posts = []
    for raw in raw_posts:
        try:
            posts.append(ExtractedPost(**raw))
        except Exception as e:
            logger.warning("Failed to parse post from API response", error=str(e), raw=raw)

    logger.info(
        "Fetched posts from Next.js API",
        extraction_job_id=extraction_job_id,
        post_count=len(posts),
    )
    return posts


# ---------------------------------------------------------------------------
# Indexing — embed and upsert into Qdrant
# ---------------------------------------------------------------------------

async def _index_posts(
    user_id: str,
    platform: str,
    extraction_job_id: str,
    posts: list[ExtractedPost],
) -> tuple[int, int]:
    """
    Convert extracted posts to VectorDocuments and upsert into Qdrant.

    Uses sha256(user_id:platform:platformPostId) as the document ID so
    re-indexing the same post is idempotent (upsert, not insert).

    Returns (indexed_count, skipped_count).
    """
    documents: list[VectorDocument] = []
    skipped = 0

    for post in posts:
        text = (post.caption or "").strip()
        if not text:
            skipped += 1
            continue

        performance_score = _compute_performance_score(post.engagement)
        doc_id = _make_doc_id(user_id, platform, post.platformPostId)

        # Determine modality from post type and attached media
        modality = "text"
        if post.media:
            media_types = {m.type for m in post.media}
            if "video" in media_types:
                modality = "video"
            elif "image" in media_types:
                modality = "image"

        doc = VectorDocument(
            id=doc_id,
            content=text,
            metadata={
                "modality": modality,
                "content_type": "user_history",
                "user_id": user_id,
                "platform": platform,
                "category": "scraped_content",
                # Post identifiers
                "post_id": post.id,
                "platform_post_id": post.platformPostId,
                "extraction_job_id": extraction_job_id,
                "connected_social_platform_id": post.connectedSocialPlatformId,
                # Content metadata
                "post_type": post.type,
                "url": post.url or "",
                "posted_at": post.postedAt or "",
                "hashtags": post.hashtags[:10],  # cap to keep metadata lean
                "mentions": post.mentions[:10],
                # Engagement
                "engagement": post.engagement.model_dump(),
                "performance_score": performance_score,
            },
        )
        documents.append(doc)

    if not documents:
        return 0, skipped

    try:
        await vector_store.add_social_content(user_id, documents)
        return len(documents), skipped
    except Exception as e:
        logger.error(
            "Failed to index extracted posts into Qdrant",
            user_id=user_id,
            platform=platform,
            extraction_job_id=extraction_job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index posts: {str(e)}",
        )


def _make_doc_id(user_id: str, platform: str, platform_post_id: str) -> str:
    """
    Deterministic document ID so the same post is never duplicated in Qdrant
    even if the webhook fires multiple times for the same job.

    Format: sha256(user_id:platform:platformPostId)[:32]
    """
    raw = f"{user_id}:{platform}:{platform_post_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _compute_performance_score(engagement: PostEngagement) -> float:
    """
    Normalise engagement metrics to a 0-1 score used for RAG ranking.

    Higher-performing posts are surfaced first as style references.
    Weights: saves (5) > shares (4) > comments (3) > likes (1) > views (0.01)
    """
    total = (
        engagement.likes
        + engagement.comments * 3
        + engagement.shares * 4
        + engagement.saves * 5
        + int(engagement.views * 0.01)
    )

    # If engagementRate is already provided, blend it in
    if engagement.engagementRate is not None:
        rate_score = min(engagement.engagementRate, 1.0)
        raw_score = min(total / 300, 1.0)
        return round((raw_score + rate_score) / 2, 4)

    return round(min(total / 300, 1.0), 4)
