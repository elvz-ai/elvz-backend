"""
AWS S3 storage utility for uploading images and videos.
"""

import asyncio
import base64
import uuid
from typing import Optional

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings

logger = structlog.get_logger(__name__)

BUCKET = "elvz-social-platform-assets"
IMAGES_FOLDER = "elvz-images"
VIDEOS_FOLDER = "elvz-videos"


def _get_client():
    """Create a boto3 S3 client using configured credentials."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )


def _public_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"


def _upload_bytes_sync(
    data: bytes,
    bucket: str,
    key: str,
    content_type: str,
) -> Optional[str]:
    """Synchronous upload — called via asyncio.to_thread."""
    try:
        client = _get_client()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        url = _public_url(bucket, key)
        logger.info("S3 upload complete", bucket=bucket, key=key)
        return url
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 upload failed", bucket=bucket, key=key, error=str(e))
        return None


async def upload_base64_image(
    base64_data: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a base64-encoded image to the elvz-images S3 bucket.

    Args:
        base64_data: Base64-encoded image data (with or without data URI prefix)
        filename: Optional custom filename (auto-generated UUID if omitted)
        folder: Optional folder/prefix inside the bucket

    Returns:
        Public URL of the uploaded image, or None on failure
    """
    # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        logger.error("Failed to decode base64 image", error=str(e))
        return None

    filename = filename or f"{uuid.uuid4()}.png"
    key = f"{IMAGES_FOLDER}/{filename}"

    return await asyncio.to_thread(
        _upload_bytes_sync, image_bytes, BUCKET, key, "image/png"
    )


async def upload_image_from_url(
    image_url: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Download an image from an external URL and re-upload it to the elvz-images folder in S3.

    Args:
        image_url: HTTP/HTTPS URL of the source image
        filename: Optional custom filename (auto-generated UUID if omitted)

    Returns:
        Public S3 URL of the uploaded image, or None on failure
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
            content_type = response.headers.get("content-type", "image/png").split(";")[0]
    except Exception as e:
        logger.error("Failed to download image from URL", url=image_url, error=str(e))
        return None

    ext = content_type.split("/")[-1] or "png"
    filename = filename or f"{uuid.uuid4()}.{ext}"
    key = f"{IMAGES_FOLDER}/{filename}"

    return await asyncio.to_thread(
        _upload_bytes_sync, image_bytes, BUCKET, key, content_type
    )


async def upload_base64_video(
    base64_data: str,
    filename: Optional[str] = None,
    content_type: str = "video/mp4",
) -> Optional[str]:
    """
    Upload a base64-encoded video to the elvz-videos S3 bucket.

    Args:
        base64_data: Base64-encoded video data (with or without data URI prefix)
        filename: Optional custom filename (auto-generated UUID if omitted)
        folder: Optional folder/prefix inside the bucket
        content_type: MIME type of the video (default: video/mp4)

    Returns:
        Public URL of the uploaded video, or None on failure
    """
    if base64_data.startswith("data:"):
        base64_data = base64_data.split(",", 1)[1]

    try:
        video_bytes = base64.b64decode(base64_data)
    except Exception as e:
        logger.error("Failed to decode base64 video", error=str(e))
        return None

    ext = content_type.split("/")[-1]
    filename = filename or f"{uuid.uuid4()}.{ext}"
    key = f"{VIDEOS_FOLDER}/{filename}"

    return await asyncio.to_thread(
        _upload_bytes_sync, video_bytes, BUCKET, key, content_type
    )
