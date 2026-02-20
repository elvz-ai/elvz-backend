#!/usr/bin/env python3
"""
Seed social_memory_test Qdrant collection from a JSON file.

Mirrors the exact VectorDocument format and metadata used by the
extraction webhook (/api/v1/webhook/extraction-complete).

Usage:
    python scripts/seed_social_memory.py posts.json \
        --user-id <userId> \
        --platform linkedin \
        [--extraction-job-id <id>] \
        [--connected-social-platform-id <id>]

JSON file formats accepted:
    1. Next.js internal API response:   {"data": [...posts...]}
    2. Flat post list:                  [...posts...]
    3. Single post object:              {...post...}

Each post object should match the ExtractedPost schema:
    {
        "id": "...",
        "extractionJobId": "...",
        "connectedSocialPlatformId": "...",
        "platformPostId": "...",
        "type": "text|image|video|carousel",
        "caption": "...",
        "url": "...",
        "postedAt": "...",
        "hashtags": [...],
        "mentions": [...],
        "media": [...],
        "engagement": {
            "likes": 0, "comments": 0, "shares": 0,
            "saves": 0, "views": 0, "engagementRate": null
        }
    }
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

# Add project root to path so app.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_doc_id(user_id: str, platform: str, platform_post_id: str) -> str:
    """Deterministic ID — identical to the webhook's _make_doc_id()."""
    raw = f"{user_id}:{platform}:{platform_post_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _compute_performance_score(engagement: dict) -> float:
    """Identical to the webhook's _compute_performance_score()."""
    likes = engagement.get("likes", 0)
    comments = engagement.get("comments", 0)
    shares = engagement.get("shares", 0)
    saves = engagement.get("saves", 0)
    views = engagement.get("views", 0)
    engagement_rate = engagement.get("engagementRate")

    total = (
        likes
        + comments * 3
        + shares * 4
        + saves * 5
        + int(views * 0.01)
    )

    if engagement_rate is not None:
        rate_score = min(float(engagement_rate), 1.0)
        raw_score = min(total / 300, 1.0)
        return round((raw_score + rate_score) / 2, 4)

    return round(min(total / 300, 1.0), 4)


def _parse_posts(data: object) -> list[dict]:
    """Accept all three JSON shapes: {data: [...]}, [...], or {...}."""
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unexpected JSON structure: {type(data)}")


async def seed(
    json_path: str,
    user_id: str,
    platform: str,
    extraction_job_id: str,
    connected_social_platform_id: str,
) -> None:
    # Import here so the path patch above takes effect first
    from app.core.vector_store import VectorDocument, vector_store

    # Load JSON
    path = Path(json_path)
    if not path.exists():
        print(f"ERROR: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with path.open() as f:
        raw_data = json.load(f)

    raw_posts = _parse_posts(raw_data)
    print(f"Loaded {len(raw_posts)} post(s) from {path.name}")

    documents: list[VectorDocument] = []
    skipped = 0

    for raw in raw_posts:
        text = (raw.get("caption") or "").strip()
        if not text:
            skipped += 1
            print(f"  [SKIP] post '{raw.get('id', '?')}' — empty caption")
            continue

        # Resolve IDs: prefer values from JSON, fall back to CLI args
        post_extraction_job_id = raw.get("extractionJobId") or extraction_job_id
        post_connected_id = raw.get("connectedSocialPlatformId") or connected_social_platform_id
        platform_post_id = raw.get("platformPostId") or raw.get("id") or ""

        engagement = raw.get("engagement") or {}
        performance_score = _compute_performance_score(engagement)
        doc_id = _make_doc_id(user_id, platform, platform_post_id)

        # Determine modality from media list
        modality = "text"
        media = raw.get("media") or []
        if media:
            media_types = {m.get("type") for m in media}
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
                "post_id": raw.get("id", ""),
                "platform_post_id": platform_post_id,
                "extraction_job_id": post_extraction_job_id,
                "connected_social_platform_id": post_connected_id,
                # Content metadata
                "post_type": raw.get("type", "text"),
                "url": raw.get("url") or "",
                "posted_at": raw.get("postedAt") or "",
                "hashtags": (raw.get("hashtags") or [])[:10],
                "mentions": (raw.get("mentions") or [])[:10],
                # Engagement
                "engagement": engagement,
                "performance_score": performance_score,
            },
        )
        documents.append(doc)
        print(f"  [OK]   post '{platform_post_id}' — modality={modality}, score={performance_score}")

    if not documents:
        print(f"\nNothing to index ({skipped} skipped — all had empty captions).")
        return

    print(f"\nIndexing {len(documents)} document(s) into social_memory_test ...")
    await vector_store.connect()
    await vector_store.add_social_content(user_id, documents)
    print(f"Done. Indexed={len(documents)}, Skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed social_memory_test Qdrant collection from a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_file", help="Path to the JSON file containing extracted posts")
    parser.add_argument("--user-id", required=True, help="User ID to associate the posts with")
    parser.add_argument("--platform", required=True, help="Platform (linkedin, instagram, twitter, etc.)")
    parser.add_argument(
        "--extraction-job-id",
        default="manual-seed",
        help="Extraction job ID (default: 'manual-seed')",
    )
    parser.add_argument(
        "--connected-social-platform-id",
        default="",
        help="connectedSocialPlatformId (optional)",
    )

    args = parser.parse_args()

    asyncio.run(
        seed(
            json_path=args.json_file,
            user_id=args.user_id,
            platform=args.platform,
            extraction_job_id=args.extraction_job_id,
            connected_social_platform_id=args.connected_social_platform_id,
        )
    )


if __name__ == "__main__":
    main()
