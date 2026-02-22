"""
Browse Qdrant collection documents.

Usage:
    python browse_collection.py --limit 10
    python browse_collection.py --user alice123
    python browse_collection.py --platform linkedin
"""
import argparse
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

def browse_collection(
    limit: int = 10,
    offset: int = 0,
    user_id: str = None,
    platform: str = None,
    content_type: str = None,
):
    """Browse documents in the collection."""

    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "elvz_memory")

    # Build filter if specified
    scroll_filter = None
    if user_id or platform or content_type:
        conditions = []
        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if platform:
            conditions.append(FieldCondition(key="platform", match=MatchValue(value=platform)))
        if content_type:
            conditions.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
        scroll_filter = Filter(must=conditions)

    # Scroll through documents
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,  # Don't fetch 3072-dim vectors to save bandwidth
    )

    points, next_offset = results

    print(f"\n📚 Collection: {collection_name}")
    print(f"   Found {len(points)} documents")
    if scroll_filter:
        print(f"   Filters: user_id={user_id}, platform={platform}, content_type={content_type}")
    print(f"   Next offset: {next_offset}\n")

    for i, point in enumerate(points, 1):
        payload = point.payload or {}
        print(f"{'='*80}")
        print(f"Document {i} (ID: {point.id})")
        print(f"{'='*80}")

        # Metadata
        print(f"📋 Metadata:")
        for key in ["user_id", "platform", "content_type", "modality", "category", "created_at"]:
            if key in payload:
                print(f"   {key}: {payload[key]}")

        # Content preview
        content = payload.get("content", "")
        print(f"\n📝 Content ({len(content)} chars):")
        print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")

        # Tags if present
        if "tags" in payload:
            print(f"\n🏷️  Tags: {', '.join(payload['tags'])}")

        print()

    return next_offset


def get_collection_stats():
    """Get collection statistics."""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "elvz_memory")

    info = client.get_collection(collection_name)
    print(f"\n📊 Collection Statistics: {collection_name}")
    print(f"   Total vectors: {info.points_count:,}")
    print(f"   Vector dimension: {info.config.params.vectors.size}")
    print(f"   Distance metric: {info.config.params.vectors.distance}")

    # Count by user_id
    print(f"\n👥 Documents by User:")
    try:
        # Get all points and count by user_id
        all_points = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=["user_id"],
            with_vectors=False,
        )[0]

        user_counts = {}
        for point in all_points:
            user_id = point.payload.get("user_id", "unknown")
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

        for user_id, count in sorted(user_counts.items(), key=lambda x: -x[1]):
            print(f"   {user_id}: {count} documents")
    except Exception as e:
        print(f"   Could not retrieve user stats: {e}")

    # Count by platform
    print(f"\n📱 Documents by Platform:")
    try:
        platform_counts = {}
        for point in all_points:
            platform = point.payload.get("platform", "unknown")
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        for platform, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
            print(f"   {platform}: {count} documents")
    except Exception as e:
        print(f"   Could not retrieve platform stats: {e}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browse Qdrant collection")
    parser.add_argument("--limit", type=int, default=10, help="Number of documents to show")
    parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    parser.add_argument("--user", type=str, help="Filter by user_id")
    parser.add_argument("--platform", type=str, help="Filter by platform")
    parser.add_argument("--content-type", type=str, help="Filter by content_type")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")

    args = parser.parse_args()

    if args.stats:
        get_collection_stats()
    else:
        browse_collection(
            limit=args.limit,
            offset=args.offset,
            user_id=args.user,
            platform=args.platform,
            content_type=args.content_type,
        )
