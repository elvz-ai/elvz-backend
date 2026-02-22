"""
Setup Qdrant collection for Elvz long-term memory.

Usage:
    python -m app.utils.setup_qdrant --setup          # Create collection + indexes
    python -m app.utils.setup_qdrant --seed            # Seed knowledge base
    python -m app.utils.setup_qdrant --status          # Check collection stats
    python -m app.utils.setup_qdrant --test "query"    # Test search
"""

import argparse
import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

# Load environment variables
load_dotenv()

# Settings class for basic setup (no app dependencies)
class Settings:
    """Minimal settings for setup utility."""

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "elvz_memory")
    qdrant_vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
    qdrant_distance = os.getenv("QDRANT_DISTANCE", "Cosine")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")


settings = Settings()


# For seeding, import the real EmbeddingService from app
def get_embedding_service():
    """Get embedding service, importing from app when needed."""
    try:
        from app.core.vector_store import EmbeddingService
        return EmbeddingService(provider="gemini")
    except Exception as e:
        # Fallback for standalone usage
        import google.generativeai as genai

        class SimpleEmbeddingService:
            def __init__(self):
                genai.configure(api_key=settings.google_api_key)
                self.model = "models/gemini-embedding-001"

            async def embed(self, text: str) -> list[float]:
                result = genai.embed_content(
                    model=self.model, content=text, task_type="retrieval_query"
                )
                return result["embedding"]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model=self.model, content=text, task_type="retrieval_document"
                    )
                    embeddings.append(result["embedding"])
                return embeddings

        return SimpleEmbeddingService()


class VectorDocument:
    """Simple document model for setup."""

    def __init__(self, id: str, content: str, metadata: dict):
        self.id = id
        self.content = content
        self.metadata = metadata


class QdrantSetup:
    """Qdrant collection setup and utilities."""

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_service = get_embedding_service()

    def setup_collection(self):
        """Create collection and all indexes."""
        print(f"\n🔧 Setting up Qdrant collection: {self.collection_name}")
        print(f"   URL: {settings.qdrant_url}")

        # Check if exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            print(f"   ⚠️  Collection already exists. Recreating...")
            self.client.delete_collection(self.collection_name)

        # Create collection
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=distance_map.get(settings.qdrant_distance, Distance.COSINE),
            ),
        )
        print(
            f"   ✅ Collection created: {settings.qdrant_vector_size}D, {settings.qdrant_distance}"
        )

        # Create payload indexes
        print("\n📇 Creating payload indexes...")
        keyword_fields = ["modality", "content_type", "user_id", "platform", "category"]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"   ✅ Index created: {field} (keyword)")

        # DateTime index
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="created_at",
            field_schema=PayloadSchemaType.DATETIME,
        )
        print("   ✅ Index created: created_at (datetime)")

        # Full-text index
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="content",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
        print("   ✅ Index created: content (full-text)")

        print("\n✨ Setup complete!\n")

    async def seed_knowledge_base(self):
        """Seed with social media best practices."""
        print("\n🌱 Seeding knowledge base...")

        knowledge_items = [
            # LinkedIn best practices
            {
                "content": "LinkedIn posts perform best between 7-9 AM on weekdays. Tuesday through Thursday see the highest engagement rates.",
                "platform": "linkedin",
                "category": "best_practices",
            },
            {
                "content": "LinkedIn: Use storytelling format with a hook in the first 2 lines. Keep paragraphs short (1-2 sentences). Add 3-5 relevant hashtags.",
                "platform": "linkedin",
                "category": "best_practices",
            },
            {
                "content": "Professional tone on LinkedIn works best. Share insights, case studies, and lessons learned. Ask thought-provoking questions to boost comments.",
                "platform": "linkedin",
                "category": "best_practices",
            },
            # Instagram best practices
            {
                "content": "Instagram: Post reels at 9 AM, 12 PM, or 7 PM. Carousels get 3x more engagement than single images. Use all 30 hashtags.",
                "platform": "instagram",
                "category": "best_practices",
            },
            {
                "content": "Instagram captions: First 125 characters are visible without 'more' click. Use emojis for visual breaks. Add CTA at the end.",
                "platform": "instagram",
                "category": "best_practices",
            },
            # Facebook best practices
            {
                "content": "Facebook: Videos get 59% more engagement than images. Native videos perform better than YouTube links. Keep videos under 2 minutes.",
                "platform": "facebook",
                "category": "best_practices",
            },
            {
                "content": "Facebook posts: Ask questions to boost comments. Use conversational tone. Respond to all comments within 1 hour for better reach.",
                "platform": "facebook",
                "category": "best_practices",
            },
            # Twitter/X best practices
            {
                "content": "Twitter/X: Threads perform 3x better than single tweets. Start with a hook. Number your thread (1/n format). Add images or GIFs.",
                "platform": "twitter",
                "category": "best_practices",
            },
            {
                "content": "Twitter timing: Tweet at 8 AM, 12 PM, or 5 PM EST. Tuesdays and Wednesdays see highest engagement. Retweet your best content at different times.",
                "platform": "twitter",
                "category": "best_practices",
            },
            # General social media
            {
                "content": "Consistency beats perfection. Post 3-5 times per week minimum. Build a content calendar. Repurpose content across platforms.",
                "platform": "general",
                "category": "best_practices",
            },
            {
                "content": "Use analytics to double down on what works. Track engagement rate, not just followers. A/B test different post formats.",
                "platform": "general",
                "category": "best_practices",
            },
            {
                "content": "Hashtag strategy: Use mix of high-volume (100k+), medium (10k-100k), and niche (<10k) tags. Research competitor hashtags.",
                "platform": "general",
                "category": "best_practices",
            },
        ]

        # Generate documents
        documents = []
        for item in knowledge_items:
            doc = VectorDocument(
                id=str(uuid4()),
                content=item["content"],
                metadata={
                    "modality": "text",
                    "content_type": "knowledge",
                    "user_id": "system",
                    "platform": item["platform"],
                    "category": item["category"],
                    "created_at": "2026-02-14T00:00:00Z",
                },
            )
            documents.append(doc)

        # Batch embed
        texts = [doc.content for doc in documents]
        print(f"   Generating embeddings for {len(texts)} items...")
        embeddings = await self.embedding_service.embed_batch(texts)

        # Create points
        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append(
                PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={**doc.metadata, "content": doc.content},
                )
            )

        # Upsert
        self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"   ✅ Seeded {len(points)} knowledge base items\n")

    def show_status(self):
        """Show collection statistics."""
        print(f"\n📊 Collection Status: {self.collection_name}\n")

        info = self.client.get_collection(self.collection_name)
        print(f"   Vectors: {info.points_count:,}")
        print(f"   Dimension: {info.config.params.vectors.size}")
        print(f"   Distance: {info.config.params.vectors.distance}")

        # Get payload schema
        if hasattr(info.config, "payload_schema"):
            print("\n   Indexed Fields:")
            for field, schema in info.config.payload_schema.items():
                print(f"     - {field}: {schema.data_type}")

        print()

    async def test_search(self, query: str, modality: str = "text"):
        """Test search with a query."""
        print(f"\n🔍 Testing search: '{query}' (modality={modality})\n")

        # Generate query embedding
        embedding = await self.embedding_service.embed(query)

        # Build filter
        filters = Filter(must=[FieldCondition(key="modality", match=MatchValue(value=modality))])

        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=filters,
            limit=5,
            with_payload=True,
        )

        if not results.points:
            print("   No results found.\n")
            return

        print(f"   Found {len(results.points)} results:\n")
        for i, point in enumerate(results.points, 1):
            payload = point.payload or {}
            content = payload.get("content", "")
            platform = payload.get("platform", "unknown")
            score = point.score

            print(f"   {i}. [{platform}] Score: {score:.3f}")
            print(f"      {content[:150]}...")
            print()


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Qdrant setup utility for Elvz")
    parser.add_argument("--setup", action="store_true", help="Create collection and indexes")
    parser.add_argument("--seed", action="store_true", help="Seed knowledge base")
    parser.add_argument("--status", action="store_true", help="Show collection stats")
    parser.add_argument("--test", type=str, help="Test search with query")
    parser.add_argument("--modality", type=str, default="text", help="Modality filter for test")

    args = parser.parse_args()

    if not any([args.setup, args.seed, args.status, args.test]):
        parser.print_help()
        return

    setup = QdrantSetup()

    if args.setup:
        setup.setup_collection()

    if args.seed:
        await setup.seed_knowledge_base()

    if args.status:
        setup.show_status()

    if args.test:
        await setup.test_search(args.test, args.modality)


if __name__ == "__main__":
    asyncio.run(main())
