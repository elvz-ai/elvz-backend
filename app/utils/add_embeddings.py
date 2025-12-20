"""
Script to add embeddings to Pinecone vector database.

Usage (CLI):
    # Add single text with user ID
    python -m app.utils.add_embeddings add --text "Your text here" --user-id "user123" --category "best_practices" --platform "linkedin"
    
    # Add paragraph (split into sentences)
    python -m app.utils.add_embeddings add --text "First sentence. Second sentence. Third sentence." --user-id "user123" --split
    
    # Search similar texts
    python -m app.utils.add_embeddings search --query "best time to post" --user-id "user123"
    
    # Seed knowledge base
    python -m app.utils.add_embeddings seed --user-id "system"

Programmatic Usage:
    from app.utils.add_embeddings import add_embedding, add_bulk_embeddings, add_text_paragraph
    
    # Single embedding with user ID
    await add_embedding(
        text="Use line breaks for readability on LinkedIn",
        user_id="user123",
        metadata={"category": "best_practices", "platform": "linkedin"}
    )
    
    # Add paragraph (as single embedding or split)
    await add_text_paragraph(
        input_text="This is a long paragraph with multiple sentences. Each one has value.",
        user_id="user123",
        category="best_practices",
        platform="linkedin",
        chunk_by_sentences=True  # Split into individual embeddings
    )
    
    # Bulk embeddings
    await add_bulk_embeddings(
        items=[
            {"text": "Hook readers in first line", "category": "best_practices", "platform": "linkedin"},
            {"text": "Use 3-5 hashtags on LinkedIn", "category": "hashtags", "platform": "linkedin"},
        ],
        user_id="user123"
    )
    
    # Search with user filter
    results = await search_similar(
        query="LinkedIn engagement tips",
        user_id="user123",  # Filter by user
        top_k=5
    )
"""

import argparse
import asyncio
import hashlib
import uuid
from typing import Optional

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings


# Initialize Pinecone
pc: Optional[Pinecone] = None
index = None


def init_pinecone():
    """Initialize Pinecone client and index."""
    global pc, index
    
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in .env")
    
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if settings.pinecone_index_name not in existing_indexes:
        print(f"Creating index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=768,  # Gemini embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment
            )
        )
        print(f"Index created: {settings.pinecone_index_name}")
    
    index = pc.Index(settings.pinecone_index_name)
    print(f"Connected to Pinecone index: {settings.pinecone_index_name}")
    return index


def get_embedding(text: str) -> list[float]:
    """Generate embedding using Gemini."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set in .env")
    
    genai.configure(api_key=settings.google_api_key)
    
    # Use Gemini's embedding model
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    
    return result['embedding']


def generate_id(text: str) -> str:
    """Generate a unique ID from text content."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


async def add_embedding(
    text: str,
    user_id: str,
    metadata: Optional[dict] = None,
    namespace: str = "default",
    custom_id: Optional[str] = None,
) -> dict:
    """
    Add a single text embedding to Pinecone.
    
    Args:
        text: The text to embed (can be a paragraph or multiple sentences)
        user_id: User ID for multi-tenancy
        metadata: Additional metadata (category, platform, etc.)
        namespace: Pinecone namespace (for multi-tenancy)
        custom_id: Optional custom ID (auto-generated if not provided)
    
    Returns:
        dict with status and ID
    """
    global index
    
    if index is None:
        init_pinecone()
    
    # Generate embedding
    print(f"Generating embedding for: {text[:50]}...")
    embedding = get_embedding(text)
    
    # Prepare metadata
    meta = {
        "text": text,
        "user_id": user_id,
        **(metadata or {})
    }
    
    # Generate ID (include user_id to ensure uniqueness per user)
    doc_id = custom_id or generate_id(f"{user_id}:{text}")
    
    # Upsert to Pinecone
    index.upsert(
        vectors=[{
            "id": doc_id,
            "values": embedding,
            "metadata": meta
        }],
        namespace=namespace
    )
    
    print(f"✅ Added embedding: {doc_id} (user: {user_id})")
    return {"status": "success", "id": doc_id, "user_id": user_id, "text": text[:50]}


async def add_bulk_embeddings(
    items: list[dict],
    user_id: str,
    namespace: str = "default",
    batch_size: int = 100,
) -> dict:
    """
    Add multiple text embeddings to Pinecone.
    
    Args:
        items: List of dicts with 'text' and optional metadata fields
        user_id: User ID for multi-tenancy
        namespace: Pinecone namespace
        batch_size: Number of vectors per batch
    
    Returns:
        dict with status and count
    """
    global index
    
    if index is None:
        init_pinecone()
    
    vectors = []
    
    for i, item in enumerate(items):
        text = item.get("text", "")
        if not text:
            continue
        
        print(f"Processing {i+1}/{len(items)}: {text[:40]}...")
        
        # Generate embedding
        embedding = get_embedding(text)
        
        # Prepare metadata (exclude 'text' key, add it separately)
        metadata = {k: v for k, v in item.items() if k != "text"}
        metadata["text"] = text
        metadata["user_id"] = user_id
        
        vectors.append({
            "id": generate_id(f"{user_id}:{text}"),
            "values": embedding,
            "metadata": metadata
        })
        
        # Batch upsert
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅ Batch upserted: {len(vectors)} vectors")
            vectors = []
    
    # Upsert remaining
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"✅ Final batch upserted: {len(vectors)} vectors")
    
    total = len(items)
    print(f"\n✅ Total embeddings added: {total} (user: {user_id})")
    return {"status": "success", "count": total, "user_id": user_id}


async def search_similar(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 5,
    namespace: str = "default",
    filter_metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Search for similar texts in Pinecone.
    
    Args:
        query: Search query text
        user_id: Optional user ID to filter results (multi-tenancy)
        top_k: Number of results to return
        namespace: Pinecone namespace
        filter_metadata: Optional additional metadata filter
    
    Returns:
        List of similar texts with scores
    """
    global index
    
    if index is None:
        init_pinecone()
    
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Build filter
    combined_filter = filter_metadata.copy() if filter_metadata else {}
    if user_id:
        combined_filter["user_id"] = user_id
    
    # Search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=combined_filter if combined_filter else None
    )
    
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "user_id": match.metadata.get("user_id", ""),
            "metadata": match.metadata
        }
        for match in results.matches
    ]


async def add_text_paragraph(
    input_text: str,
    user_id: str,
    category: str = "general",
    platform: str = "general",
    namespace: str = "knowledge_base",
    chunk_by_sentences: bool = False,
) -> dict:
    """
    Add a text paragraph to Pinecone.
    
    Args:
        input_text: String paragraph to embed
        user_id: User ID for multi-tenancy
        category: Category (best_practices, content, engagement, etc.)
        platform: Platform (linkedin, twitter, instagram, general)
        namespace: Pinecone namespace
        chunk_by_sentences: If True, split paragraph into sentences
    
    Returns:
        dict with status and IDs
    """
    if chunk_by_sentences:
        # Split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        results = []
        for sentence in sentences:
            result = await add_embedding(
                text=sentence,
                user_id=user_id,
                metadata={"category": category, "platform": platform},
                namespace=namespace
            )
            results.append(result)
        
        return {
            "status": "success",
            "count": len(results),
            "user_id": user_id,
            "ids": [r["id"] for r in results]
        }
    else:
        # Add as single embedding
        result = await add_embedding(
            text=input_text,
            user_id=user_id,
            metadata={"category": category, "platform": platform},
            namespace=namespace
        )
        return result


# ============================================
# Pre-built knowledge base for Social Media
# ============================================

SOCIAL_MEDIA_BEST_PRACTICES = [
    # LinkedIn
    {"text": "Use line breaks for readability on LinkedIn posts. White space makes content scannable.", "category": "best_practices", "platform": "linkedin"},
    {"text": "Hook readers in the first line. LinkedIn shows only 2-3 lines before 'see more'.", "category": "best_practices", "platform": "linkedin"},
    {"text": "Optimal LinkedIn post length is 1300-2000 characters for thought leadership.", "category": "best_practices", "platform": "linkedin"},
    {"text": "Use 3-5 relevant hashtags on LinkedIn. Too many looks spammy.", "category": "hashtags", "platform": "linkedin"},
    {"text": "Post on LinkedIn Tuesday-Thursday, 8-10 AM for B2B engagement.", "category": "timing", "platform": "linkedin"},
    {"text": "Carousels on LinkedIn get 3x more engagement than single images.", "category": "visual", "platform": "linkedin"},
    {"text": "Ask questions at the end of LinkedIn posts to drive comments.", "category": "engagement", "platform": "linkedin"},
    {"text": "Share personal stories on LinkedIn - authenticity drives engagement.", "category": "content", "platform": "linkedin"},
    {"text": "Use professional blue tones in LinkedIn visuals to match platform aesthetics.", "category": "visual", "platform": "linkedin"},
    {"text": "Tag relevant people and companies on LinkedIn to expand reach.", "category": "engagement", "platform": "linkedin"},
    
    # Twitter/X
    {"text": "Keep tweets concise - 71-100 characters get most engagement.", "category": "best_practices", "platform": "twitter"},
    {"text": "Use 1-2 hashtags maximum on Twitter. More reduces engagement.", "category": "hashtags", "platform": "twitter"},
    {"text": "Tweet during lunch hours (12-1 PM) for highest engagement.", "category": "timing", "platform": "twitter"},
    {"text": "Threads work great on Twitter for long-form content.", "category": "content", "platform": "twitter"},
    {"text": "GIFs and short videos get 6x more retweets than photos.", "category": "visual", "platform": "twitter"},
    
    # Instagram
    {"text": "Use 20-30 hashtags on Instagram, mix popular and niche.", "category": "hashtags", "platform": "instagram"},
    {"text": "Instagram Reels get 2x more reach than static posts.", "category": "content", "platform": "instagram"},
    {"text": "Post on Instagram 11 AM - 1 PM and 7-9 PM for best engagement.", "category": "timing", "platform": "instagram"},
    {"text": "Use vertical format (1080x1350) for Instagram feed posts.", "category": "visual", "platform": "instagram"},
    {"text": "Stories with polls and questions get 3x more engagement.", "category": "engagement", "platform": "instagram"},
    
    # General
    {"text": "Consistency is key - post at same times daily for algorithm favor.", "category": "best_practices", "platform": "general"},
    {"text": "Respond to comments within first hour to boost algorithm visibility.", "category": "engagement", "platform": "general"},
    {"text": "Use a hook-story-CTA structure for engaging social posts.", "category": "content", "platform": "general"},
    {"text": "Include faces in images - posts with faces get 38% more engagement.", "category": "visual", "platform": "general"},
    {"text": "End posts with a clear call-to-action to drive engagement.", "category": "content", "platform": "general"},
]


async def seed_knowledge_base(user_id: str = "system"):
    """Seed Pinecone with social media best practices."""
    print(f"\n🌱 Seeding knowledge base with social media best practices (user: {user_id})...\n")
    
    result = await add_bulk_embeddings(
        SOCIAL_MEDIA_BEST_PRACTICES,
        user_id=user_id,
        namespace="knowledge_base"
    )
    
    print(f"\n✅ Knowledge base seeded with {result['count']} entries!")
    return result


# ============================================
# CLI Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Add embeddings to Pinecone")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add single embedding or paragraph
    add_parser = subparsers.add_parser("add", help="Add text embedding (single or paragraph)")
    add_parser.add_argument("--text", "-t", required=True, help="Text/paragraph to embed")
    add_parser.add_argument("--user-id", "-u", required=True, help="User ID (required)")
    add_parser.add_argument("--category", "-c", default="general", help="Category")
    add_parser.add_argument("--platform", "-p", default="general", help="Platform")
    add_parser.add_argument("--namespace", "-n", default="knowledge_base", help="Namespace")
    add_parser.add_argument("--split", "-s", action="store_true", help="Split paragraph into sentences")
    
    # Search
    search_parser = subparsers.add_parser("search", help="Search similar texts")
    search_parser.add_argument("--query", "-q", required=True, help="Search query")
    search_parser.add_argument("--user-id", "-u", help="Filter by user ID")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--platform", "-p", help="Filter by platform")
    search_parser.add_argument("--namespace", "-n", default="knowledge_base", help="Namespace")
    
    # Seed
    seed_parser = subparsers.add_parser("seed", help="Seed knowledge base with best practices")
    seed_parser.add_argument("--user-id", "-u", default="system", help="User ID (default: system)")
    
    args = parser.parse_args()
    
    if args.command == "add":
        if args.split:
            # Split paragraph into sentences
            asyncio.run(add_text_paragraph(
                input_text=args.text,
                user_id=args.user_id,
                category=args.category,
                platform=args.platform,
                namespace=args.namespace,
                chunk_by_sentences=True
            ))
        else:
            asyncio.run(add_embedding(
                text=args.text,
                user_id=args.user_id,
                metadata={"category": args.category, "platform": args.platform},
                namespace=args.namespace
            ))
    
    elif args.command == "search":
        filter_meta = {"platform": args.platform} if args.platform else None
        results = asyncio.run(search_similar(
            query=args.query,
            user_id=args.user_id,
            top_k=args.top_k,
            namespace=args.namespace,
            filter_metadata=filter_meta
        ))
        
        print(f"\n🔍 Search results for: '{args.query}'\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.3f}] {r['text'][:80]}...")
            print(f"   User: {r['user_id']}, Platform: {r['metadata'].get('platform', 'N/A')}, Category: {r['metadata'].get('category', 'N/A')}\n")
    
    elif args.command == "seed":
        asyncio.run(seed_knowledge_base(user_id=args.user_id))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

