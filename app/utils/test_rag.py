"""
RAG Test Script - Debug Pinecone retrieval and LLM generation.

Usage:
    # Basic search
    python -m app.utils.test_rag --query "best practices for LinkedIn posts"
    
    # With user_id filter
    python -m app.utils.test_rag --query "AI trends" --user-id "test-user-123"
    
    # With platform filter
    python -m app.utils.test_rag --query "engagement tips" --platform "linkedin"
    
    # Full RAG (retrieval + generation)
    python -m app.utils.test_rag --query "How to write engaging LinkedIn posts about AI" --rag
    
    # Interactive mode
    python -m app.utils.test_rag --interactive
"""

import argparse
import asyncio
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, "/Users/krishnayadav/Documents/test_projects/elvz-backend")

try:
    import google.generativeai as genai
    from pinecone import Pinecone
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install google-generativeai pinecone-client")
    sys.exit(1)

from app.core.config import settings


# ============================================
# Pinecone Setup
# ============================================

def init_pinecone():
    """Initialize Pinecone client."""
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")
    
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    print(f"✅ Connected to Pinecone index: {settings.pinecone_index_name}")
    return index


def get_embedding(text: str) -> list[float]:
    """Generate embedding using Gemini."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env")
    
    genai.configure(api_key=settings.google_api_key)
    
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"  # Use query type for searching
    )
    return result['embedding']


# ============================================
# Search Functions
# ============================================

def search_pinecone(
    index,
    query: str,
    user_id: Optional[str] = None,
    platform: Optional[str] = None,
    category: Optional[str] = None,
    namespace: str = "knowledge_base",
    top_k: int = 5,
) -> list[dict]:
    """Search Pinecone for similar documents."""
    
    print("\n" + "="*70)
    print("🔍 PINECONE SEARCH")
    print("="*70)
    print(f"   Query: {query}")
    print(f"   Namespace: {namespace}")
    print(f"   Top K: {top_k}")
    
    # Build filter
    filters = {}
    if user_id:
        filters["user_id"] = {"$in": [user_id, "system"]}
    if platform:
        filters["platform"] = platform
    if category:
        filters["category"] = category
    
    print(f"   Filters: {filters if filters else 'None'}")
    print("-"*70)
    
    # Generate query embedding
    print("   Generating embedding...")
    query_embedding = get_embedding(query)
    print(f"   Embedding dimension: {len(query_embedding)}")
    
    # Search
    print("   Searching Pinecone...")
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filters if filters else None
    )
    
    print(f"   Raw matches: {len(results.matches)}")
    print("="*70)
    
    # Format results
    formatted_results = []
    for match in results.matches:
        metadata = match.metadata or {}
        formatted_results.append({
            "id": match.id,
            "score": match.score,
            "content": metadata.get("text", metadata.get("content", "")),
            "metadata": {k: v for k, v in metadata.items() if k not in ["text", "content"]}
        })
    
    return formatted_results


def print_results(results: list[dict]):
    """Pretty print search results."""
    if not results:
        print("\n❌ No results found!")
        return
    
    print("\n" + "="*70)
    print(f"📄 RETRIEVED CHUNKS ({len(results)} results)")
    print("="*70)
    
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f}")
        print(f"    ID: {r['id'][:40]}...")
        print(f"    Content:")
        
        # Word wrap content
        content = r['content']
        lines = [content[j:j+65] for j in range(0, len(content), 65)]
        for line in lines[:4]:  # Show first 4 lines
            print(f"        {line}")
        if len(lines) > 4:
            print(f"        ... ({len(content)} chars total)")
        
        print(f"    Metadata: {r['metadata']}")
    
    print("\n" + "="*70)


# ============================================
# RAG Generation
# ============================================

def generate_rag_response(query: str, context_chunks: list[dict]) -> str:
    """Generate response using RAG (Retrieval-Augmented Generation)."""
    
    if not context_chunks:
        return "No context available for generation."
    
    # Build context from chunks
    context_text = "\n\n".join([
        f"[Source {i+1}]: {chunk['content']}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    # RAG prompt
    rag_prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so but still try to provide a helpful response.

## Context (Retrieved from Knowledge Base):
{context_text}

## User Question:
{query}

## Instructions:
1. Use the context to inform your answer
2. Be specific and cite relevant information from the context
3. If the context is helpful, mention it
4. Keep the response concise but informative

## Answer:"""

    print("\n" + "="*70)
    print("🤖 RAG GENERATION")
    print("="*70)
    print(f"   Context chunks used: {len(context_chunks)}")
    print(f"   Total context length: {len(context_text)} chars")
    print("-"*70)
    print("   Generating response with Gemini...")
    
    # Generate with Gemini
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.google_model_primary)
    
    response = model.generate_content(rag_prompt)
    
    print("="*70)
    
    return response.text


def print_rag_response(response: str):
    """Pretty print RAG response."""
    print("\n" + "="*70)
    print("💬 RAG RESPONSE")
    print("="*70)
    print(response)
    print("="*70 + "\n")


# ============================================
# Interactive Mode
# ============================================

async def interactive_mode(index):
    """Run interactive RAG testing."""
    print("\n" + "="*70)
    print("🎯 INTERACTIVE RAG TEST MODE")
    print("="*70)
    print("Commands:")
    print("   /quit - Exit")
    print("   /user <id> - Set user_id filter")
    print("   /platform <name> - Set platform filter")
    print("   /clear - Clear filters")
    print("   /rag - Toggle RAG generation (on/off)")
    print("   <query> - Search with current filters")
    print("="*70 + "\n")
    
    user_id = None
    platform = None
    rag_enabled = True
    
    while True:
        try:
            query = input(f"\n🔎 Query [user:{user_id or 'all'}, platform:{platform or 'all'}, rag:{rag_enabled}]: ").strip()
            
            if not query:
                continue
            
            if query.lower() == "/quit":
                print("Goodbye! 👋")
                break
            
            if query.startswith("/user "):
                user_id = query.split(" ", 1)[1].strip() or None
                print(f"   ✓ User ID set to: {user_id}")
                continue
            
            if query.startswith("/platform "):
                platform = query.split(" ", 1)[1].strip() or None
                print(f"   ✓ Platform set to: {platform}")
                continue
            
            if query == "/clear":
                user_id = None
                platform = None
                print("   ✓ Filters cleared")
                continue
            
            if query == "/rag":
                rag_enabled = not rag_enabled
                print(f"   ✓ RAG generation: {'ON' if rag_enabled else 'OFF'}")
                continue
            
            # Search
            results = search_pinecone(
                index=index,
                query=query,
                user_id=user_id,
                platform=platform,
            )
            print_results(results)
            
            # RAG generation
            if rag_enabled and results:
                response = generate_rag_response(query, results)
                print_rag_response(response)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================
# Main CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Test RAG with Pinecone + Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.utils.test_rag --query "LinkedIn best practices"
  python -m app.utils.test_rag --query "AI trends" --user-id "user123" --rag
  python -m app.utils.test_rag --interactive
        """
    )
    
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--user-id", "-u", type=str, help="Filter by user ID")
    parser.add_argument("--platform", "-p", type=str, help="Filter by platform")
    parser.add_argument("--category", "-c", type=str, help="Filter by category")
    parser.add_argument("--namespace", "-n", type=str, default="knowledge_base", help="Pinecone namespace")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--rag", action="store_true", help="Enable RAG generation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Validate
    if not args.query and not args.interactive:
        parser.print_help()
        print("\n❌ Error: Either --query or --interactive is required")
        sys.exit(1)
    
    # Initialize
    print("\n🚀 Initializing RAG Test...")
    index = init_pinecone()
    
    if args.interactive:
        asyncio.run(interactive_mode(index))
    else:
        # Search
        results = search_pinecone(
            index=index,
            query=args.query,
            user_id=args.user_id,
            platform=args.platform,
            category=args.category,
            namespace=args.namespace,
            top_k=args.top_k,
        )
        print_results(results)
        
        # RAG generation
        if args.rag and results:
            response = generate_rag_response(args.query, results)
            print_rag_response(response)
        elif args.rag and not results:
            print("\n⚠️  No results to generate from. Skipping RAG.")


if __name__ == "__main__":
    main()

