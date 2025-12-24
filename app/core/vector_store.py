"""
Vector store implementation for RAG operations.
Supports Pinecone and Weaviate backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.config import settings

logger = structlog.get_logger(__name__)


class VectorDocument(BaseModel):
    """Document to be stored in vector database."""
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: Optional[list[float]] = None


class VectorSearchResult(BaseModel):
    """Search result from vector database."""
    id: str
    content: str
    metadata: dict[str, Any]
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def upsert(self, documents: list[VectorDocument], namespace: str = "") -> None:
        """Insert or update documents."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete(self, ids: list[str], namespace: str = "") -> None:
        """Delete documents by ID."""
        pass


class EmbeddingService:
    """Service for generating embeddings using Gemini (Google) or OpenAI."""
    
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            self.model = "models/embedding-001"
            self.dimension = 768
        else:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.model = "text-embedding-3-small"
            self.dimension = 1536
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if self.provider == "gemini":
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        else:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if self.provider == "gemini":
            import google.generativeai as genai
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        else:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self):
        self._index = None
        self._pc = None
        # Use Gemini for embeddings by default
        self.embedding_service = EmbeddingService(provider="gemini")
    
    def _ensure_connected(self) -> None:
        """Lazily initialize Pinecone connection on first use."""
        if self._index is None:
            try:
                from pinecone import Pinecone
                
                if not settings.pinecone_api_key:
                    raise ValueError("PINECONE_API_KEY not configured")
                
                self._pc = Pinecone(api_key=settings.pinecone_api_key)
                self._index = self._pc.Index(settings.pinecone_index_name)
                logger.info("Pinecone connected", index=settings.pinecone_index_name)
            except Exception as e:
                logger.error("Pinecone connection failed", error=str(e))
                raise
    
    async def connect(self) -> None:
        """Initialize Pinecone connection (for explicit connection)."""
        self._ensure_connected()
    
    @property
    def index(self):
        """Get Pinecone index, auto-connecting if needed."""
        self._ensure_connected()
        return self._index
    
    async def upsert(self, documents: list[VectorDocument], namespace: str = "") -> None:
        """Insert or update documents in Pinecone."""
        # Generate embeddings for documents without them
        texts_to_embed = []
        indices_to_embed = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            embeddings = await self.embedding_service.embed_batch(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, embeddings):
                documents[idx].embedding = embedding
        
        # Prepare vectors for upsert
        vectors = [
            {
                "id": doc.id,
                "values": doc.embedding,
                "metadata": {**doc.metadata, "content": doc.content},
            }
            for doc in documents
        ]
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logger.info("Pinecone upsert complete", count=len(documents), namespace=namespace)
    
    async def search(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Search Pinecone for similar documents."""
        # Debug: Print search parameters
        print(f"\n🔍 PINECONE SEARCH:")
        print(f"   Query: {query[:80]}...")
        print(f"   Namespace: {namespace}")
        print(f"   Top K: {top_k}")
        print(f"   Filters: {filter_metadata}")
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)
        
        # Build query parameters
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        
        if filter_metadata:
            query_params["filter"] = filter_metadata
        
        # Execute query
        results = self.index.query(**query_params)
        
        # Debug: Print raw results count
        print(f"   Raw matches: {len(results.matches)}")
        
        # Convert to search results
        search_results = []
        for match in results.matches:
            metadata = match.metadata or {}
            # Check both 'content' and 'text' keys (add_embeddings.py uses 'text')
            content = metadata.pop("content", "") or metadata.get("text", "")
            search_results.append(
                VectorSearchResult(
                    id=match.id,
                    content=content,
                    metadata=metadata,
                    score=match.score,
                )
            )
        
        logger.debug("Pinecone search complete", query=query[:50], results=len(search_results))
        return search_results
    
    async def delete(self, ids: list[str], namespace: str = "") -> None:
        """Delete documents from Pinecone."""
        self.index.delete(ids=ids, namespace=namespace)
        logger.info("Pinecone delete complete", count=len(ids), namespace=namespace)


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(self):
        self._client = None
        self.embedding_service = EmbeddingService()
    
    async def connect(self) -> None:
        """Initialize Weaviate connection."""
        try:
            import weaviate
            
            if settings.weaviate_api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=settings.weaviate_api_key)
                self._client = weaviate.Client(
                    url=settings.weaviate_url,
                    auth_client_secret=auth_config,
                )
            else:
                self._client = weaviate.Client(url=settings.weaviate_url)
            
            logger.info("Weaviate connected", url=settings.weaviate_url)
        except Exception as e:
            logger.error("Weaviate connection failed", error=str(e))
            raise
    
    @property
    def client(self):
        """Get Weaviate client."""
        if self._client is None:
            raise RuntimeError("Weaviate not connected. Call connect() first.")
        return self._client
    
    async def upsert(self, documents: list[VectorDocument], namespace: str = "") -> None:
        """Insert or update documents in Weaviate."""
        class_name = namespace.title().replace("_", "") or "Document"
        
        # Ensure class exists
        if not self.client.schema.exists(class_name):
            class_schema = {
                "class": class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "doc_id", "dataType": ["string"]},
                ],
            }
            self.client.schema.create_class(class_schema)
        
        # Generate embeddings
        texts_to_embed = []
        indices_to_embed = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            embeddings = await self.embedding_service.embed_batch(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, embeddings):
                documents[idx].embedding = embedding
        
        # Batch import
        with self.client.batch as batch:
            for doc in documents:
                batch.add_data_object(
                    data_object={
                        "content": doc.content,
                        "doc_id": doc.id,
                        **doc.metadata,
                    },
                    class_name=class_name,
                    vector=doc.embedding,
                )
        
        logger.info("Weaviate upsert complete", count=len(documents), class_name=class_name)
    
    async def search(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """Search Weaviate for similar documents."""
        class_name = namespace.title().replace("_", "") or "Document"
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)
        
        # Build query
        query_builder = (
            self.client.query
            .get(class_name, ["content", "doc_id"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(top_k)
            .with_additional(["certainty"])
        )
        
        # Execute query
        results = query_builder.do()
        
        # Convert to search results
        search_results = []
        data = results.get("data", {}).get("Get", {}).get(class_name, [])
        
        for item in data:
            additional = item.pop("_additional", {})
            content = item.pop("content", "")
            doc_id = item.pop("doc_id", "")
            
            search_results.append(
                VectorSearchResult(
                    id=doc_id,
                    content=content,
                    metadata=item,
                    score=additional.get("certainty", 0.0),
                )
            )
        
        logger.debug("Weaviate search complete", query=query[:50], results=len(search_results))
        return search_results
    
    async def delete(self, ids: list[str], namespace: str = "") -> None:
        """Delete documents from Weaviate."""
        class_name = namespace.title().replace("_", "") or "Document"
        
        for doc_id in ids:
            self.client.data_object.delete(
                uuid=doc_id,
                class_name=class_name,
            )
        
        logger.info("Weaviate delete complete", count=len(ids), class_name=class_name)


class VectorStore:
    """
    Unified vector store interface.
    
    Indexes:
    - knowledge_base: Best practices, case studies, frameworks
    - content_examples: High-performing social posts, blogs, ads
    - user_content: Per-user content history
    """
    
    def __init__(self, backend: str = "pinecone"):
        if backend == "pinecone":
            self._store: BaseVectorStore = PineconeVectorStore()
        else:
            self._store: BaseVectorStore = WeaviateVectorStore()
        self.backend = backend
    
    async def connect(self) -> None:
        """Initialize vector store connection."""
        await self._store.connect()
    
    # Knowledge Base Operations
    async def add_knowledge(
        self,
        documents: list[VectorDocument],
        category: str = "best_practices",
    ) -> None:
        """Add documents to knowledge base."""
        for doc in documents:
            doc.metadata["category"] = category
        await self._store.upsert(documents, namespace="knowledge_base")
    
    async def search_knowledge(
        self,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        platform: Optional[str] = None,
        industry: Optional[str] = None,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search knowledge base with optional user_id filtering."""
        filters = {}
        if user_id:
            # Search user-specific knowledge first, then fall back to system
            filters["user_id"] = {"$in": [user_id, "system"]}
        if category:
            filters["category"] = category
        if platform:
            filters["platform"] = platform
        if industry:
            filters["industry"] = industry
        
        return await self._store.search(
            query=query,
            namespace="knowledge_base",
            top_k=top_k,
            filter_metadata=filters if filters else None,
        )
    
    # Content Examples Operations
    async def add_content_examples(
        self,
        documents: list[VectorDocument],
        content_type: str = "social_posts",
    ) -> None:
        """Add content examples."""
        for doc in documents:
            doc.metadata["content_type"] = content_type
        await self._store.upsert(documents, namespace="content_examples")
    
    async def search_content_examples(
        self,
        query: str,
        user_id: Optional[str] = None,
        content_type: Optional[str] = None,
        platform: Optional[str] = None,
        min_performance: Optional[float] = None,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search content examples with optional user_id filtering."""
        filters = {}
        if user_id:
            # Search user-specific examples first, then fall back to system
            filters["user_id"] = {"$in": [user_id, "system"]}
        if content_type:
            filters["content_type"] = content_type
        if platform:
            filters["platform"] = platform
        
        results = await self._store.search(
            query=query,
            namespace="knowledge_base",  # Use unified knowledge_base namespace
            top_k=top_k * 2,  # Get more for filtering
            filter_metadata=filters if filters else None,
        )
        
        # Filter by performance if specified
        if min_performance is not None:
            results = [
                r for r in results
                if r.metadata.get("performance_score", 0) >= min_performance
            ]
        
        return results[:top_k]
    
    # User Content Operations
    async def add_user_content(
        self,
        user_id: str,
        documents: list[VectorDocument],
    ) -> None:
        """Add user's content for voice profile."""
        namespace = f"user_{user_id}"
        await self._store.upsert(documents, namespace=namespace)
    
    async def search_user_content(
        self,
        user_id: str,
        query: str,
        content_type: Optional[str] = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """Search user's content history."""
        namespace = f"user_{user_id}"
        filters = {"content_type": content_type} if content_type else None
        
        return await self._store.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            filter_metadata=filters,
        )
    
    async def delete_user_content(self, user_id: str, doc_ids: list[str]) -> None:
        """Delete user's content."""
        namespace = f"user_{user_id}"
        await self._store.delete(doc_ids, namespace=namespace)


# Global vector store instance
vector_store = VectorStore()

