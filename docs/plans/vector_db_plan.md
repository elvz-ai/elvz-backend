# Qdrant Vector DB — Long-Term Memory Plan

## Overview

Replace Pinecone with **Qdrant** as the primary vector database for long-term memory. Create a **single unified collection** (`elvz_memory`) that stores text, image, audio, and video embeddings with **metadata-based modality filtering** to avoid searching the entire collection on every retrieval.

Additionally, **piggyback modality detection onto the existing intent classification LLM call** (no new LLM call) so the system knows which modalities to search during retrieval.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      User Interaction                                │
│   Text message / Image upload / Audio clip / Video reference         │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Intent Classifier (EXISTING LLM call)                   │
│   ┌────────────────────────────────────────────────────────┐         │
│   │  ENHANCED JSON OUTPUT:                                  │         │
│   │  {                                                      │         │
│   │    "type": "artifact",                                  │         │
│   │    "confidence": 0.9,                                   │         │
│   │    "entities": { "platform": "linkedin", ... },         │         │
│   │    "reasoning": "...",                                  │         │
│   │    "search_modalities": ["text"]  ← NEW FIELD           │         │
│   │  }                                                      │         │
│   └────────────────────────────────────────────────────────┘         │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Memory Retriever Node                                   │
│   Reads state["current_intent"]["search_modalities"]                 │
│   Passes modality filter → RAG Retriever → Qdrant                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Qdrant (Single Collection)                         │
│                                                                      │
│   Collection: "elvz_memory"                                         │
│   Vector size: 768 (Gemini embedding-001)                           │
│   Distance: Cosine                                                   │
│                                                                      │
│   ┌────────────────────────────────────────────────────────┐         │
│   │  Payload (Metadata) Fields:                             │         │
│   │  ─────────────────────────────────────────              │         │
│   │  modality:     "text" | "image" | "audio" | "video"    │  INDEX  │
│   │  content_type: "knowledge" | "example" | "user_history" │  INDEX  │
│   │  user_id:      "user_123" | "system"                    │  INDEX  │
│   │  platform:     "linkedin" | "instagram" | ...           │  INDEX  │
│   │  category:     "best_practices" | "social_posts" | ...  │  INDEX  │
│   │  created_at:   "2026-02-14T..."                         │  INDEX  │
│   │  content:      "The actual text content..."             │         │
│   │  source_url:   "https://..."  (for images/videos)       │         │
│   │  description:  "Text desc of non-text content"          │         │
│   │  engagement:   { likes: 100, shares: 20, ... }          │         │
│   │  tags:         ["ai", "blockchain", ...]                │         │
│   └────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Qdrant Credentials

```
QDRANT_URL=http://34.217.75.65:6333
QDRANT_API_KEY=Iz0ulVGF3zqHIiZsTOQFnbI7M69n21E3
```

---

## Phase 1: Config & Collection Setup

### 1.1 Add Qdrant Settings to `app/core/config.py`

```python
# Vector Database - Qdrant
qdrant_url: str = "http://localhost:6333"
qdrant_api_key: str = ""
qdrant_collection_name: str = "elvz_memory"
qdrant_vector_size: int = 768  # Gemini embedding-001 dimension
qdrant_distance: str = "Cosine"  # Cosine | Euclid | Dot
```

### 1.2 Add to `.env` and `.env.example`

```env
# =====================
# Vector Database - Qdrant
# =====================
QDRANT_URL=http://34.217.75.65:6333
QDRANT_API_KEY=Iz0ulVGF3zqHIiZsTOQFnbI7M69n21E3
QDRANT_COLLECTION_NAME=elvz_memory
QDRANT_VECTOR_SIZE=768
QDRANT_DISTANCE=Cosine
```

### 1.3 Collection Schema

**Single collection** `elvz_memory` with payload indexes:

| Payload Field  | Type     | Index Type | Purpose |
|---------------|----------|------------|---------|
| `modality`    | keyword  | keyword    | Filter by text/image/audio/video |
| `content_type`| keyword  | keyword    | Filter by knowledge/example/user_history |
| `user_id`     | keyword  | keyword    | Multi-tenant filtering |
| `platform`    | keyword  | keyword    | Platform-specific retrieval |
| `category`    | keyword  | keyword    | Sub-category filtering |
| `created_at`  | datetime | range      | Time-based retrieval & TTL |

**Why single collection?**
- Qdrant payload filtering is extremely fast (indexed)
- Avoids collection sprawl & management overhead
- Cross-modality search becomes trivial (just remove/change filter)
- Simpler backup/restore strategy
- Better resource utilization

**Why NOT separate collections per modality?**
- Different modalities can still share the same embedding dimension (768 from Gemini)
- With payload indexes, filtering by `modality="text"` is O(1), same as separate collections
- Future cross-modality queries (e.g., "find images similar to this text") are easier

---

## Phase 2: QdrantVectorStore Implementation

### 2.1 New Class in `app/core/vector_store.py`

Add `QdrantVectorStore` as a new backend alongside existing Pinecone and Weaviate:

```python
class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""

    def __init__(self):
        self._client = None
        self.embedding_service = EmbeddingService(provider="gemini")
        self.collection_name = settings.qdrant_collection_name

    async def connect(self) -> None:
        """Initialize Qdrant connection and ensure collection exists."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PayloadSchemaType,
            PayloadIndexParams, TextIndexParams, TokenizerType
        )

        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )

        # Create collection if not exists
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=distance_map.get(settings.qdrant_distance, Distance.COSINE),
                ),
            )

            # Create payload indexes for fast filtering
            for field in ["modality", "content_type", "user_id", "platform", "category"]:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

            # DateTime index for created_at
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="created_at",
                field_schema=PayloadSchemaType.DATETIME,
            )

            # Full-text index on content for hybrid search
            self._client.create_payload_index(
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

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Qdrant not connected. Call connect() first.")
        return self._client

    async def upsert(self, documents: list[VectorDocument], namespace: str = "") -> None:
        """Upsert documents. namespace maps to content_type in payload."""
        from qdrant_client.models import PointStruct

        # Generate embeddings for docs missing them
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

        # Build points
        points = []
        for doc in documents:
            payload = {
                **doc.metadata,
                "content": doc.content,
            }
            # Map legacy namespace to content_type if not set
            if "content_type" not in payload and namespace:
                payload["content_type"] = namespace
            # Default modality to text
            if "modality" not in payload:
                payload["modality"] = "text"

            points.append(PointStruct(
                id=doc.id,    # Use string UUID
                vector=doc.embedding,
                payload=payload,
            ))

        # Batch upsert (Qdrant handles large batches internally)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    async def search(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search with payload filtering."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)

        # Build Qdrant filter from metadata
        qdrant_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, dict) and "$in" in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                    )
                elif isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            qdrant_filter = Filter(must=conditions)

        # Execute search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        # Convert to VectorSearchResult
        search_results = []
        for point in results.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            search_results.append(VectorSearchResult(
                id=str(point.id),
                content=content,
                metadata=payload,
                score=point.score,
            ))

        return search_results

    async def delete(self, ids: list[str], namespace: str = "") -> None:
        """Delete points by ID."""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
```

### 2.2 Update VectorStore Factory

In the `VectorStore.__init__`, add Qdrant as a backend option:

```python
class VectorStore:
    def __init__(self, backend: str = "qdrant"):  # Change default to qdrant
        if backend == "qdrant":
            self._store = QdrantVectorStore()
        elif backend == "pinecone":
            self._store = PineconeVectorStore()
        else:
            self._store = WeaviateVectorStore()
        self.backend = backend
```

### 2.3 Backward Compatibility

The existing `VectorStore` domain methods (`search_knowledge`, `search_content_examples`, `search_user_content`) use `namespace` for Pinecone. For Qdrant, these map to payload fields:

| Pinecone Namespace | Qdrant Payload Filter |
|---|---|
| `knowledge_base` | `content_type = "knowledge"` |
| `content_examples` | `content_type = "example"` |
| `user_{user_id}` | `content_type = "user_history"` AND `user_id = "{user_id}"` |

The `QdrantVectorStore.upsert()` already maps `namespace → content_type` in payload, and the `VectorStore` high-level methods already pass `filter_metadata` which gets translated to Qdrant filters.

---

## Phase 3: Modality Detection (Piggyback on Intent Classifier)

### 3.1 The Problem

During RAG retrieval, we need to know **which modalities to search** (text, image, audio, video). Making a separate LLM call is wasteful.

### 3.2 The Solution

**Piggyback on the existing intent classification LLM call** in `app/agents/conversational_graph/nodes/intent.py`.

The intent classifier already calls `llm_client.generate_fast()` with JSON output. We just add one more field to the expected output:

#### Current Prompt Output Schema:
```json
{
    "type": "artifact|qa|...",
    "confidence": 0.9,
    "entities": { ... },
    "reasoning": "..."
}
```

#### Enhanced Prompt Output Schema:
```json
{
    "type": "artifact|qa|...",
    "confidence": 0.9,
    "entities": { ... },
    "reasoning": "...",
    "search_modalities": ["text", "image"]
}
```

### 3.3 Changes to Intent Classifier (`nodes/intent.py`)

Update `INTENT_CLASSIFICATION_PROMPT` to add:

```
Also determine which content modalities are relevant for searching context:
- "text" - Always include for text-based queries
- "image" - Include when user mentions images, visuals, photos, graphics, design
- "audio" - Include when user mentions audio, podcast, voice, sound
- "video" - Include when user mentions video, reels, stories, clips

Respond in JSON format:
{{
    "type": "...",
    "confidence": 0.0-1.0,
    "entities": {{ ... }},
    "reasoning": "...",
    "search_modalities": ["text"]  // Always include "text", add others if relevant
}}
```

Update the `IntentClassification` TypedDict in `state.py`:

```python
class IntentClassification(TypedDict, total=False):
    type: str
    confidence: float
    entities: dict
    reasoning: str
    search_modalities: list[str]  # NEW: ["text", "image", "audio", "video"]
```

Update `_classify_intent()` to extract the new field:

```python
return IntentClassification(
    type=result.get("type", "artifact"),
    confidence=float(result.get("confidence", 0.8)),
    entities=result.get("entities", {}),
    reasoning=result.get("reasoning", ""),
    search_modalities=result.get("search_modalities", ["text"]),
)
```

Update `_fallback_classify()` to set default:

```python
# Add to all fallback returns:
search_modalities=["text"],
```

### 3.4 Cost: ZERO Additional LLM Calls

The intent classifier already runs every turn. Adding `search_modalities` to the JSON schema adds ~50 output tokens. No new API call needed.

---

## Phase 4: Modality-Aware Retrieval

### 4.1 Update Memory Retriever Node (`nodes/memory.py`)

Pass modalities from intent to RAG retriever:

```python
# In MemoryRetrieverNode.__call__():
intent = state.get("current_intent") or {}
search_modalities = intent.get("search_modalities", ["text"])

# Pass to RAG retriever
rag_context = await rag_retriever.retrieve(
    query=current_input,
    user_id=user_id,
    conversation_id=conversation_id,
    context_types=["user_history", "knowledge", "examples"],
    platforms=platforms,
    modalities=search_modalities,  # NEW parameter
)
```

### 4.2 Update RAG Retriever (`services/rag_retriever.py`)

Add `modalities` parameter to `retrieve()` and `_search_by_type()`:

```python
async def retrieve(
    self,
    query: str,
    user_id: str,
    ...
    modalities: list[str] | None = None,  # NEW
) -> dict:
    modalities = modalities or ["text"]
    # Pass to each search call
    ...

async def _search_by_type(
    self,
    ctx_type: str,
    query: str,
    user_id: str,
    platforms: list[str] | None = None,
    modalities: list[str] | None = None,  # NEW
    top_k: int = 5,
) -> list[dict]:
    # Build filter with modality
    filters = {}
    if modalities:
        filters["modality"] = {"$in": modalities}
    # ... rest of existing filter logic
```

### 4.3 Search Flow

```
User: "Create a LinkedIn post about AI with a relevant image"
  ↓
Intent Classifier → search_modalities: ["text", "image"]
  ↓
Memory Retriever → passes modalities to RAG Retriever
  ↓
RAG Retriever → builds Qdrant filter:
  {
    "must": [
      {"key": "modality", "match": {"any": ["text", "image"]}},
      {"key": "user_id", "match": {"any": ["user_123", "system"]}},
      {"key": "content_type", "match": {"value": "knowledge"}}
    ]
  }
  ↓
Qdrant → Returns ONLY text + image vectors (skips audio/video)
```

---

## Phase 5: Storing Multi-Modal Content

### 5.1 Storage Patterns

#### Text Content (most common)
```python
doc = VectorDocument(
    id=str(uuid4()),
    content="LinkedIn best practices for 2026...",
    metadata={
        "modality": "text",
        "content_type": "knowledge",
        "user_id": "system",
        "platform": "linkedin",
        "category": "best_practices",
        "created_at": datetime.utcnow().isoformat(),
    }
)
# Embedding: Gemini embedding-001 on content text
```

#### Image Content
```python
doc = VectorDocument(
    id=str(uuid4()),
    content="A professional infographic showing AI trends...",  # Text description
    metadata={
        "modality": "image",
        "content_type": "user_history",
        "user_id": "user_123",
        "platform": "instagram",
        "source_url": "https://storage.example.com/img_abc.png",
        "dimensions": "1080x1080",
        "format": "png",
        "description": "AI trends infographic with blue theme",
        "created_at": datetime.utcnow().isoformat(),
    }
)
# Embedding: Gemini embedding-001 on description text
# (Same embedding model as text — semantic similarity still works)
```

#### Audio Content
```python
doc = VectorDocument(
    id=str(uuid4()),
    content="Podcast episode discussing blockchain in social media marketing...",
    metadata={
        "modality": "audio",
        "content_type": "knowledge",
        "user_id": "system",
        "source_url": "https://storage.example.com/podcast_ep42.mp3",
        "duration_seconds": 1800,
        "transcript_preview": "In this episode, we dive into...",
        "created_at": datetime.utcnow().isoformat(),
    }
)
```

#### Video Content
```python
doc = VectorDocument(
    id=str(uuid4()),
    content="Instagram reel showing behind-the-scenes of product launch...",
    metadata={
        "modality": "video",
        "content_type": "user_history",
        "user_id": "user_123",
        "platform": "instagram",
        "source_url": "https://storage.example.com/reel_xyz.mp4",
        "duration_seconds": 30,
        "thumbnail_url": "https://storage.example.com/thumb_xyz.jpg",
        "created_at": datetime.utcnow().isoformat(),
    }
)
```

### 5.2 Embedding Strategy (Important Architectural Decision)

**Decision: Use TEXT embeddings (Gemini embedding-001) for ALL modalities.**

**Why?**
1. All modalities get a text description/caption stored in `content`
2. Text embeddings on descriptions capture semantic meaning well
3. Avoids maintaining multiple embedding models (CLIP, whisper, etc.)
4. Consistent 768-dimension vectors across all modalities
5. Single collection, single vector config, simple architecture

**Future upgrade path (if needed):**
- Add a second named vector for image-specific embeddings (CLIP)
- Qdrant supports **named vectors** — one point can have multiple vectors:
  ```python
  vectors_config={
      "text": VectorParams(size=768, distance=Distance.COSINE),
      "image": VectorParams(size=512, distance=Distance.COSINE),  # CLIP
  }
  ```
- This is a non-breaking change — existing points keep working

---

## Phase 6: Memory Saver Integration

### 6.1 Update Memory Saver Node (`nodes/saver.py`)

After generating content, persist artifacts to Qdrant for future retrieval:

```python
# In MemorySaverNode._save_working_memory() or a new method:

async def _save_to_long_term_memory(self, state: ConversationState) -> None:
    """Persist generated artifacts to vector store for future retrieval."""
    artifacts = state.get("artifacts", [])
    if not artifacts:
        return

    documents = []
    for artifact in artifacts:
        content = (artifact.get("content") or {})
        text = content.get("text", "")
        if not text:
            continue

        doc = VectorDocument(
            id=artifact.get("id", str(uuid4())),
            content=text,
            metadata={
                "modality": "text",
                "content_type": "user_history",
                "user_id": state["user_id"],
                "platform": artifact.get("platform", "unknown"),
                "category": "generated_content",
                "conversation_id": state["conversation_id"],
                "created_at": datetime.utcnow().isoformat(),
                "engagement": {},  # To be updated later
                "tags": [],  # Could extract from hashtags
            }
        )
        documents.append(doc)

    if documents:
        await vector_store.add_user_content(state["user_id"], documents)
```

---

## Phase 7: Collection Setup Utility

### 7.1 New File: `app/utils/setup_qdrant.py`

CLI utility to initialize collection, create indexes, and seed with knowledge base:

```python
"""
Setup Qdrant collection for Elvz long-term memory.

Usage:
    python -m app.utils.setup_qdrant --setup          # Create collection + indexes
    python -m app.utils.setup_qdrant --seed            # Seed knowledge base
    python -m app.utils.setup_qdrant --status          # Check collection stats
    python -m app.utils.setup_qdrant --test "query"    # Test search
"""
```

Key operations:
- Create collection with 768-dim Cosine vectors
- Create all payload indexes (modality, content_type, user_id, platform, category, created_at)
- Create full-text index on `content` field for hybrid search
- Seed with existing knowledge base data (25 social media best practices)
- Print collection stats (point count, indexed fields, etc.)

---

## Phase 8: Production Architecture Recommendations

### 8.1 Qdrant Deployment (Current: Single Node)

Your Qdrant is running at `http://34.217.75.65:6333` — this is a single node. For production:

| Concern | Recommendation |
|---------|---------------|
| **High Availability** | Deploy Qdrant in cluster mode (3+ nodes) with replication factor 2 |
| **Snapshots** | Schedule daily snapshots via `POST /collections/{name}/snapshots` |
| **TLS** | Put behind nginx/ALB with TLS termination (currently HTTP) |
| **Auth** | API key is set (good), but rotate every 90 days |
| **Monitoring** | Qdrant exposes Prometheus metrics at `/metrics` — scrape with Grafana |

### 8.2 Index Optimization

```python
# Quantization for faster search on large collections (>100k vectors)
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig

client.update_collection(
    collection_name="elvz_memory",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type="int8",
            quantile=0.99,
            always_ram=True,
        )
    ),
)
```

### 8.3 Hybrid Search (BM25 + Dense Vector)

Qdrant supports **full-text search** alongside vector search. We use this for:
- **Dense vectors**: Semantic similarity (main search)
- **Full-text (BM25)**: Exact keyword matches (supplementary)

The `content` field has a text index. To combine:

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

# Reciprocal Rank Fusion of vector + text search
results = client.query_points(
    collection_name="elvz_memory",
    prefetch=[
        Prefetch(query=query_embedding, limit=20),  # Vector search
        Prefetch(
            query=models.FilterQuery(filter=text_filter),
            limit=20,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Rank fusion
    limit=10,
)
```

### 8.4 TTL / Data Lifecycle

Qdrant doesn't have built-in TTL. Implement with a background task:

```python
# Celery task: Run daily
async def cleanup_old_vectors():
    """Delete vectors older than 6 months."""
    six_months_ago = (datetime.utcnow() - timedelta(days=180)).isoformat()

    client.delete(
        collection_name="elvz_memory",
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=Range(lt=six_months_ago),
                    ),
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value="user_history"),
                    ),
                ]
            )
        ),
    )
```

### 8.5 Embedding Cache

Cache embeddings in Redis to avoid re-computing for repeated queries:

```python
# In EmbeddingService.embed():
cache_key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
cached = await cache.get(cache_key)
if cached:
    return json.loads(cached)

embedding = await self._compute_embedding(text)
await cache.set(cache_key, json.dumps(embedding), ttl=3600)
return embedding
```

### 8.6 Batch Ingestion Pipeline

For bulk uploads (e.g., importing a user's content library):

```python
async def bulk_ingest(
    user_id: str,
    items: list[dict],  # [{content, modality, platform, source_url, ...}]
    batch_size: int = 50,
):
    """Parallel batch embedding + upsert."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [item["content"] for item in batch]

        # Batch embed (single API call for 50 texts)
        embeddings = await embedding_service.embed_batch(texts)

        documents = [
            VectorDocument(
                id=str(uuid4()),
                content=item["content"],
                embedding=emb,
                metadata={
                    "modality": item.get("modality", "text"),
                    "content_type": "user_history",
                    "user_id": user_id,
                    "platform": item.get("platform", "unknown"),
                    "source_url": item.get("source_url"),
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            for item, emb in zip(batch, embeddings)
        ]

        await vector_store._store.upsert(documents)
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `app/utils/setup_qdrant.py` | Collection setup & seeding utility |

## Files to Modify

| File | Changes |
|------|---------|
| `app/core/config.py` | Add Qdrant settings (url, api_key, collection, vector_size, distance) |
| `app/core/vector_store.py` | Add `QdrantVectorStore` class, change default backend to "qdrant" |
| `app/agents/conversational_graph/state.py` | Add `search_modalities` to `IntentClassification` |
| `app/agents/conversational_graph/nodes/intent.py` | Enhance prompt to output `search_modalities` |
| `app/agents/conversational_graph/nodes/memory.py` | Pass modalities to RAG retriever |
| `app/services/rag_retriever.py` | Accept `modalities` param, add to search filters |
| `app/agents/conversational_graph/nodes/saver.py` | Persist artifacts to vector store |
| `.env` / `.env.example` | Add Qdrant env vars |
| `requirements.txt` | Add `qdrant-client>=1.12.0` |

## Files NOT to Touch

| File | Reason |
|------|--------|
| `app/services/memory_manager.py` | Calls vector_store indirectly — no changes needed |
| `app/core/model_config.py` | No new LLM task type needed |
| `app/core/llm_clients.py` | No new LLM calls |
| All graph nodes except intent/memory/saver | No changes needed |
| `app/agents/conversational_graph/graph.py` | Graph structure unchanged |

---

## Implementation Order

1. **Config**: Add Qdrant settings to `config.py`, `.env`, `.env.example`
2. **Dependency**: Add `qdrant-client` to `requirements.txt`
3. **Backend**: Implement `QdrantVectorStore` in `vector_store.py`
4. **Setup**: Create `setup_qdrant.py` utility, run to initialize collection
5. **Intent**: Enhance prompt and parsing in `intent.py` + state schema
6. **Retriever**: Add modality filtering to `rag_retriever.py`
7. **Memory Node**: Pass modalities from intent to retriever in `memory.py`
8. **Saver**: Add artifact persistence to vector store in `saver.py`
9. **Migrate Data**: Run seed script to populate knowledge base in Qdrant
10. **Test**: Verify end-to-end retrieval with modality filtering

---

## Verification

```bash
# 1. Setup collection
python -m app.utils.setup_qdrant --setup

# 2. Seed knowledge base
python -m app.utils.setup_qdrant --seed

# 3. Check collection stats
python -m app.utils.setup_qdrant --status
# Expected: 25+ points, 6 indexed payload fields

# 4. Test search with modality filter
python -m app.utils.setup_qdrant --test "LinkedIn best practices" --modality text

# 5. Start server and send chat
uvicorn app.api.main:app --reload
# POST /api/v1/chat/v2 with: "Create a LinkedIn post about AI"
# Verify logs show: search_modalities=["text"], qdrant filter applied

# 6. Check Qdrant dashboard
# http://34.217.75.65:6333/dashboard
```


List all collections:


curl http://localhost:6333/collections
Get details of a specific collection (vector size, count, config):


curl http://localhost:6333/collections/elvz_memory
Count points in a collection:


curl http://localhost:6333/collections/elvz_memory/points/count \
  -H "Content-Type: application/json" \
  -d '{"exact": true}'
Browse points (scroll through content):


curl -X POST http://localhost:6333/collections/elvz_memory/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "with_payload": true,
    "with_vector": false
  }'
Filter by user_id to see a specific user's data:


curl -X POST http://localhost:6333/collections/elvz_memory/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 20,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [
        { "key": "user_id", "match": { "value": "dev-user-001" } }
      ]
    }
  }'
Filter by content_type (e.g. scraped posts vs conversation turns):


curl -X POST http://localhost:6333/collections/elvz_memory/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 20,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [
        { "key": "content_type", "match": { "value": "user_history" } },
        { "key": "platform", "match": { "value": "linkedin" } }
      ]
    }
  }'
If you have a Qdrant API key set (QDRANT_API_KEY in .env), add it to every request:


curl http://localhost:6333/collections \
  -H "api-key: your-api-key"
Alternatively, use the Qdrant Web UI — it's built in and available at:


http://localhost:6333/dashboard
It gives you a visual browser for collections and points without needing curl.
---

## Pinecone → Qdrant Migration Checklist

- [ ] Add qdrant-client to requirements.txt
- [ ] Add Qdrant config to config.py and .env
- [ ] Implement QdrantVectorStore class
- [ ] Create setup utility
- [ ] Run collection setup on Qdrant server
- [ ] Seed knowledge base
- [ ] Change VectorStore default backend to "qdrant"
- [ ] Add search_modalities to intent classifier
- [ ] Add modality filtering to RAG retriever
- [ ] Add artifact persistence to memory saver
- [ ] Test end-to-end flow
- [ ] Decommission Pinecone (after validation)
