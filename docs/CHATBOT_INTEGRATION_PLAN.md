# Chatbot Integration Plan for elvz-backend

## Overview

This plan integrates a conversational AI layer on top of the existing `social_media_manager` content generation pipeline. The new layer adds multi-turn conversations, memory management, query decomposition, guardrails, and human-in-the-loop (HITL) workflows.

---

## Current Architecture (Reusable Components)

| Component | Location | Reuse Strategy |
|-----------|----------|----------------|
| Platform Orchestrator | `app/agents/platform_orchestrator/orchestrator.py` | Wrap with new conversational graph |
| Intent Classifier | `app/agents/platform_orchestrator/intent_classifier.py` | Enhance for conversation intents |
| Social Media Manager | `app/agents/elves/social_media_manager/orchestrator.py` | Call as subgraph for artifact generation |
| Vector Store | `app/core/vector_store.py` | Extend for conversation embeddings |
| Redis Cache | `app/core/cache.py` | Add conversation working memory |
| LLM Clients | `app/core/llm_clients.py` | Use for all new nodes |
| WebSocket | `app/api/websocket.py` | Enhance for rich event streaming |
| Base Tool | `app/tools/base.py` | Pattern for new services |

---

## New Directory Structure

```
app/
├── agents/
│   └── conversational_graph/           # NEW
│       ├── __init__.py
│       ├── graph.py                     # Master LangGraph StateGraph
│       ├── state.py                     # ConversationState TypedDict
│       └── nodes/
│           ├── __init__.py
│           ├── guardrail.py             # AWS Bedrock content safety
│           ├── intent.py                # Conversation intent classification
│           ├── decomposer.py            # Multi-platform query decomposition
│           ├── memory.py                # Memory retrieval node
│           ├── context.py               # Context building with token budget
│           ├── router.py                # Route to Q&A or artifact generation
│           ├── follow_up.py             # HITL follow-up detection/generation
│           ├── data_checker.py          # Social media data availability
│           ├── orchestrator.py          # Multi-platform coordination
│           ├── aggregator.py            # Stream event aggregation
│           └── saver.py                 # Memory persistence
├── services/                            # NEW
│   ├── __init__.py
│   ├── memory_manager.py                # 4-layer memory system
│   ├── rag_retriever.py                 # Semantic search with reranking
│   ├── query_decomposer.py              # Multi-platform query parsing
│   ├── conversation_service.py          # Conversation CRUD
│   ├── artifact_service.py              # Artifact management
│   └── hitl_service.py                  # Human-in-the-loop workflows
├── middleware/                          # NEW
│   ├── __init__.py
│   └── guardrails.py                    # AWS Bedrock integration
├── models/
│   ├── conversation.py                  # NEW: Conversation, Message
│   ├── artifact.py                      # NEW: Artifact, ArtifactBatch
│   └── hitl.py                          # NEW: HITLRequest
├── api/routes/
│   ├── conversations.py                 # NEW: Conversation endpoints
│   └── artifacts.py                     # NEW: Artifact endpoints
└── core/
    ├── checkpointer.py                  # NEW: LangGraph PostgreSQL checkpointer
    └── observability.py                 # NEW: LangFuse integration
```

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Dependency Upgrades

**File:** `requirements.txt`

```diff
- langgraph==0.0.26
- langchain==0.1.4
+ langgraph==0.2.50
+ langgraph-checkpoint-postgres==2.0.10
+ langchain==0.3.14
+ langchain-core==0.3.28

# New dependencies
+ boto3==1.35.0                    # AWS Bedrock
+ langfuse==2.0.0                  # Observability
+ sentry-sdk[fastapi]==2.0.0       # Error tracking
```

### 1.2 Database Models

**New File:** `app/models/conversation.py`

```python
# Tables: conversations, messages
# Key fields: conversation_id, thread_id (for LangGraph), user_id, status
# Relationships: Conversation -> Messages (1:N)
```

**New File:** `app/models/artifact.py`

```python
# Tables: artifacts, artifact_batches, query_decompositions
# Key fields: artifact_id, batch_id, platform, content (JSONB), status
```

**New File:** `app/models/hitl.py`

```python
# Table: hitl_requests
# Key fields: request_type, status, expires_at
```

### 1.3 Alembic Migrations

```bash
# Setup
alembic init migrations

# Migrations to create:
001_add_conversations.py
002_add_artifacts.py
003_add_hitl_requests.py
004_add_query_decompositions.py
```

### 1.4 LangGraph Checkpointer

**New File:** `app/core/checkpointer.py`

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def get_checkpointer() -> AsyncPostgresSaver:
    """PostgreSQL-backed checkpointer for state persistence."""
    checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)
    await checkpointer.setup()
    return checkpointer
```

### 1.5 Configuration Updates

**File:** `app/core/config.py` - Add:

```python
# AWS Bedrock
aws_region: str = "us-east-1"
bedrock_guardrail_id: str = ""

# LangFuse
langfuse_public_key: str = ""
langfuse_secret_key: str = ""

# Sentry
sentry_dsn: str = ""
```

---

## Phase 2: Memory System (Week 3-4)

### 2.1 Memory Manager

**New File:** `app/services/memory_manager.py`

4-layer architecture:
1. **Working Memory** (Redis) - Current conversation context, 1-hour TTL
2. **Short-Term Memory** (PostgreSQL) - Recent messages, 50 message limit
3. **Long-Term Memory** (Vector Store) - Semantic embeddings, 6-month retention
4. **User Profile Memory** (PostgreSQL + Cache) - Brand voice, preferences

Key methods:
- `get_working_memory(conversation_id)` - From Redis
- `get_recent_messages(conversation_id, limit=10)` - From PostgreSQL
- `retrieve_relevant_context(query, user_id, top_k=5)` - From Vector Store
- `get_user_profile(user_id)` - From cache or database

### 2.2 RAG Retriever

**New File:** `app/services/rag_retriever.py`

Uses existing `app/core/vector_store.py` with enhancements:
- Query reformulation based on conversation context
- Token budget management (6000 token limit)
- Multi-namespace search (knowledge_base, content_examples, user_history)

---

## Phase 3: Conversational Graph (Week 5-6)

### 3.1 State Schema

**New File:** `app/agents/conversational_graph/state.py`

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class ConversationState(TypedDict):
    # Core
    conversation_id: str
    user_id: str
    thread_id: str
    messages: Annotated[list, add_messages]
    current_input: str

    # Intent & Decomposition
    current_intent: dict  # {type, confidence, entities}
    decomposed_queries: list[dict]  # [{platform, query, priority}]

    # Memory
    working_memory: dict
    retrieved_memory: list[dict]
    user_profile: dict

    # Safety
    guardrail_passed: bool
    guardrail_violations: list[str]

    # HITL
    needs_follow_up: bool
    follow_up_questions: list[str]
    requires_approval: bool

    # Artifacts
    artifacts: list[dict]
    artifact_batch_id: str

    # Streaming
    stream_events: list[dict]

    # Metadata
    execution_trace: list[dict]
    errors: list[str]
```

### 3.2 Graph Nodes

| Node | Purpose | Key Logic |
|------|---------|-----------|
| `guardrail_check` | AWS Bedrock content safety | Block harmful inputs |
| `intent_classifier` | Classify conversation intent | qa/artifact/clarification |
| `query_decomposer` | Parse multi-platform queries | Extract platforms, create sub-queries |
| `memory_retriever` | Fetch relevant context | 4-layer memory lookup |
| `context_builder` | Build LLM context | Token budget management |
| `router` | Route to appropriate handler | Q&A vs artifact generation |
| `follow_up_detector` | Check if more info needed | Missing platform/topic detection |
| `follow_up_generator` | Create follow-up questions | Natural language questions |
| `data_checker` | Verify data availability | Check scraped social data exists |
| `multi_platform_orchestrator` | Coordinate artifact generation | Call social_media_manager per platform |
| `stream_aggregator` | Collect streaming events | Progressive updates |
| `memory_saver` | Persist state | Save to Redis + PostgreSQL |

### 3.3 Graph Structure

**New File:** `app/agents/conversational_graph/graph.py`

```
Entry -> guardrail_check
    |
    ├─[blocked]─> memory_saver -> END
    |
    └─[passed]─> intent_classifier -> query_decomposer -> memory_retriever
                                                              |
                                                              v
                                                        context_builder
                                                              |
                                                              v
                                                           router
                                                              |
                    ┌─────────────────────────────────────────┴────────────────┐
                    |                                                          |
                    v                                                          v
            follow_up_detector                                          data_checker
                    |                                                          |
                    v                                                          v
            follow_up_generator                               multi_platform_orchestrator
                    |                                                          |
                    └───────────────────> stream_aggregator <──────────────────┘
                                                |
                                                v
                                          memory_saver -> END
```

### 3.4 Integration with Existing Elves

The `multi_platform_orchestrator` node calls existing `SocialMediaManagerElf.execute()`:

```python
# In nodes/orchestrator.py
from app.agents.elves.social_media_manager.orchestrator import SocialMediaManagerElf

async def __call__(self, state: ConversationState) -> ConversationState:
    elf = SocialMediaManagerElf()

    for query in state["decomposed_queries"]:
        result = await elf.execute(
            request={"platform": query["platform"], "topic": query["topic"]},
            context={"user_id": state["user_id"], **state["working_memory"]}
        )
        state["artifacts"].append(result)

    return state
```

---

## Phase 4: API Integration (Week 7-8)

### 4.1 Enhanced Chat Endpoint

**File:** `app/api/routes/chat.py` - Modify:

```python
@router.post("", response_model=EnhancedChatResponse)
async def chat(request: EnhancedChatRequest):
    # Get or create conversation
    conversation = await conversation_service.get_or_create(...)

    # Build initial state
    initial_state = build_initial_state(conversation, request)

    # Invoke conversational graph with checkpointing
    graph = await get_conversational_graph()
    config = {"configurable": {"thread_id": str(conversation.thread_id)}}
    result = await graph.ainvoke(initial_state, config)

    # Save messages and return response
    await conversation_service.add_message(...)
    return EnhancedChatResponse(...)
```

### 4.2 Enhanced WebSocket

**File:** `app/api/websocket.py` - Add event types:

```python
# New event types
"node_progress"     # Graph node execution progress
"platform_progress" # Per-platform generation status
"artifact_chunk"    # Streaming artifact content
"hitl_request"      # Human-in-the-loop prompt
"hitl_response"     # User HITL response
```

### 4.3 New Endpoints

**New File:** `app/api/routes/conversations.py`
- `GET /conversations` - List user conversations
- `GET /conversations/{id}` - Get conversation with messages
- `DELETE /conversations/{id}` - Delete conversation

**New File:** `app/api/routes/artifacts.py`
- `GET /artifacts/{id}` - Get specific artifact
- `GET /artifact-batches/{id}` - Get multi-platform batch
- `POST /artifacts/{id}/feedback` - User feedback

---

## Phase 5: AWS & Observability (Week 9-10)

### 5.1 AWS Bedrock Guardrails

**New File:** `app/middleware/guardrails.py`

```python
import boto3

class BedrockGuardrail:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime', region_name=settings.aws_region)

    async def check(self, content: str) -> tuple[bool, list[str]]:
        response = self.client.apply_guardrail(
            guardrailIdentifier=settings.bedrock_guardrail_id,
            content=[{'text': {'text': content}}]
        )
        passed = response['action'] != 'BLOCKED'
        violations = [a['topic'] for a in response.get('assessments', [])]
        return passed, violations
```

### 5.2 LangFuse Integration

**New File:** `app/core/observability.py`

```python
from langfuse.callback import CallbackHandler

def get_langfuse_handler(user_id: str, session_id: str) -> CallbackHandler:
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        user_id=user_id,
        session_id=session_id
    )
```

### 5.3 Sentry Integration

Add to `app/api/main.py`:

```python
import sentry_sdk

sentry_sdk.init(
    dsn=settings.sentry_dsn,
    environment=settings.environment,
    traces_sample_rate=0.1
)
```

---

## Verification Plan

### Unit Tests
```bash
# Test memory manager
pytest tests/unit/test_memory_manager.py

# Test query decomposer
pytest tests/unit/test_query_decomposer.py

# Test graph nodes
pytest tests/unit/test_graph_nodes.py
```

### Integration Tests
```bash
# Test full conversation flow
pytest tests/integration/test_conversation_flow.py

# Test checkpointing
pytest tests/integration/test_checkpointing.py

# Test WebSocket streaming
pytest tests/integration/test_websocket_streaming.py
```

### Manual Testing
1. Start server: `uvicorn app.api.main:app --reload`
2. Connect WebSocket: `ws://localhost:8000/ws/stream/{client_id}`
3. Send test message: `{"type": "chat", "data": {"message": "Generate LinkedIn and Facebook posts about AI"}}`
4. Verify streaming events received
5. Verify artifacts saved to database
6. Verify checkpointing by restarting and resuming conversation

### Load Testing
```bash
# 100 concurrent WebSocket connections
locust -f tests/load/locustfile.py --host=ws://localhost:8000
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LangGraph 0.0.26 → 0.2.x breaking changes | Wrapper layer + parallel testing before cutover |
| AWS Bedrock rate limits | Circuit breaker + fallback to OpenAI moderation |
| Checkpointer performance | PostgreSQL connection pooling + async writes |
| Memory bloat | Token budget limits + TTL on Redis keys |

---

## Critical Files Summary

### Must Modify
- `requirements.txt` - Upgrade LangGraph, add new deps
- `app/core/config.py` - Add AWS/LangFuse/Sentry settings
- `app/api/routes/chat.py` - Use new conversational graph
- `app/api/websocket.py` - Add rich event streaming
- `app/api/main.py` - Register new routes, init observability

### Must Create
- `app/agents/conversational_graph/` - Entire new module
- `app/services/` - Memory, RAG, conversation services
- `app/middleware/guardrails.py` - AWS Bedrock
- `app/models/conversation.py`, `artifact.py`, `hitl.py`
- `app/core/checkpointer.py`, `observability.py`
- `migrations/versions/` - Alembic migrations

### Keep Unchanged
- `app/agents/elves/social_media_manager/` - Called as subgraph
- `app/core/vector_store.py` - Extended, not replaced
- `app/core/cache.py` - Extended with new key patterns
- `app/core/llm_clients.py` - Used by new nodes

---

## Implementation Order

1. **Week 1**: Upgrade dependencies, create models, run migrations
2. **Week 2**: Implement checkpointer, test state persistence
3. **Week 3**: Build memory manager, test 4-layer retrieval
4. **Week 4**: Implement RAG retriever with token budgeting
5. **Week 5**: Create graph nodes (guardrail, intent, decomposer)
6. **Week 6**: Create graph nodes (memory, router, follow-up, orchestrator)
7. **Week 7**: Build master graph, integrate with existing Elves
8. **Week 8**: Enhance chat endpoint, add conversation CRUD
9. **Week 9**: Integrate AWS Bedrock, LangFuse, Sentry
10. **Week 10**: Testing, optimization, documentation
