# Code Flow Documentation

This document explains how a user message flows through the conversational graph system.

## 📊 Overview

```
User sends "hey"
    ↓
API Endpoint (chat_v2.py)
    ↓
invoke_conversation() (graph.py)
    ↓
LangGraph Execution (12 nodes)
    ↓
Response returned to user
    ↓
Execution logged to PostgreSQL
```

---

## 🚀 Entry Point

### File: `app/api/routes/chat_v2.py`

**Endpoint:** `POST /api/v1/chat/v2`

```python
@router.post("", response_model=ConversationalChatResponse)
async def conversational_chat(
    request: ConversationalChatRequest,
    user_id: str = Depends(get_current_user_id),
) -> ConversationalChatResponse:
    """
    Main entry point for conversational chat.
    
    Flow:
    1. Get or create conversation in database
    2. Call invoke_conversation() to execute graph
    3. Extract response, artifacts, and metadata
    4. Return response to user
    """
    
    # Line 74-95: Get or create conversation
    if request.conversation_id:
        conversation = await conversation_service.get_conversation(request.conversation_id)
        conversation_id = request.conversation_id
        thread_id = conversation.thread_id
    else:
        conversation = await conversation_service.create_conversation(
            user_id=user_id,
            title=request.message[:100],
            metadata=request.context or {},
        )
        conversation_id = conversation.id
        thread_id = conversation.thread_id
    
    # Line 98-104: ⭐ THIS IS WHERE THE GRAPH STARTS
    result_state = await invoke_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,
        user_input=request.message,  # User's message: "hey"
        config=request.context,
    )
    
    # Line 107-140: Extract and return response
    final_response = result_state.get("final_response", "I'm processing your request.")
    suggestions = result_state.get("suggestions", [])
    artifacts = result_state.get("artifacts", [])
    
    return ConversationalChatResponse(
        response=final_response,
        conversation_id=conversation_id,
        thread_id=thread_id,
        artifacts=artifacts,
        suggestions=suggestions,
        hitl_request=hitl_request,
        metadata=metadata,
    )
```

**Key Points:**
- Creates or retrieves conversation from database
- Calls `invoke_conversation()` which executes the LangGraph
- Extracts final response and artifacts from result state
- Returns structured response to user

---

## 🔄 Graph Invocation

### File: `app/agents/conversational_graph/graph.py`

**Function:** `invoke_conversation()`

```python
async def invoke_conversation(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    config: Optional[dict] = None,
) -> ConversationState:
    """
    Execute the conversational graph with user input.
    
    Flow:
    1. Generate execution ID for monitoring
    2. Log execution start (PostgreSQL)
    3. Get compiled graph
    4. Create initial state with user input
    5. Execute graph (goes through all nodes)
    6. Log execution completion with node outputs
    7. Return final state
    """
    
    # Line 244-251: Initialize monitoring
    import time
    from uuid import uuid4
    from app.services.execution_monitor import execution_logger, ExecutionStatus
    
    execution_id = str(uuid4())
    start_time = time.time()
    
    # Line 254-259: Log execution start (non-blocking write to PostgreSQL)
    execution_logger.log_execution_started(
        execution_id=execution_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_message=user_input,
    )
    
    # Line 262: Get the compiled graph instance
    graph = await get_conversational_graph()
    
    # Line 265-270: Create initial state with user input
    initial_state = create_initial_state(
        conversation_id=conversation_id,
        user_id=user_id,
        thread_id=thread_id,
        user_input=user_input,
    )
    
    # Line 273: Store execution_id in state for tracking
    initial_state["execution_id"] = execution_id
    
    # Line 276-283: Build config with thread_id for checkpointing
    invoke_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    if config:
        invoke_config.update(config)
    
    # Line 287: ⚡ EXECUTE THE GRAPH (this runs all nodes)
    result = await graph.ainvoke(initial_state, invoke_config)
    
    # Line 290-303: Calculate duration and determine status
    duration_ms = int((time.time() - start_time) * 1000)
    
    errors = result.get("errors", [])
    failed_nodes = [
        trace["node"]
        for trace in result.get("execution_trace", [])
        if trace.get("status") == "failed"
    ]
    
    if errors or failed_nodes:
        status = ExecutionStatus.PARTIAL if result.get("final_response") else ExecutionStatus.FAILED
    else:
        status = ExecutionStatus.COMPLETED
    
    # Line 306-314: Log execution completion (non-blocking)
    execution_logger.log_execution_completed(
        execution_id=execution_id,
        status=status,
        duration_ms=duration_ms,
        response_message=result.get("final_response", ""),
        execution_trace=result.get("execution_trace", []),
        error_summary="; ".join(errors) if errors else None,
        failed_nodes=failed_nodes,
        start_time=start_time,
    )
    
    return result
```

**Key Points:**
- Generates unique execution ID for tracking
- Logs to PostgreSQL asynchronously (non-blocking)
- Creates initial state with user's message
- Executes graph via `graph.ainvoke()`
- Tracks errors and failed nodes
- Logs completion with detailed node outputs

---

## 📈 Graph Structure

### File: `app/agents/conversational_graph/graph.py`

**Function:** `create_conversational_graph()`

```python
def create_conversational_graph(checkpointer=None) -> StateGraph:
    """
    Create the master conversational graph.
    
    Graph Flow (for message "hey"):
    
    Entry ──> guardrail_check (line 97)
              │
              ├─[blocked]──────────────────────────> memory_saver ──> END
              │
              └─[passed]──> intent_classifier (line 118)
                                  │
                                  ▼
                            query_decomposer (line 118)
                                  │
                                  ▼
                            memory_retriever (line 119)
                                  │
                                  ▼
                            context_builder (line 120)
                                  │
                                  ▼
                                router (line 121)
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
        follow_up_detector   data_checker   stream_aggregator
                │                 │           (for Q&A like "hey")
                │                 │                 │
        [needs_follow_up]   [artifact gen]          ▼
                │                 │           memory_saver
                ▼                 │                 │
        follow_up_generator       │                 ▼
                │                 │                END
                │                 ▼
                │       multi_platform_orchestrator
                │                 │
                └─────────────────┘
                          │
                          ▼
                   stream_aggregator
                          │
                          ▼
                    memory_saver
                          │
                          ▼
                         END
    """
    
    workflow = StateGraph(ConversationState)
    
    # ==================== Add Nodes (Lines 83-94) ====================
    workflow.add_node("guardrail_check", guardrail_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("query_decomposer", query_decomposer_node)
    workflow.add_node("memory_retriever", memory_retriever_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("router", router_node)
    workflow.add_node("follow_up_detector", follow_up_detector_node)
    workflow.add_node("follow_up_generator", follow_up_generator_node)
    workflow.add_node("data_checker", data_checker_node)
    workflow.add_node("multi_platform_orchestrator", multi_platform_orchestrator_node)
    workflow.add_node("stream_aggregator", stream_aggregator_node)
    workflow.add_node("memory_saver", memory_saver_node)
    
    # ==================== Set Entry Point (Line 97) ====================
    workflow.set_entry_point("guardrail_check")
    
    # ==================== Conditional Edges ====================
    
    # Guardrail routing (Lines 100-114)
    def guardrail_router(state):
        if state.get("guardrail_passed", True):
            return "intent_classifier"
        return "memory_saver"  # Blocked content skips to end
    
    workflow.add_conditional_edges(
        "guardrail_check",
        guardrail_router,
        {
            "intent_classifier": "intent_classifier",
            "memory_saver": "memory_saver",
        },
    )
    
    # Main sequential flow (Lines 118-121)
    workflow.add_edge("intent_classifier", "query_decomposer")
    workflow.add_edge("query_decomposer", "memory_retriever")
    workflow.add_edge("memory_retriever", "context_builder")
    workflow.add_edge("context_builder", "router")
    
    # Router decides next path (Lines 124-153)
    def router_decision(state):
        route = (state.get("working_memory") or {}).get("route", "artifact_generation")
        needs_data_check = (state.get("working_memory") or {}).get("needs_data_check", False)
        
        if route == "qa_response":
            # Q&A like "hey" goes directly to aggregator
            return "stream_aggregator"
        
        if route == "process_clarification":
            return "follow_up_detector"
        
        if route in ["artifact_generation", "modification"]:
            if needs_data_check:
                return "data_checker"
            return "follow_up_detector"
        
        return "data_checker"
    
    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            "follow_up_detector": "follow_up_detector",
            "data_checker": "data_checker",
            "stream_aggregator": "stream_aggregator",
        },
    )
    
    # Follow-up path (Lines 156-174)
    def follow_up_router(state):
        if state.get("needs_follow_up", False):
            return "follow_up_generator"
        return "data_checker"
    
    workflow.add_conditional_edges("follow_up_detector", follow_up_router, ...)
    workflow.add_edge("follow_up_generator", "stream_aggregator")
    
    # Artifact generation path (Lines 177-180)
    workflow.add_edge("data_checker", "multi_platform_orchestrator")
    workflow.add_edge("multi_platform_orchestrator", "stream_aggregator")
    
    # Final path (Lines 184-185)
    workflow.add_edge("stream_aggregator", "memory_saver")
    workflow.add_edge("memory_saver", END)
    
    # Compile and return (Lines 188-191)
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
```

**Key Points:**
- 12 nodes in total
- Entry point is `guardrail_check`
- Conditional routing based on intent and context
- Q&A messages skip artifact generation
- All paths end at `memory_saver` before END

---

## 🔍 Node Execution Flow (Example: "hey")

### 1. Guardrail Check

**File:** `app/agents/conversational_graph/nodes/guardrail.py`

```python
async def __call__(self, state: ConversationState):
    """
    Check content safety using AWS Bedrock Guardrails.
    
    For "hey":
    - Input: "hey"
    - Action: Check for harmful content, PII, prompt injection
    - Result: PASS (no violations)
    - Output: guardrail_passed=True, guardrail_action="pass"
    - Next: intent_classifier
    """
    
    if not self.enabled:
        # Guardrails not configured, pass through
        state["guardrail_passed"] = True
        state["guardrail_action"] = "pass"
    else:
        # Apply Bedrock guardrail
        result = await self._apply_guardrail(state["current_input"])
        state["guardrail_passed"] = result["passed"]
        state["guardrail_violations"] = result.get("violations", [])
        state["guardrail_action"] = result.get("action", "pass")
    
    # Add to execution trace
    add_execution_trace(
        state,
        "guardrail_check",
        "completed",
        execution_time,
        metadata={
            "guardrail_passed": state["guardrail_passed"],
            "guardrail_action": state["guardrail_action"],
            "violations": state.get("guardrail_violations", []),
            "enabled": self.enabled,
        }
    )
    
    return state
```

---

### 2. Intent Classifier

**File:** `app/agents/conversational_graph/nodes/intent.py`

```python
async def __call__(self, state: ConversationState):
    """
    Classify user intent using LLM.
    
    For "hey":
    - Input: "hey"
    - Action: LLM classifies intent type
    - Result: type="qa", confidence=0.9
    - Output: intent_type="qa", entities={}
    - Next: query_decomposer
    """
    
    # Build conversation context from recent messages
    context = self._build_context(state)
    
    # Classify intent using LLM
    intent = await self._classify_intent(
        state["current_input"],
        context,
    )
    # For "hey" → {"type": "qa", "confidence": 0.9, "entities": {}, "reasoning": "Greeting"}
    
    # Store in state
    state["current_intent"] = intent
    state["intent_history"].append(intent.get("type", "unknown"))
    
    # Set multi-platform flag
    if intent.get("type") == "multi_platform":
        state["is_multi_platform"] = True
    
    # Add to execution trace
    add_execution_trace(
        state,
        "intent_classifier",
        "completed",
        execution_time,
        metadata={
            "intent_type": intent.get("type"),      # "qa"
            "confidence": intent.get("confidence"), # 0.9
            "entities": (intent.get("entities") or {}),
            "reasoning": (intent.get("reasoning") or "")[:100],
        }
    )
    
    return state
```

---

### 3. Query Decomposer

**File:** `app/agents/conversational_graph/nodes/decomposer.py`

```python
async def __call__(self, state: ConversationState):
    """
    Decompose query into platform-specific sub-queries.
    
    For "hey":
    - Input: intent_type="qa"
    - Action: Create single passthrough query
    - Result: Single "general" platform query
    - Output: query_count=1, platforms=["general"]
    - Next: memory_retriever
    """
    
    intent = state.get("current_intent") or {}
    
    # Only decompose for artifact or multi_platform intents
    if intent.get("type") not in ["artifact", "multi_platform"]:
        # For Q&A, clarification, etc., create single passthrough query
        state["decomposed_queries"] = [
            DecomposedQuery(
                platform="general",
                query=state["current_input"],  # "hey"
                topic=(intent.get("entities") or {}).get("topic", ""),
                priority=1,
                status="pending",
            )
        ]
        state["decomposition_complete"] = True
    
    # Add to execution trace
    add_execution_trace(
        state,
        "query_decomposer",
        "completed",
        execution_time,
        metadata={
            "query_count": len(state["decomposed_queries"]),  # 1
            "is_multi_platform": state.get("is_multi_platform", False),  # False
            "platforms": [q.get("platform") for q in state["decomposed_queries"]],  # ["general"]
            "queries": [
                {
                    "platform": q.get("platform"),
                    "topic": (q.get("topic") or "")[:50],
                }
                for q in state["decomposed_queries"]
            ],
        }
    )
    
    return state
```

---

### 4. Memory Retriever

**File:** `app/agents/conversational_graph/nodes/memory.py`

```python
async def __call__(self, state: ConversationState):
    """
    Retrieve relevant context from multi-layer memory.
    
    For "hey":
    - Input: user_id, conversation_id
    - Action: Retrieve working memory, user profile, RAG context
    - Result: retrieved_count=0 (no RAG for greetings), has_user_profile=True
    - Output: Retrieved memory context
    - Next: context_builder
    """
    
    user_id = state["user_id"]
    conversation_id = state["conversation_id"]
    current_input = state["current_input"]
    
    # Layer 1: Working memory from Redis
    working_memory = await memory_manager.get_working_memory(conversation_id)
    state["working_memory"].update(working_memory)
    
    # Layer 2: User profile
    user_profile = await memory_manager.get_user_profile(user_id)
    state["user_profile"] = user_profile
    
    # Layer 3 & 4: RAG retrieval based on intent
    intent_type = (state.get("current_intent") or {}).get("type", "artifact")
    
    if intent_type in ["artifact", "multi_platform"]:
        # For artifact generation, get specialized context
        platforms = self._get_platforms(state)
        rag_context = await rag_retriever.retrieve(
            query=current_input,
            user_id=user_id,
            conversation_id=conversation_id,
            context_types=["user_history", "knowledge", "examples"],
            platforms=platforms,
        )
    else:
        # For Q&A or clarification, lighter retrieval
        rag_context = await rag_retriever.retrieve(
            query=current_input,
            user_id=user_id,
            conversation_id=conversation_id,
            context_types=["knowledge"],
        )
    
    state["retrieved_memory"] = self._flatten_rag_results(rag_context)
    
    # Add to execution trace
    add_execution_trace(
        state,
        "memory_retriever",
        "completed",
        execution_time,
        metadata={
            "retrieved_count": len(state["retrieved_memory"]),
            "has_user_profile": state["user_profile"] is not None,
        }
    )
    
    return state
```

---

### 5. Router

**File:** `app/agents/conversational_graph/nodes/router.py`

```python
async def __call__(self, state: ConversationState):
    """
    Route conversation based on intent and context.
    
    For "hey":
    - Input: intent_type="qa"
    - Action: Determine route based on intent
    - Result: route="qa_response"
    - Output: Sets route in working_memory
    - Next: stream_aggregator (Q&A path)
    """
    
    intent = (state.get("current_intent") or {})
    intent_type = intent.get("type", "qa")
    entities = (intent.get("entities") or {})
    
    # Determine route based on intent
    if intent_type == "qa":
        route = "qa_response"  # ✅ "hey" takes this path
        needs_data_check = False
        reasoning = "Q&A intent detected, routing to direct response"
    
    elif intent_type == "clarification":
        route = "process_clarification"
        needs_data_check = False
        reasoning = "Processing user clarification"
    
    elif intent_type in ["artifact", "multi_platform"]:
        route = "artifact_generation"
        needs_data_check = True
        reasoning = "Content generation requested"
    
    else:
        route = "artifact_generation"
        needs_data_check = False
        reasoning = "Default routing to artifact generation"
    
    # Store route in working memory
    state["working_memory"]["route"] = route
    state["working_memory"]["needs_data_check"] = needs_data_check
    
    # Add to execution trace
    add_execution_trace(
        state,
        "router",
        "completed",
        execution_time,
        metadata={
            "route": route,  # "qa_response"
            "needs_data_check": needs_data_check,  # False
            "reasoning": reasoning,
        }
    )
    
    return state
```

---

### 6. Stream Aggregator (Q&A Path)

**File:** `app/agents/conversational_graph/nodes/aggregator.py`

```python
async def __call__(self, state: ConversationState):
    """
    Aggregate and format final response.
    
    For "hey":
    - Input: route="qa_response"
    - Action: Generate conversational response
    - Result: "Hello! How can I help you today?"
    - Output: final_response set
    - Next: memory_saver
    """
    
    route = (state.get("working_memory") or {}).get("route", "artifact_generation")
    
    if route == "qa_response":
        # Generate conversational Q&A response
        state["final_response"] = self._generate_qa_response(state)
        # For "hey" → "Hello! How can I help you today?"
        
        state["suggestions"] = [
            "Create a LinkedIn post",
            "Generate social media content",
            "Help with copywriting",
        ]
    
    elif route == "artifact_generation":
        # Format artifact generation response
        artifacts = state.get("artifacts", [])
        state["final_response"] = self._format_artifacts(artifacts)
        state["suggestions"] = self._get_suggestions(artifacts)
    
    # Add to execution trace
    add_execution_trace(
        state,
        "stream_aggregator",
        "completed",
        execution_time,
        metadata={
            "has_response": bool(state.get("final_response")),
            "response_preview": ((state.get("final_response") or "")[:100] + "..."
                               if state.get("final_response") and len(state.get("final_response") or "") > 100
                               else (state.get("final_response") or "")),
            "artifact_count": len(state.get("artifacts", [])),
            "event_count": len(events),
        }
    )
    
    return state

def _generate_qa_response(self, state: ConversationState) -> str:
    """Generate response for Q&A queries."""
    user_input = state.get("current_input", "").lower()
    
    if user_input in ["hi", "hello", "hey", "greetings"]:
        return (
            "Hello! I'm your AI content creation assistant. "
            "I can help you with:\n\n"
            "• Creating social media posts for LinkedIn, Instagram, Facebook, and more\n"
            "• Writing blog posts and articles\n"
            "• Generating SEO-optimized content\n"
            "• Managing your content creation tasks\n\n"
            "What would you like to create today?"
        )
    
    # ... other Q&A patterns
```

---

### 7. Memory Saver

**File:** `app/agents/conversational_graph/nodes/saver.py`

```python
async def __call__(self, state: ConversationState):
    """
    Save conversation state to persistent storage.
    
    For "hey":
    - Input: final_response set
    - Action: Save working memory, vector embeddings, message
    - Result: Conversation persisted
    - Output: saved=True
    - Next: END
    """
    
    conversation_id = state["conversation_id"]
    
    # 1. Save working memory to Redis
    await self._save_working_memory(state)
    
    # 2. Save to vector store if applicable
    if state.get("final_response") or state.get("artifacts"):
        await self._save_to_vector_store(state)
    
    # 3. Save message to database
    if state.get("final_response"):
        await self._save_message(state)
    
    # 4. Update token usage
    state["total_tokens_used"] += (state.get("working_memory") or {}).get(
        "context_tokens", 0
    )
    
    # 5. Calculate total execution time
    if state.get("execution_start_time"):
        total_time = datetime.utcnow() - state["execution_start_time"]
        state["working_memory"]["total_execution_ms"] = int(
            total_time.total_seconds() * 1000
        )
    
    # Add to execution trace
    add_execution_trace(
        state,
        "memory_saver",
        "completed",
        execution_time,
        metadata={
            "saved": True,
            "total_messages": len(state.get("messages", [])),
            "total_execution_ms": (state.get("working_memory") or {}).get("total_execution_ms", 0),
        }
    )
    
    return state
```

---

## 📊 Execution Flow Summary (for "hey")

| Step | Node | Duration | Action | Output |
|------|------|----------|--------|--------|
| 1 | `guardrail_check` | 50ms | Check content safety | ✅ Pass |
| 2 | `intent_classifier` | 450ms | LLM classification | intent_type="qa" |
| 3 | `query_decomposer` | 20ms | Create single query | platform="general" |
| 4 | `memory_retriever` | 120ms | Fetch user context | retrieved_count=0 |
| 5 | `context_builder` | 80ms | Build LLM context | context_tokens=150 |
| 6 | `router` | 10ms | Route by intent | route="qa_response" |
| 7 | `stream_aggregator` | 200ms | Generate response | "Hello! How can I help..." |
| 8 | `memory_saver` | 150ms | Save to DB/Redis | saved=True |

**Total:** ~1,080ms

**Nodes Skipped:**
- `follow_up_detector` (Q&A doesn't need follow-up detection)
- `follow_up_generator` (No follow-up needed)
- `data_checker` (Q&A doesn't need data checking)
- `multi_platform_orchestrator` (No artifact generation for Q&A)

---

## 🔍 Monitoring & Logging

### Execution Logger

**File:** `app/services/execution_monitor.py`

```python
class ExecutionLogger:
    """
    Background async logger that writes to PostgreSQL.
    
    Key Features:
    - Non-blocking writes via asyncio.Queue
    - ~1-2ms latency impact on main execution
    - Tracks all node outputs with elapsed time
    - Stores execution trace with metadata
    """
    
    def log_execution_started(self, execution_id, conversation_id, user_id, request_message):
        """Log execution start (non-blocking)."""
        self._queue.put_nowait({
            "type": "execution_started",
            "execution_id": execution_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "request_message": request_message,
            "timestamp": datetime.utcnow(),
        })
    
    def log_execution_completed(self, execution_id, status, duration_ms, 
                                response_message, execution_trace, error_summary, 
                                failed_nodes, start_time):
        """Log execution completion (non-blocking)."""
        self._queue.put_nowait({
            "type": "execution_completed",
            "execution_id": execution_id,
            "status": status,
            "duration_ms": duration_ms,
            "response_message": response_message,
            "execution_trace": execution_trace,
            "error_summary": error_summary,
            "failed_nodes": failed_nodes,
            "start_time": start_time,
        })
    
    def _build_node_outputs(self, execution_trace: list) -> dict:
        """
        Build node outputs from execution trace.
        
        Returns:
        {
            "guardrail_check": {
                "elapsed_ms": 50,      # Cumulative time from start
                "duration_ms": 50,     # Individual node time
                "status": "completed",
                "output": {
                    "guardrail_passed": True,
                    "violations": []
                }
            },
            "intent_classifier": {
                "elapsed_ms": 500,     # 50 + 450
                "duration_ms": 450,
                "status": "completed",
                "output": {
                    "intent_type": "qa",
                    "confidence": 0.9
                }
            }
        }
        """
        node_outputs = {}
        cumulative_time_ms = 0
        
        for trace in execution_trace:
            node_name = trace.get("node")
            node_duration_ms = trace.get("time_ms", 0)
            
            # Add to cumulative time
            cumulative_time_ms += node_duration_ms
            
            # Extract metadata
            metadata = trace.get("metadata", {})
            output_summary = {}
            
            if metadata:
                for key, value in metadata.items():
                    if key not in ['__internal__', 'raw_response', 'full_context']:
                        output_summary[key] = value
            
            node_outputs[node_name] = {
                "elapsed_ms": cumulative_time_ms,
                "duration_ms": node_duration_ms,
                "status": trace.get("status", "unknown"),
                "output": output_summary if output_summary else None,
            }
        
        return node_outputs
```

### Database Schema

**File:** `app/models/execution_log.py`

```python
class ExecutionLog(Base):
    """
    Execution log model for PostgreSQL.
    
    Stores:
    - Request/response messages
    - Full execution trace (all nodes)
    - Node outputs with timing
    - Failed nodes and errors
    - Total duration
    """
    
    __tablename__ = "execution_logs"
    
    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"))
    user_id = Column(String(255), nullable=False)
    
    request_message = Column(Text, nullable=False)    # "hey"
    response_message = Column(Text)                   # "Hello! How can I help..."
    
    status = Column(Enum(ExecutionStatus))            # COMPLETED
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    total_duration_ms = Column(Integer)               # 1080
    
    execution_trace = Column(JSON)                    # Full trace array
    node_outputs = Column(JSON)                       # Processed outputs
    
    error_summary = Column(Text)                      # Combined errors
    failed_nodes = Column(JSON)                       # List of failed nodes
```

---

## 🎯 Quick Reference

### Where to Start Reading

1. **API Entry:** `app/api/routes/chat_v2.py:42`
2. **Graph Execution:** `app/agents/conversational_graph/graph.py:224`
3. **Graph Structure:** `app/agents/conversational_graph/graph.py:37`
4. **First Node:** `app/agents/conversational_graph/nodes/intent.py:77`
5. **Router Logic:** `app/agents/conversational_graph/nodes/router.py:39`

### Key Files

- **Entry Point:** `app/api/routes/chat_v2.py`
- **Graph Definition:** `app/agents/conversational_graph/graph.py`
- **State Schema:** `app/agents/conversational_graph/state.py`
- **All Nodes:** `app/agents/conversational_graph/nodes/`
- **Monitoring:** `app/services/execution_monitor.py`

### Common Patterns

#### Safe State Access
```python
# ❌ UNSAFE - fails if value is None
state.get("key", {}).get("subkey")

# ✅ SAFE - handles None values
(state.get("key") or {}).get("subkey")
```

#### String Operations
```python
# ❌ UNSAFE - fails if value is None
value.get("text", "")[:100]

# ✅ SAFE - handles None values
(value.get("text") or "")[:100]
```

#### Adding Execution Trace
```python
add_execution_trace(
    state,
    "node_name",
    "completed",
    execution_time,
    metadata={
        "key": "value",
        "data": result_data,
    }
)
```

---

## 📚 Related Documentation

- **Monitoring System:** `MONITORING.md`
- **Project Memory:** `~/.claude/projects/.../memory/MEMORY.md`
- **API Documentation:** See FastAPI `/docs` endpoint
