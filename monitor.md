# Execution Monitoring Pipeline

## Overview

A comprehensive monitoring system that tracks node execution through the LangGraph conversational pipeline, stores logs in PostgreSQL, creates tree visualizations of execution flow, and optionally uses LLM for failure analysis.

---

## Current Issues Being Tracked

From the logs, these errors are occurring:
1. **`filter_metadata` error** in `data_checker.py` - VectorStore method signature mismatch
2. **`NoneType` error** in content generation - State values are `None` instead of empty dicts
3. **Embedding model error** - `models/embedding-001 is not found` in Pinecone searches

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Graph Execution                                  │
│  guardrail → intent → decomposer → memory → context → router → ...  │
│       │         │          │          │         │         │         │
│       ▼         ▼          ▼          ▼         ▼         ▼         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              ExecutionLogger (Background)                    │   │
│   │   - Captures node start/end/error events                    │   │
│   │   - Writes to PostgreSQL asynchronously                     │   │
│   │   - Minimal latency impact on main execution                │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL Tables                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │execution_logs│  │node_executions│ │  execution_edges         │   │
│  │ - id         │  │ - id          │ │  - parent_node_id        │   │
│  │ - conv_id    │  │ - exec_log_id │ │  - child_node_id         │   │
│  │ - user_id    │  │ - node_name   │ │  - edge_type (conditional)│  │
│  │ - status     │  │ - status      │ │                          │   │
│  │ - tree_json  │  │ - started_at  │ │                          │   │
│  └──────────────┘  │ - duration_ms │ └──────────────────────────┘   │
│                    │ - error_msg   │                                │
│                    └──────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   API Endpoints                                      │
│  GET /monitoring/executions/{id}/tree  → Tree visualization JSON    │
│  GET /monitoring/failures              → Failed executions list     │
│  GET /monitoring/stats/nodes           → Per-node success rates     │
│  POST /monitoring/analyze/{id}         → LLM failure analysis       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Database Models

### New File: `app/models/execution_log.py`

```python
from sqlalchemy import Column, String, Integer, DateTime, Text, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

class ExecutionStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some nodes failed

class NodeStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ExecutionLog(Base):
    """Root execution record for a conversation turn."""
    __tablename__ = "execution_logs"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    user_id = Column(String(255), nullable=False)
    user_input = Column(Text, nullable=False)
    status = Column(Enum(ExecutionStatus), default=ExecutionStatus.RUNNING)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    total_duration_ms = Column(Integer)
    tree_json = Column(JSON)  # Pre-computed tree visualization
    llm_analysis = Column(Text)  # Optional LLM failure analysis
    error_summary = Column(Text)

    # Relationships
    node_executions = relationship("NodeExecution", back_populates="execution_log")

class NodeExecution(Base):
    """Individual node execution within a graph run."""
    __tablename__ = "node_executions"

    id = Column(String(36), primary_key=True)
    execution_log_id = Column(String(36), ForeignKey("execution_logs.id"), nullable=False)
    node_name = Column(String(100), nullable=False)
    status = Column(Enum(NodeStatus), default=NodeStatus.PENDING)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    input_summary = Column(Text)  # Truncated input for debugging
    output_summary = Column(Text)  # Truncated output
    error_message = Column(Text)
    error_traceback = Column(Text)
    metadata = Column(JSON)  # Additional node-specific data
    sequence_order = Column(Integer)  # Order of execution

    # Relationships
    execution_log = relationship("ExecutionLog", back_populates="node_executions")

class ExecutionEdge(Base):
    """Edges between nodes (for tree visualization)."""
    __tablename__ = "execution_edges"

    id = Column(String(36), primary_key=True)
    execution_log_id = Column(String(36), ForeignKey("execution_logs.id"), nullable=False)
    parent_node_id = Column(String(36), ForeignKey("node_executions.id"))
    child_node_id = Column(String(36), ForeignKey("node_executions.id"), nullable=False)
    edge_type = Column(String(50))  # "sequential", "conditional_true", "conditional_false"
    condition_value = Column(String(100))  # e.g., "guardrail_passed=True"
```

### Migration: `migrations/versions/xxx_add_execution_monitoring.py`

Creates all three tables with proper indexes on `conversation_id`, `status`, `started_at`.

---

## Phase 2: Execution Logger Service

### New File: `app/services/execution_monitor.py`

```python
import asyncio
from datetime import datetime
from uuid import uuid4
from typing import Optional
import structlog
from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.execution_log import ExecutionLog, NodeExecution, ExecutionEdge, ExecutionStatus, NodeStatus
from app.core.database import async_session_factory

logger = structlog.get_logger(__name__)

class ExecutionLogger:
    """Background execution logger with minimal latency impact."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background log writer."""
        self._task = asyncio.create_task(self._process_queue())
        logger.info("ExecutionLogger started")

    async def stop(self):
        """Stop background writer gracefully."""
        if self._task:
            self._task.cancel()
            logger.info("ExecutionLogger stopped")

    async def _process_queue(self):
        """Process log events from queue."""
        while True:
            try:
                event = await self._queue.get()
                await self._write_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Failed to write log event", error=str(e))

    async def _write_event(self, event: dict):
        """Write single event to database."""
        async with async_session_factory() as session:
            try:
                if event["type"] == "execution_started":
                    log = ExecutionLog(
                        id=event["execution_id"],
                        conversation_id=event["conversation_id"],
                        user_id=event["user_id"],
                        user_input=event["user_input"][:1000],  # Truncate
                        status=ExecutionStatus.RUNNING,
                        started_at=event["timestamp"],
                    )
                    session.add(log)

                elif event["type"] == "node_started":
                    node = NodeExecution(
                        id=event["node_id"],
                        execution_log_id=event["execution_id"],
                        node_name=event["node_name"],
                        status=NodeStatus.RUNNING,
                        started_at=event["timestamp"],
                        sequence_order=event["sequence"],
                    )
                    session.add(node)

                elif event["type"] == "node_completed":
                    await session.execute(
                        update(NodeExecution)
                        .where(NodeExecution.id == event["node_id"])
                        .values(
                            status=NodeStatus.COMPLETED,
                            completed_at=event["timestamp"],
                            duration_ms=event["duration_ms"],
                            output_summary=event.get("output_summary", "")[:500],
                            metadata=event.get("metadata"),
                        )
                    )

                elif event["type"] == "node_failed":
                    await session.execute(
                        update(NodeExecution)
                        .where(NodeExecution.id == event["node_id"])
                        .values(
                            status=NodeStatus.FAILED,
                            completed_at=event["timestamp"],
                            duration_ms=event["duration_ms"],
                            error_message=event["error"][:1000],
                            error_traceback=event.get("traceback", "")[:2000],
                        )
                    )

                elif event["type"] == "execution_completed":
                    tree_json = await self._build_tree(session, event["execution_id"])
                    await session.execute(
                        update(ExecutionLog)
                        .where(ExecutionLog.id == event["execution_id"])
                        .values(
                            status=event["status"],
                            completed_at=event["timestamp"],
                            total_duration_ms=event["duration_ms"],
                            tree_json=tree_json,
                            error_summary=event.get("error_summary"),
                        )
                    )

                await session.commit()
            except Exception as e:
                logger.error("Database write failed", error=str(e), event_type=event["type"])
                await session.rollback()

    # Public API - non-blocking
    def log_execution_started(self, execution_id: str, conversation_id: str, user_id: str, user_input: str):
        self._queue.put_nowait({
            "type": "execution_started",
            "execution_id": execution_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_input": user_input,
            "timestamp": datetime.utcnow(),
        })

    def log_node_started(self, execution_id: str, node_id: str, node_name: str, sequence: int):
        self._queue.put_nowait({
            "type": "node_started",
            "execution_id": execution_id,
            "node_id": node_id,
            "node_name": node_name,
            "sequence": sequence,
            "timestamp": datetime.utcnow(),
        })

    def log_node_completed(self, node_id: str, duration_ms: int, output_summary: str = None, metadata: dict = None):
        self._queue.put_nowait({
            "type": "node_completed",
            "node_id": node_id,
            "duration_ms": duration_ms,
            "output_summary": output_summary,
            "metadata": metadata,
            "timestamp": datetime.utcnow(),
        })

    def log_node_failed(self, node_id: str, duration_ms: int, error: str, traceback: str = None):
        self._queue.put_nowait({
            "type": "node_failed",
            "node_id": node_id,
            "duration_ms": duration_ms,
            "error": error,
            "traceback": traceback,
            "timestamp": datetime.utcnow(),
        })

    def log_execution_completed(self, execution_id: str, status: ExecutionStatus, duration_ms: int, error_summary: str = None):
        self._queue.put_nowait({
            "type": "execution_completed",
            "execution_id": execution_id,
            "status": status,
            "duration_ms": duration_ms,
            "error_summary": error_summary,
            "timestamp": datetime.utcnow(),
        })

    async def _build_tree(self, session: AsyncSession, execution_id: str) -> dict:
        """Build tree JSON from node executions."""
        result = await session.execute(
            select(NodeExecution)
            .where(NodeExecution.execution_log_id == execution_id)
            .order_by(NodeExecution.sequence_order)
        )
        nodes = result.scalars().all()

        tree = {
            "execution_id": execution_id,
            "nodes": [],
            "edges": [],
            "summary": {
                "total_nodes": len(nodes),
                "completed": sum(1 for n in nodes if n.status == NodeStatus.COMPLETED),
                "failed": sum(1 for n in nodes if n.status == NodeStatus.FAILED),
                "skipped": sum(1 for n in nodes if n.status == NodeStatus.SKIPPED),
            },
            "failed_nodes": [n.node_name for n in nodes if n.status == NodeStatus.FAILED],
        }

        prev_node_id = None
        for node in nodes:
            tree["nodes"].append({
                "id": node.id,
                "name": node.node_name,
                "status": node.status.value,
                "duration_ms": node.duration_ms,
                "error": node.error_message,
            })

            if prev_node_id:
                tree["edges"].append({
                    "from": prev_node_id,
                    "to": node.id,
                    "type": "sequential",
                })
            prev_node_id = node.id

        return tree

# Global singleton
execution_logger = ExecutionLogger()
```

---

## Phase 3: Integration Points

### Modify: `app/agents/conversational_graph/graph.py`

```python
from app.services.execution_monitor import execution_logger, ExecutionStatus
from uuid import uuid4
import time

async def invoke_conversation(
    conversation_id: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    config: dict | None = None,
) -> ConversationState:
    """Invoke the conversational graph with monitoring."""
    start_time = time.time()
    execution_id = str(uuid4())

    # Log execution start
    execution_logger.log_execution_started(
        execution_id=execution_id,
        conversation_id=conversation_id,
        user_id=user_id,
        user_input=user_input,
    )

    # Store execution_id in state for nodes to access
    initial_state = create_initial_state(conversation_id, user_id, thread_id, user_input)
    initial_state["execution_id"] = execution_id
    initial_state["node_sequence"] = 0

    try:
        result = await graph.ainvoke(initial_state, graph_config)

        # Determine final status
        errors = result.get("errors", [])
        status = ExecutionStatus.COMPLETED if not errors else ExecutionStatus.PARTIAL

        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=status,
            duration_ms=int((time.time() - start_time) * 1000),
            error_summary="; ".join(errors) if errors else None,
        )

        return result

    except Exception as e:
        execution_logger.log_execution_completed(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            duration_ms=int((time.time() - start_time) * 1000),
            error_summary=str(e),
        )
        raise
```

### Modify: `app/agents/conversational_graph/state.py`

Enhance `add_execution_trace` to log to monitor:

```python
def add_execution_trace(
    state: ConversationState,
    node: str,
    status: str,
    execution_time_ms: int,
    error: str | None = None,
    metadata: dict | None = None,
):
    """Add execution trace entry and log to monitor."""
    # Existing trace logic
    trace_entry = {
        "node": node,
        "status": status,
        "execution_time_ms": execution_time_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if error:
        trace_entry["error"] = error
    if metadata:
        trace_entry["metadata"] = metadata

    state["execution_trace"].append(trace_entry)

    # NEW: Monitoring integration
    try:
        from app.services.execution_monitor import execution_logger

        execution_id = state.get("execution_id")
        if execution_id:
            node_id = f"{execution_id}_{node}"

            if status == "completed":
                execution_logger.log_node_completed(
                    node_id=node_id,
                    duration_ms=execution_time_ms,
                    metadata=metadata,
                )
            elif status == "failed":
                execution_logger.log_node_failed(
                    node_id=node_id,
                    duration_ms=execution_time_ms,
                    error=error or "Unknown error",
                )
    except Exception:
        pass  # Don't let monitoring break main flow
```

### Add to each node's `__call__` method

Add node start logging at the beginning of each node:

```python
async def __call__(self, state: ConversationState) -> ConversationState:
    start_time = time.time()

    # NEW: Log node start
    execution_id = state.get("execution_id")
    if execution_id:
        from app.services.execution_monitor import execution_logger
        state["node_sequence"] = state.get("node_sequence", 0) + 1
        node_id = f"{execution_id}_{self.__class__.__name__}"
        execution_logger.log_node_started(
            execution_id=execution_id,
            node_id=node_id,
            node_name=self.__class__.__name__,
            sequence=state["node_sequence"],
        )

    # ... existing node logic ...
```

---

## Phase 4: API Endpoints

### New File: `app/api/routes/monitoring.py`

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

from app.core.database import get_session
from app.models.execution_log import ExecutionLog, NodeExecution, ExecutionStatus, NodeStatus

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/executions/{execution_id}/tree")
async def get_execution_tree(
    execution_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get tree visualization of execution flow."""
    result = await session.execute(
        select(ExecutionLog).where(ExecutionLog.id == execution_id)
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return {
        "execution_id": execution.id,
        "conversation_id": execution.conversation_id,
        "user_input": execution.user_input,
        "status": execution.status.value,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "total_duration_ms": execution.total_duration_ms,
        "tree": execution.tree_json,
        "error_summary": execution.error_summary,
        "llm_analysis": execution.llm_analysis,
    }

@router.get("/conversations/{conversation_id}/executions")
async def list_conversation_executions(
    conversation_id: str,
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List all executions for a conversation."""
    result = await session.execute(
        select(ExecutionLog)
        .where(ExecutionLog.conversation_id == conversation_id)
        .order_by(ExecutionLog.started_at.desc())
        .limit(limit)
    )
    executions = result.scalars().all()

    return [
        {
            "id": e.id,
            "status": e.status.value,
            "started_at": e.started_at.isoformat() if e.started_at else None,
            "total_duration_ms": e.total_duration_ms,
            "error_summary": e.error_summary,
            "failed_nodes": e.tree_json.get("failed_nodes", []) if e.tree_json else [],
        }
        for e in executions
    ]

@router.get("/failures")
async def list_failures(
    limit: int = Query(50, le=200),
    user_id: str = None,
    hours: int = Query(24, le=168),
    session: AsyncSession = Depends(get_session),
):
    """List recent failed executions."""
    since = datetime.utcnow() - timedelta(hours=hours)

    query = select(ExecutionLog).where(
        and_(
            ExecutionLog.status.in_([ExecutionStatus.FAILED, ExecutionStatus.PARTIAL]),
            ExecutionLog.started_at >= since,
        )
    )

    if user_id:
        query = query.where(ExecutionLog.user_id == user_id)

    query = query.order_by(ExecutionLog.started_at.desc()).limit(limit)

    result = await session.execute(query)
    executions = result.scalars().all()

    return [
        {
            "id": e.id,
            "conversation_id": e.conversation_id,
            "user_id": e.user_id,
            "user_input": e.user_input[:100] + "..." if len(e.user_input) > 100 else e.user_input,
            "status": e.status.value,
            "started_at": e.started_at.isoformat(),
            "total_duration_ms": e.total_duration_ms,
            "error_summary": e.error_summary,
            "failed_nodes": e.tree_json.get("failed_nodes", []) if e.tree_json else [],
        }
        for e in executions
    ]

@router.get("/stats/nodes")
async def get_node_stats(
    days: int = Query(7, le=30),
    session: AsyncSession = Depends(get_session),
):
    """Get per-node success/failure statistics."""
    since = datetime.utcnow() - timedelta(days=days)

    result = await session.execute(
        select(
            NodeExecution.node_name,
            NodeExecution.status,
            func.count(NodeExecution.id).label("count"),
            func.avg(NodeExecution.duration_ms).label("avg_duration_ms"),
        )
        .where(NodeExecution.started_at >= since)
        .group_by(NodeExecution.node_name, NodeExecution.status)
    )
    rows = result.all()

    # Aggregate by node
    stats = {}
    for row in rows:
        node_name = row.node_name
        if node_name not in stats:
            stats[node_name] = {
                "node_name": node_name,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "avg_duration_ms": 0,
            }

        stats[node_name]["total"] += row.count
        if row.status == NodeStatus.COMPLETED:
            stats[node_name]["completed"] = row.count
            stats[node_name]["avg_duration_ms"] = round(row.avg_duration_ms or 0, 2)
        elif row.status == NodeStatus.FAILED:
            stats[node_name]["failed"] = row.count

    # Calculate success rate
    for node in stats.values():
        if node["total"] > 0:
            node["success_rate"] = round(node["completed"] / node["total"] * 100, 1)
        else:
            node["success_rate"] = 0

    return sorted(stats.values(), key=lambda x: x["success_rate"])

@router.post("/executions/{execution_id}/analyze")
async def analyze_failure(
    execution_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Use LLM to analyze a failed execution."""
    from app.core.llm_clients import llm_client, LLMMessage
    import json

    # Get execution with nodes
    result = await session.execute(
        select(ExecutionLog).where(ExecutionLog.id == execution_id)
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Get failed nodes
    node_result = await session.execute(
        select(NodeExecution)
        .where(
            and_(
                NodeExecution.execution_log_id == execution_id,
                NodeExecution.status == NodeStatus.FAILED,
            )
        )
    )
    failed_nodes = node_result.scalars().all()

    if not failed_nodes:
        return {"analysis": "No failed nodes found in this execution."}

    # Build analysis prompt
    prompt = f"""Analyze this failed graph execution and suggest fixes:

User Input: {execution.user_input}
Total Duration: {execution.total_duration_ms}ms
Status: {execution.status.value}

Failed Nodes:
{chr(10).join(f"- {n.node_name}: {n.error_message}" for n in failed_nodes)}

Full Execution Tree:
{json.dumps(execution.tree_json, indent=2) if execution.tree_json else "Not available"}

Provide:
1. Root cause analysis
2. Which node failed first and why
3. Suggested code fixes
4. Prevention strategies
"""

    response = await llm_client.generate_for_task(
        task="failure_analysis",
        messages=[
            LLMMessage(role="system", content="You are a debugging expert analyzing LangGraph execution failures. Be concise and actionable."),
            LLMMessage(role="user", content=prompt),
        ],
    )

    # Store analysis
    execution.llm_analysis = response.content
    await session.commit()

    return {"analysis": response.content}
```

### Register in `app/api/main.py`

```python
from app.api.routes.monitoring import router as monitoring_router

app.include_router(monitoring_router, prefix=settings.api_v1_prefix)

# Start execution logger on startup
@app.on_event("startup")
async def start_execution_logger():
    from app.services.execution_monitor import execution_logger
    await execution_logger.start()

@app.on_event("shutdown")
async def stop_execution_logger():
    from app.services.execution_monitor import execution_logger
    await execution_logger.stop()
```

---

## Phase 5: Tree Visualization JSON Format

Example response from `GET /monitoring/executions/{id}/tree`:

```json
{
  "execution_id": "abc-123",
  "conversation_id": "conv-456",
  "user_input": "Create a LinkedIn post about AI",
  "status": "partial",
  "started_at": "2026-02-13T10:00:00Z",
  "completed_at": "2026-02-13T10:00:05Z",
  "total_duration_ms": 5234,
  "tree": {
    "execution_id": "abc-123",
    "summary": {
      "total_nodes": 12,
      "completed": 10,
      "failed": 1,
      "skipped": 1
    },
    "nodes": [
      {
        "id": "abc-123_GuardrailNode",
        "name": "GuardrailNode",
        "status": "completed",
        "duration_ms": 120,
        "error": null
      },
      {
        "id": "abc-123_IntentClassifierNode",
        "name": "IntentClassifierNode",
        "status": "completed",
        "duration_ms": 450,
        "error": null
      },
      {
        "id": "abc-123_MultiPlatformOrchestratorNode",
        "name": "MultiPlatformOrchestratorNode",
        "status": "failed",
        "duration_ms": 2100,
        "error": "'NoneType' object has no attribute 'get'"
      }
    ],
    "edges": [
      {"from": "abc-123_GuardrailNode", "to": "abc-123_IntentClassifierNode", "type": "sequential"},
      {"from": "abc-123_IntentClassifierNode", "to": "abc-123_QueryDecomposerNode", "type": "sequential"}
    ],
    "failed_nodes": ["MultiPlatformOrchestratorNode"]
  },
  "error_summary": "'NoneType' object has no attribute 'get'",
  "llm_analysis": null
}
```

---

## Phase 6: LLM Analysis Response

Example response from `POST /monitoring/executions/{id}/analyze`:

```json
{
  "analysis": "## Root Cause Analysis\n\nThe `MultiPlatformOrchestratorNode` failed with `'NoneType' object has no attribute 'get'`. This indicates that a dictionary value expected to exist was `None`.\n\n## First Failure Point\n\nThe error occurred in `_synthesize_results()` when accessing `state.get('content', {}).get('text')`. When `content` is explicitly `None`, the default `{}` is not used.\n\n## Suggested Fix\n\n```python\n# Change from:\ncontent = state.get('content', {})\n\n# To:\ncontent = state.get('content') or {}\n```\n\n## Prevention\n\n1. Always use `state.get('key') or {}` pattern\n2. Add validation at node entry points\n3. Initialize all state keys with empty defaults"
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `app/models/execution_log.py` | Database models (ExecutionLog, NodeExecution, ExecutionEdge) |
| `app/services/execution_monitor.py` | Background logger with async queue |
| `app/api/routes/monitoring.py` | REST API endpoints |
| `migrations/versions/xxx_add_execution_monitoring.py` | Database migration |

## Files to Modify

| File | Changes |
|------|---------|
| `app/agents/conversational_graph/graph.py` | Add execution start/end logging |
| `app/agents/conversational_graph/state.py` | Enhance `add_execution_trace` to log to monitor |
| `app/api/main.py` | Register monitoring router, start/stop logger |
| Each node in `nodes/` | Add node start logging call |

---

## Verification Steps

1. **Run migration**: `alembic upgrade head`
2. **Start server**: `uvicorn app.api.main:app --reload`
3. **Send chat request**: `POST /api/v1/chat/v2`
4. **View execution tree**: `GET /api/v1/monitoring/executions/{id}/tree`
5. **Check failures**: `GET /api/v1/monitoring/failures`
6. **View node stats**: `GET /api/v1/monitoring/stats/nodes`
7. **Analyze failure**: `POST /api/v1/monitoring/executions/{id}/analyze`

---

## Quick Fixes Needed First

Before implementing monitoring, fix the underlying errors:

### 1. Fix `filter_metadata` error in `app/services/data_checker.py` or `memory_manager.py`

```python
# Change from:
await memory_manager.retrieve_user_content(
    ...,
    filter_metadata={"platform": platform},
)

# To:
await memory_manager.retrieve_user_content(
    ...,
    platform=platform,
)
```

### 2. Fix `NoneType` error in content generation

```python
# In app/agents/elves/social_media_manager/orchestrator.py
# Change from:
content = state.get("content", {})

# To:
content = state.get("content") or {}
```
