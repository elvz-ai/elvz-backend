"""
AI Assistant Elf - Manages tasks, schedules, communication, and research.
Uses dynamic DAG based on request type.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Optional

import structlog
from langgraph.graph import END, StateGraph

from app.agents.elves.base import BaseElf
from app.core.llm_clients import LLMMessage, llm_client
from app.tools.registry import tool_registry

logger = structlog.get_logger(__name__)


class TaskManager:
    """Manages tasks, priorities, and schedules."""
    
    name = "task_manager"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Manage tasks."""
        request = state.get("user_request", {})
        message = request.get("message", request.get("topic", ""))
        
        system_prompt = """You are an expert productivity assistant.
Analyze the request and create or organize tasks.

Respond with JSON:
{
    "tasks": [
        {
            "id": "task_1",
            "title": "task title",
            "description": "detailed description",
            "priority": "high/medium/low",
            "due_date": "YYYY-MM-DD or null",
            "status": "pending"
        }
    ],
    "schedule_suggestions": [
        {"task_id": "task_1", "suggested_time": "YYYY-MM-DD HH:MM", "reason": "why"}
    ]
}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Request: {message}"),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {
                "tasks": result.get("tasks", []),
                "schedule_suggestions": result.get("schedule_suggestions", []),
            }
        except:
            return {"tasks": [], "schedule_suggestions": []}


class EmailManager:
    """Drafts emails and manages communication."""
    
    name = "email_manager"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Draft or manage emails."""
        request = state.get("user_request", {})
        message = request.get("message", request.get("topic", ""))
        email_context = request.get("email_context", {})
        
        system_prompt = """You are an expert email writer.
Draft professional, clear, and effective emails.

Guidelines:
- Clear subject line
- Professional greeting
- Concise body (respect reader's time)
- Clear call-to-action
- Professional sign-off

Respond with JSON:
{
    "email": {
        "subject": "email subject",
        "body": "email body with proper formatting",
        "tone": "detected/applied tone"
    },
    "alternatives": [
        {"subject": "alt subject", "body": "alt body (shorter/longer/different tone)"}
    ]
}"""

        user_prompt = f"""Draft an email for: {message}

Context: {email_context if email_context else 'General email'}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {
                "email": result.get("email", {}),
                "email_alternatives": result.get("alternatives", []),
            }
        except:
            return {"email": {"subject": "", "body": response.content}}


class Researcher:
    """Conducts research on topics."""
    
    name = "researcher"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Conduct research."""
        request = state.get("user_request", {})
        topic = request.get("topic", request.get("message", ""))
        depth = request.get("depth", "moderate")  # brief, moderate, comprehensive
        
        # Use web search tool if available
        search_results = await self._web_search(topic)
        
        system_prompt = f"""You are an expert research analyst.
Provide well-organized research findings on the given topic.
Research depth: {depth}

Include:
- Executive summary
- Key findings
- Supporting evidence/sources
- Conclusions/recommendations"""

        user_prompt = f"""Research topic: {topic}

Web search results:
{search_results}

Respond with JSON:
{{
    "summary": "executive summary",
    "key_findings": ["finding 1", "finding 2"],
    "details": "detailed analysis",
    "sources": ["source 1", "source 2"],
    "recommendations": ["recommendation 1"]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_smart(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"research": result}
        except:
            return {"research": {"summary": response.content}}
    
    async def _web_search(self, query: str) -> str:
        """Perform web search if tool available."""
        try:
            from app.tools.research_tools import WebSearchInput
            
            tool = tool_registry.get("web_search")
            if tool:
                input_data = WebSearchInput(query=query, num_results=5)
                result = await tool.execute(input_data)
                
                if result.success:
                    results = result.data.get("results", [])
                    formatted = []
                    for r in results:
                        formatted.append(f"- {r.get('title', '')}: {r.get('snippet', '')}")
                    return "\n".join(formatted)
        except Exception as e:
            logger.warning("Web search failed", error=str(e))
        
        return "No web search results available."


class MeetingAssistant:
    """Processes meeting notes and extracts action items."""
    
    name = "meeting_assistant"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Process meeting notes."""
        request = state.get("user_request", {})
        notes = request.get("meeting_notes", request.get("message", ""))
        
        system_prompt = """You are an expert meeting facilitator.
Analyze meeting notes and extract key information.

Respond with JSON:
{
    "summary": "concise meeting summary",
    "key_decisions": ["decision 1", "decision 2"],
    "action_items": [
        {
            "task": "action item description",
            "assignee": "person responsible or 'TBD'",
            "due_date": "date or 'TBD'"
        }
    ],
    "follow_ups": ["follow up item 1"],
    "next_steps": "recommended next steps"
}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Meeting notes:\n{notes}"),
        ]
        
        response = await llm_client.generate(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"meeting_summary": result}
        except:
            return {"meeting_summary": {"summary": response.content}}


class DocumentGenerator:
    """Generates formatted documents."""
    
    name = "document_generator"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Generate document."""
        request = state.get("user_request", {})
        doc_type = request.get("document_type", "report")
        topic = request.get("topic", request.get("message", ""))
        key_points = request.get("key_points", [])
        
        # Use research if available
        research = state.get("research", {})
        
        system_prompt = f"""You are an expert document writer.
Create a well-structured {doc_type}.

Document types:
- report: formal, data-driven, objective
- proposal: persuasive, solution-focused
- summary: concise, key-points focused
- memo: brief, action-oriented

Respond with JSON:
{{
    "title": "document title",
    "content": "full document in markdown",
    "sections": ["section 1 title", "section 2 title"]
}}"""

        user_prompt = f"""Create a {doc_type} about: {topic}

Key points to include: {key_points}

Additional context: {research.get('summary', 'None')}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_smart(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"document": result}
        except:
            return {"document": {"title": topic, "content": response.content}}


class AIAssistantElf(BaseElf):
    """
    AI Assistant Elf - Manages tasks, schedules, communication, and research.
    
    Mini-Agents:
    1. Task Manager - Organizes tasks and schedules
    2. Email Manager - Drafts and manages emails
    3. Researcher - Conducts research
    4. Meeting Assistant - Processes meeting notes
    5. Document Generator - Creates formatted documents
    
    Workflow: Dynamic DAG based on request type
    """
    
    name = "ai_assistant"
    description = "Manages tasks, schedules, communication, and research"
    version = "1.0"
    
    def __init__(self):
        self.task_manager = TaskManager()
        self.email_manager = EmailManager()
        self.researcher = Researcher()
        self.meeting_assistant = MeetingAssistant()
        self.document_generator = DocumentGenerator()
        
        super().__init__()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("classify_request", self._classify_request)
        workflow.add_node("task_management", self._run_task_management)
        workflow.add_node("email_drafting", self._run_email_drafting)
        workflow.add_node("research", self._run_research)
        workflow.add_node("meeting_processing", self._run_meeting_processing)
        workflow.add_node("document_generation", self._run_document_generation)
        workflow.add_node("synthesize", self._synthesize_results)
        
        workflow.set_entry_point("classify_request")
        
        # Dynamic routing based on request type
        workflow.add_conditional_edges(
            "classify_request",
            self._route_to_agent,
            {
                "task": "task_management",
                "email": "email_drafting",
                "research": "research",
                "meeting": "meeting_processing",
                "document": "document_generation",
                "general": "research",  # Default to research for general queries
            }
        )
        
        # All paths lead to synthesize
        workflow.add_edge("task_management", "synthesize")
        workflow.add_edge("email_drafting", "synthesize")
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("meeting_processing", "synthesize")
        workflow.add_conditional_edges(
            "document_generation",
            self._needs_research_for_doc,
            {"yes": "research", "no": "synthesize"}
        )
        
        workflow.add_edge("synthesize", END)
        
        self._workflow = workflow.compile()
    
    async def execute(self, request: dict, context: dict) -> dict:
        """Execute assistant workflow."""
        start_time = time.time()
        
        initial_state = {
            "user_request": request,
            "context": context,
            "request_type": None,
            "tasks": [],
            "email": None,
            "research": None,
            "meeting_summary": None,
            "document": None,
            "execution_trace": [],
            "errors": [],
        }
        
        logger.info("AI Assistant executing")
        
        try:
            final_state = await self._workflow.ainvoke(initial_state)
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._build_response(final_state, execution_time_ms)
        except Exception as e:
            logger.error("Assistant workflow failed", error=str(e))
            return {"error": str(e), "response": "I encountered an error processing your request."}
    
    async def _classify_request(self, state: dict) -> dict:
        """Classify the type of request."""
        request = state.get("user_request", {})
        message = request.get("message", request.get("topic", "")).lower()
        
        # Keyword-based classification
        if any(kw in message for kw in ["task", "todo", "reminder", "schedule", "deadline"]):
            request_type = "task"
        elif any(kw in message for kw in ["email", "draft", "reply", "message", "send"]):
            request_type = "email"
        elif any(kw in message for kw in ["research", "find", "search", "look up", "investigate"]):
            request_type = "research"
        elif any(kw in message for kw in ["meeting", "notes", "action items", "follow up"]):
            request_type = "meeting"
        elif any(kw in message for kw in ["document", "report", "proposal", "memo", "write"]):
            request_type = "document"
        else:
            request_type = "general"
        
        state["request_type"] = request_type
        state["execution_trace"].append({"agent": "classifier", "result": request_type})
        
        return state
    
    def _route_to_agent(self, state: dict) -> str:
        """Route to appropriate agent based on classification."""
        return state.get("request_type", "general")
    
    def _needs_research_for_doc(self, state: dict) -> str:
        """Check if document needs research first."""
        request = state.get("user_request", {})
        return "yes" if request.get("needs_research") and not state.get("research") else "no"
    
    async def _run_task_management(self, state: dict) -> dict:
        result = await self.task_manager.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "task_manager", "status": "completed"})
        return state
    
    async def _run_email_drafting(self, state: dict) -> dict:
        result = await self.email_manager.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "email_manager", "status": "completed"})
        return state
    
    async def _run_research(self, state: dict) -> dict:
        result = await self.researcher.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "researcher", "status": "completed"})
        return state
    
    async def _run_meeting_processing(self, state: dict) -> dict:
        result = await self.meeting_assistant.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "meeting_assistant", "status": "completed"})
        return state
    
    async def _run_document_generation(self, state: dict) -> dict:
        result = await self.document_generator.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "document_generator", "status": "completed"})
        return state
    
    async def _synthesize_results(self, state: dict) -> dict:
        """Synthesize results based on request type."""
        request_type = state.get("request_type", "general")
        
        if request_type == "task":
            state["final_output"] = {
                "type": "task_management",
                "tasks": state.get("tasks", []),
                "schedule_suggestions": state.get("schedule_suggestions", []),
            }
        elif request_type == "email":
            state["final_output"] = {
                "type": "email",
                "draft": state.get("email", {}),
                "alternatives": state.get("email_alternatives", []),
            }
        elif request_type == "research":
            state["final_output"] = {
                "type": "research",
                "findings": state.get("research", {}),
            }
        elif request_type == "meeting":
            state["final_output"] = {
                "type": "meeting_summary",
                **state.get("meeting_summary", {}),
            }
        elif request_type == "document":
            state["final_output"] = {
                "type": "document",
                **state.get("document", {}),
            }
        else:
            # General response
            state["final_output"] = {
                "type": "general",
                "response": state.get("research", {}).get("summary", "I can help with that."),
            }
        
        return state
    
    def _build_response(self, state: dict, execution_time_ms: int) -> dict:
        final = state.get("final_output", {})
        return {
            **final,
            "execution_time_ms": execution_time_ms,
            "request_type": state.get("request_type"),
        }

