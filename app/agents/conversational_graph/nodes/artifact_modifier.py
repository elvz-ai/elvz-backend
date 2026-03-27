"""
Artifact Modifier Node — Tool-Calling Modification System.

Replaces full elf re-generation for modification requests with targeted
tool calls. The LLM decides which tools to call based on the user's
request, and each tool maps to an internal service.

Tools:
- update_text: AI re-optimization via SocialMediaManagerElf
- update_hashtags: AI-generated or direct hashtag replacement
- update_schedule: Schedule parsing + direct update
- regenerate_image: Image improvement via VisualAgent
- generate_image: Fresh image generation via VisualAgent
- direct_edit: Direct field value assignment
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.llm_clients import LLMMessage, llm_client
from app.services.artifact_service import artifact_service

logger = structlog.get_logger(__name__)


def _safe_content(artifact) -> dict:
    """Safely get artifact content as a dict, handling JSON strings."""
    content = artifact.content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {}
    return content or {}


# ── Tool Definitions (schema the LLM sees) ─────────────────────────────

MODIFICATION_TOOLS = [
    {
        "name": "update_text",
        "description": "Rewrite or optimize the post text using AI. Use when the user wants to change tone, length, style, or content of the post text.",
        "parameters": {
            "prompt": "string (required) — instruction for how to modify the text"
        },
    },
    {
        "name": "update_hashtags",
        "description": "Update the hashtags on the post. Use when the user wants different, more, or fewer hashtags.",
        "parameters": {
            "hashtags": "array of strings (optional) — explicit hashtag list if user provides them",
            "prompt": "string (optional) — instruction for AI to generate hashtags (e.g., 'trending hashtags for AI'). If provided, AI generates them.",
        },
    },
    {
        "name": "update_schedule",
        "description": "Change the posting schedule/time.",
        "parameters": {
            "prompt": "string (required) — natural language scheduling instruction (e.g., 'tomorrow morning', 'next Monday at 9am EST')",
        },
    },
    {
        "name": "regenerate_image",
        "description": "Regenerate or improve the post's existing image. Use when the user wants to modify the current image.",
        "parameters": {
            "prompt": "string (required) — instruction for image modification (e.g., 'make it brighter', 'add watermark')",
        },
    },
    {
        "name": "generate_image",
        "description": "Generate a new AI image for the post from scratch. Use when the post has no image and user wants one.",
        "parameters": {
            "prompt": "string (required) — description of what image to generate",
        },
    },
    {
        "name": "direct_edit",
        "description": "Directly set specific field values without AI processing. Use for explicit value assignments like 'change the CTA to Visit our website'.",
        "parameters": {
            "text": "string (optional) — new post text",
            "hashtags": "array of strings (optional) — new hashtag list",
            "cta": "string (optional) — new call-to-action text",
            "hook": "string (optional) — new hook/opener text",
        },
    },
]


# ── Tool Planning Prompt ────────────────────────────────────────────────

TOOL_PLANNING_PROMPT = """You are modifying a social media post based on the user's request.
Decide which tools to call and with what parameters.

Current post:
- Platform: {platform}
- Text: {text}
- Hashtags: {hashtags}
- Schedule: {schedule}
- Has image: {has_image}
- Image prompt: {image_prompt}

User request: "{user_message}"

Available tools:
{tools_json}

Rules:
- Use the minimum number of tools needed to satisfy the request
- For compound requests (e.g., "make it shorter and add trending hashtags"), return multiple tool calls
- If user gives explicit values (e.g., "change hashtags to #AI #ML"), use direct_edit
- If user wants AI-driven changes (e.g., "better hashtags"), use the AI tool (update_hashtags with prompt)
- For text rewrites, use update_text
- For image changes, use regenerate_image (if image exists) or generate_image (if no image)
- Always return valid JSON

Return JSON with tool calls and a friendly response message:
{{
  "tool_calls": [{{"tool": "tool_name", "parameters": {{...}}}}],
  "response_message": "A brief, friendly 1-2 sentence confirmation of what you're changing. Be conversational and natural. No bullet points or markdown. If calling update_text, leave response_message empty — the content agent will provide it."
}}

If the user is asking a question (not modifying), return:
{{"no_tools": true, "response": "your conversational answer"}}"""


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a single tool execution."""
    tool_name: str
    success: bool
    message: str = ""
    error: str = ""
    updates: dict = None

    def __post_init__(self):
        if self.updates is None:
            self.updates = {}


# ── Tool Executor ───────────────────────────────────────────────────────

class ToolExecutor:
    """Executes modification tools against internal services using mini-agents directly."""

    def __init__(self):
        self._visual_agent = None

    async def _get_visual_agent(self):
        if self._visual_agent is None:
            from app.agents.elves.social_media_manager.mini_agents.visual import VisualAgent
            self._visual_agent = VisualAgent()
        return self._visual_agent

    async def execute_tools(
        self,
        tool_calls: list[dict],
        artifact,
        user_id: str,
    ) -> list[ToolResult]:
        """Execute tool calls sequentially (order matters for compound edits)."""
        # Track original ID for refresh — artifact may become a version row after first tool
        original_id = artifact.parent_artifact_id or artifact.id

        results = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("tool", "unknown")
            params = call.get("parameters") or {}
            if isinstance(params, str):
                params = {"prompt": params}

            result = await self._execute_single(tool_name, params, artifact, user_id)
            results.append(result)

            # Refresh to latest version after each tool
            if result.success:
                refreshed = await artifact_service.get_current_artifact(original_id)
                if refreshed:
                    artifact = refreshed

        return results

    async def _execute_single(
        self,
        tool_name: str,
        params: dict,
        artifact,
        user_id: str,
    ) -> ToolResult:
        """Execute a single tool."""
        handler = {
            "update_text": self._execute_update_text,
            "update_hashtags": self._execute_update_hashtags,
            "update_schedule": self._execute_update_schedule,
            "regenerate_image": self._execute_regenerate_image,
            "generate_image": self._execute_generate_image,
            "direct_edit": self._execute_direct_edit,
        }.get(tool_name)

        if not handler:
            logger.warning("Unknown tool requested", tool=tool_name)
            return ToolResult(tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}")

        try:
            return await handler(params, artifact, user_id)
        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))
            return ToolResult(tool_name=tool_name, success=False, error=str(e))

    # ── Individual Tool Implementations ─────────────────────────────

    async def _execute_update_text(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Re-optimize text via ContentAgent directly (no planner/persona/optimization)."""
        prompt = params.get("prompt", "")
        current_content = _safe_content(artifact)
        current_text = current_content.get("text") or current_content.get("post_text") or ""

        from app.agents.elves.social_media_manager.mini_agents.content import ContentAgent

        content_agent = ContentAgent()
        agent_state = {
            "user_request": {
                "platform": artifact.platform or "linkedin",
                "topic": current_text[:100],
                "content_type": "thought_leadership",
                "goals": ["engagement"],
            },
            "previous_content": current_text,
            "modification_feedback": prompt,
            "plan": {
                "content_strategy": f"Modify the existing post based on user instruction: {prompt}",
            },
        }
        agent_context = {
            "user_id": user_id,
        }

        result = await content_agent.execute(agent_state, agent_context)
        content = result.get("content") or {}

        # Normalize content structure — ContentAgent may return content as a string
        if isinstance(content, str):
            text = content
        else:
            text = content.get("post_text") or content.get("text") or ""
        if not text:
            return ToolResult(tool_name="update_text", success=False, error="Optimization produced no content")

        updates = {
            "text": text,
            "post_text": text,
        }
        if isinstance(content, dict):
            if content.get("hook"):
                updates["hook"] = content["hook"]
            if content.get("cta"):
                updates["cta"] = content["cta"]

        await artifact_service.update_artifact_content(
            artifact_id=artifact.id,
            updates=updates,
            source="regeneration",
            prompt=prompt,
        )

        # Use ContentAgent's natural response_message if available
        response_msg = ""
        if isinstance(content, dict):
            response_msg = content.get("response_message", "")

        return ToolResult(tool_name="update_text", success=True, message=response_msg or "Updated post text", updates=updates)

    async def _execute_update_hashtags(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Update hashtags — either from explicit list or AI-generated."""
        explicit_hashtags = params.get("hashtags")

        if explicit_hashtags and isinstance(explicit_hashtags, list):
            # Direct update with explicit values
            await artifact_service.update_artifact_content(
                artifact_id=artifact.id,
                updates={"hashtags": explicit_hashtags},
                source="user_edit",
            )
            return ToolResult(
                tool_name="update_hashtags", success=True,
                message=f"Updated hashtags to {', '.join(explicit_hashtags)}",
                updates={"hashtags": explicit_hashtags},
            )

        # AI-generated hashtags via OptimizationAgent
        prompt = params.get("prompt", "generate relevant hashtags")
        current_content = _safe_content(artifact)
        post_text = current_content.get("text") or current_content.get("post_text") or ""

        try:
            from app.agents.elves.social_media_manager.mini_agents.optimization import OptimizationAgent

            optimization_agent = OptimizationAgent()
            agent_state = {
                "user_request": {
                    "platform": artifact.platform or "linkedin",
                    "topic": post_text[:100],
                },
                "content": {"post_text": post_text},
                "plan": {
                    "hashtag_strategy": f"User instruction: {prompt}",
                },
            }
            agent_context = {"user_id": user_id, "timezone": "UTC"}

            result = await optimization_agent.execute(agent_state, agent_context)
            hashtags = result.get("hashtags") or []

            # Guard against hashtags being returned as a string
            if isinstance(hashtags, str):
                hashtags = [h.strip() for h in hashtags.split(",") if h.strip()]

            # Normalize: extract tag string from dict or string
            normalized = []
            for h in hashtags:
                tag = h.get("tag", h) if isinstance(h, dict) else str(h)
                if not tag.startswith("#"):
                    tag = f"#{tag}"
                normalized.append(tag)

            if not normalized:
                return ToolResult(tool_name="update_hashtags", success=False, error="No hashtags generated")

            await artifact_service.update_artifact_content(
                artifact_id=artifact.id,
                updates={"hashtags": normalized},
                source="regeneration",
                prompt=prompt,
            )

            return ToolResult(
                tool_name="update_hashtags", success=True,
                message=f"Updated hashtags: {', '.join(normalized)}",
                updates={"hashtags": normalized},
            )
        except Exception as e:
            return ToolResult(tool_name="update_hashtags", success=False, error=str(e))

    async def _execute_update_schedule(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Parse natural language schedule and update."""
        prompt = params.get("prompt", "")

        # Use LLM to parse natural language time
        messages = [
            LLMMessage(
                role="system",
                content="You parse natural language scheduling instructions into structured data. Return valid JSON.",
            ),
            LLMMessage(
                role="user",
                content=f'Parse this scheduling instruction into a schedule object:\n"{prompt}"\n\nReturn JSON: {{"datetime": "ISO 8601 timestamp", "timezone": "timezone name", "reason": "brief explanation"}}',
            ),
        ]

        try:
            response = await llm_client.generate_fast(messages, json_mode=True)
            schedule = json.loads(response.content)

            # Guard against LLM returning a string instead of dict
            if isinstance(schedule, str):
                schedule = {"datetime": schedule, "reason": prompt}

            await artifact_service.update_artifact_content(
                artifact_id=artifact.id,
                updates={"schedule": schedule, "posting_schedule": schedule},
                source="user_edit",
            )

            return ToolResult(
                tool_name="update_schedule", success=True,
                message=f"Updated schedule: {(schedule.get('datetime') or prompt) if isinstance(schedule, dict) else prompt}",
                updates={"schedule": schedule},
            )
        except Exception as e:
            return ToolResult(tool_name="update_schedule", success=False, error=str(e))

    async def _execute_regenerate_image(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Regenerate image with improvement instructions."""
        prompt = params.get("prompt", "")
        current_content = _safe_content(artifact)
        original_prompt = current_content.get("image_prompt") or ""

        combined_prompt = f"{original_prompt}. Improvement: {prompt}" if original_prompt else prompt

        visual_agent = await self._get_visual_agent()
        image_result = await visual_agent._generate_image(
            description=combined_prompt,
            style="Professional, high-quality social media visual",
            dimensions="1200 x 630",
        )

        if not image_result or not image_result.get("url"):
            return ToolResult(tool_name="regenerate_image", success=False, error="Image generation failed")

        updates = {
            "image_url": image_result["url"],
            "image_prompt": combined_prompt,
        }

        await artifact_service.update_artifact_content(
            artifact_id=artifact.id,
            updates=updates,
            source="image_regeneration",
            prompt=prompt,
        )

        return ToolResult(
            tool_name="regenerate_image", success=True,
            message="Regenerated image",
            updates=updates,
        )

    async def _execute_generate_image(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Generate a fresh image from scratch."""
        prompt = params.get("prompt", "")

        visual_agent = await self._get_visual_agent()
        image_result = await visual_agent._generate_image(
            description=prompt,
            style="Professional, high-quality social media visual",
            dimensions="1200 x 630",
        )

        if not image_result or not image_result.get("url"):
            return ToolResult(tool_name="generate_image", success=False, error="Image generation failed")

        updates = {
            "image_url": image_result["url"],
            "image_prompt": prompt,
        }

        await artifact_service.update_artifact_content(
            artifact_id=artifact.id,
            updates=updates,
            source="image_regeneration",
            prompt=prompt,
        )

        return ToolResult(
            tool_name="generate_image", success=True,
            message="Generated new image",
            updates=updates,
        )

    async def _execute_direct_edit(self, params: dict, artifact, user_id: str) -> ToolResult:
        """Directly set field values without AI processing."""
        updates = {}
        for field in ["text", "hashtags", "cta", "hook"]:
            if params.get(field) is not None:
                updates[field] = params[field]

        if not updates:
            return ToolResult(tool_name="direct_edit", success=False, error="No fields to update")

        await artifact_service.update_artifact_content(
            artifact_id=artifact.id,
            updates=updates,
            source="user_edit",
        )

        fields = list(updates.keys())
        return ToolResult(
            tool_name="direct_edit", success=True,
            message=f"Updated {', '.join(fields)}",
            updates=updates,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _extract_elf_content(self, result: dict) -> dict:
        """Extract normalized content from elf response (same as artifacts.py _extract_content)."""
        variations = result.get("post_variations") or []
        if variations:
            variation = variations[0]
            raw = variation.get("content") or {}
            return {
                "text": raw.get("post_text") or raw.get("text") or "",
                "hook": raw.get("hook") or "",
                "cta": raw.get("cta") or "",
                "hashtags": [
                    h.get("tag", h) if isinstance(h, dict) else h
                    for h in variation.get("hashtags") or []
                ],
                "schedule": variation.get("posting_schedule") or {},
            }

        content = {}
        if "content" in result:
            raw_content = result["content"]
            if isinstance(raw_content, dict):
                content.update(raw_content)
            elif isinstance(raw_content, str):
                content["text"] = raw_content
        elif "final_output" in result:
            output = result["final_output"]
            if isinstance(output, dict):
                content.update(output)

        content.setdefault("text", result.get("post_text") or "")
        content.setdefault("hashtags", result.get("hashtags") or [])
        content.setdefault("schedule", result.get("timing") or {})
        return content


# ── Artifact Modifier Node ──────────────────────────────────────────────

class ArtifactModifierNode:
    """
    Tool-calling node for artifact modifications.

    Replaces full elf pipeline for modification requests. Uses LLM to
    select tools, executes them sequentially, and formats the response.
    """

    def __init__(self):
        self._executor = ToolExecutor()

    async def __call__(self, state: ConversationState) -> ConversationState:
        start_time = time.time()
        state["current_node"] = "artifact_modifier"

        add_stream_event(state, "node_started", node="artifact_modifier")

        try:
            # 1. Get target artifact (already resolved by router)
            target = state.get("target_artifact") or state.get("last_artifact")

            # ── DEBUG: Print artifact state at entry ──
            print("\n" + "=" * 80)
            print("🔧 ARTIFACT MODIFIER NODE — ENTRY DEBUG")
            print("=" * 80)
            print(f"  target_artifact from state: {json.dumps(state.get('target_artifact'), indent=2, default=str) if state.get('target_artifact') else 'None'}")
            print(f"  last_artifact from state:   {json.dumps(state.get('last_artifact'), indent=2, default=str) if state.get('last_artifact') else 'None'}")
            print(f"  artifact_history count:     {len(state.get('artifact_history') or [])}")
            print(f"  artifact_history IDs:       {[a.get('id') for a in (state.get('artifact_history') or [])]}")
            print(f"  artifacts in state:         {len(state.get('artifacts') or [])}")
            print(f"  modification_feedback:      {state.get('modification_feedback')}")
            print(f"  current_input:              {state.get('current_input')}")
            print(f"  current_intent:             {state.get('current_intent')}")
            print("=" * 80 + "\n")

            # Redis may return last_artifact as a JSON string
            if isinstance(target, str):
                try:
                    target = json.loads(target)
                except (json.JSONDecodeError, TypeError):
                    target = None
            if not target or not isinstance(target, dict) or not target.get("id"):
                state["final_response"] = "I couldn't find the post to modify. Could you try again?"
                self._trace(state, start_time, "no_target")
                return state

            # 2. Load fresh artifact from DB (latest version)
            artifact = await artifact_service.get_current_artifact(target["id"])
            if not artifact:
                state["final_response"] = "That post no longer exists. Would you like to create a new one?"
                self._trace(state, start_time, "artifact_not_found")
                return state

            # ── DEBUG: Print loaded artifact details ──
            loaded_content = _safe_content(artifact)
            print("\n" + "-" * 80)
            print("📄 LOADED ARTIFACT FROM DB")
            print("-" * 80)
            print(f"  artifact.id:              {artifact.id}")
            print(f"  artifact.parent_id:       {artifact.parent_artifact_id}")
            print(f"  artifact.platform:        {artifact.platform}")
            print(f"  artifact.version:         {getattr(artifact, 'version', 'N/A')}")
            print(f"  content.text:             {(loaded_content.get('text') or loaded_content.get('post_text') or '')[:200]}")
            print(f"  content.hashtags:         {loaded_content.get('hashtags', [])}")
            print(f"  content.image_url:        {(loaded_content.get('image_url') or 'None')[:100]}")
            print(f"  content.image_prompt:     {(loaded_content.get('image_prompt') or 'None')[:100]}")
            print(f"  content.schedule:         {loaded_content.get('schedule') or loaded_content.get('posting_schedule')}")
            print("-" * 80 + "\n")

            # 3. Ask LLM which tools to call
            user_message = state.get("modification_feedback") or state.get("current_input", "")
            plan_result = await self._plan_tools(artifact, user_message)

            # Unpack (tool_calls, response_message) tuple
            if isinstance(plan_result, tuple):
                tool_calls, planning_response = plan_result
            else:
                tool_calls, planning_response = plan_result, ""

            # 4. Handle "no tools" case (user asked a question, not a modification)
            if isinstance(tool_calls, dict) and tool_calls.get("no_tools"):
                state["final_response"] = tool_calls.get("response", "Could you clarify what you'd like to change?")
                self._trace(state, start_time, "no_tools")
                return state

            if not tool_calls:
                state["final_response"] = "I'm not sure what to change. Could you be more specific?"
                self._trace(state, start_time, "empty_tool_calls")
                return state

            # 5. Execute tools sequentially
            # ── DEBUG: Print selected tools before execution ──
            print("\n" + "-" * 80)
            print("🔨 TOOLS SELECTED FOR EXECUTION")
            print("-" * 80)
            for i, tc in enumerate(tool_calls):
                print(f"  [{i+1}] Tool: {tc.get('tool')}")
                print(f"      Params: {json.dumps(tc.get('parameters', {}), indent=8, default=str)}")
            print("-" * 80 + "\n")

            logger.info(
                "Executing modification tools",
                tool_count=len(tool_calls),
                tools=[c.get("tool") for c in tool_calls],
                artifact_id=artifact.id,
            )

            results = await self._executor.execute_tools(tool_calls, artifact, state["user_id"])

            # ── DEBUG: Print tool execution results ──
            print("\n" + "-" * 80)
            print("✅ TOOL EXECUTION RESULTS")
            print("-" * 80)
            for i, r in enumerate(results):
                status = "✅ SUCCESS" if r.success else "❌ FAILED"
                print(f"  [{i+1}] {r.tool_name}: {status}")
                print(f"      Message: {r.message}")
                if r.error:
                    print(f"      Error:   {r.error}")
                if r.updates:
                    print(f"      Updates: {json.dumps(r.updates, indent=8, default=str)[:500]}")
            print("-" * 80 + "\n")

            # 6. Build response
            state["final_response"] = self._build_response(results, planning_response)

            # 7. Refresh artifact (latest version) and put in state
            original_id = target["id"]  # always the original ID from state
            updated_artifact = await artifact_service.get_current_artifact(original_id)
            if updated_artifact:
                artifact_dict = updated_artifact.to_dict()
                state["artifacts"] = [artifact_dict]
                state["last_artifact"] = artifact_dict
                # Push artifact event for SSE streaming
                add_stream_event(state, "artifact_updated", content=artifact_dict, node="artifact_modifier")
                # Update artifact_history
                history = list(state.get("artifact_history") or [])
                for i, entry in enumerate(history):
                    if entry.get("id") == artifact_dict.get("id"):
                        history[i] = artifact_dict
                        break
                state["artifact_history"] = history

            self._trace(state, start_time, "completed", extra={
                "tools_executed": len(results),
                "tools_succeeded": sum(1 for r in results if r.success),
                "tools_failed": sum(1 for r in results if not r.success),
            })

            logger.info(
                "Artifact modification complete",
                tools_executed=len(results),
                all_success=all(r.success for r in results),
            )

        except Exception as e:
            logger.error("Artifact modification failed", error=str(e))
            # Fallback message — don't crash the graph
            state["final_response"] = (
                "I had trouble modifying the post. "
                "Could you try rephrasing your request?"
            )
            state["errors"].append(f"Artifact modifier error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "artifact_modifier", "failed", execution_time, str(e))

        return state

    async def _plan_tools(self, artifact, user_message: str) -> tuple[list[dict] | dict, str]:
        """Ask LLM which tools to call. Returns (tool_calls, response_message)."""
        content = _safe_content(artifact)

        prompt = TOOL_PLANNING_PROMPT.format(
            platform=artifact.platform or "unknown",
            text=(content.get("text") or content.get("post_text") or "")[:500],
            hashtags=json.dumps(content.get("hashtags", [])),
            schedule=json.dumps(content.get("schedule") or content.get("posting_schedule") or {}),
            has_image=bool(content.get("image_url")),
            image_prompt=(content.get("image_prompt") or "none")[:200],
            user_message=user_message,
            tools_json=json.dumps(MODIFICATION_TOOLS, indent=2),
        )

        # ── DEBUG: Print the full prompt sent to the LLM ──
        print("\n" + "=" * 80)
        print("📨 TOOL PLANNING PROMPT SENT TO LLM")
        print("=" * 80)
        print(f"  System: You are a tool planner for post modifications. Respond with valid JSON only.")
        print(f"  User prompt:\n{prompt}")
        print("=" * 80 + "\n")

        messages = [
            LLMMessage(
                role="system",
                content="You are a tool planner for post modifications. Respond with valid JSON only.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            from app.core.model_config import TaskType
            response = await llm_client.generate_for_task(TaskType.TOOL_PLANNING, messages, json_mode=True)
            result = json.loads(response.content)

            # ── DEBUG: Print the LLM response and selected tools ──
            print("\n" + "=" * 80)
            print("🤖 TOOL PLANNING LLM RESPONSE")
            print("=" * 80)
            print(f"  Raw LLM content: {response.content}")
            print(f"  Parsed result:   {json.dumps(result, indent=2, default=str)}")
            print("=" * 80 + "\n")

            logger.info("Tool planning LLM response", raw_result=result)

            # Extract response_message from dict results
            response_message = ""
            if isinstance(result, dict):
                response_message = result.get("response_message", "")

            # Handle various LLM response formats
            if isinstance(result, list):
                return self._normalize_tool_calls(result), ""
            if isinstance(result, dict):
                if result.get("no_tools"):
                    return result, ""
                # LLM wrapped in {"tool_calls": [...]}
                if "tool_calls" in result and isinstance(result["tool_calls"], list):
                    return self._normalize_tool_calls(result["tool_calls"]), response_message
                # Single tool call as object {"tool": "...", "parameters": {...}}
                if result.get("tool"):
                    return self._normalize_tool_calls([result]), response_message
                # LLM wrapped in {"tools": [...]}
                if "tools" in result and isinstance(result["tools"], list):
                    return self._normalize_tool_calls(result["tools"]), response_message

            logger.warning("Unexpected tool planning response format", result_type=type(result).__name__)
            return [], ""
        except Exception as e:
            logger.error("Tool planning failed", error=str(e))
            return [], ""

    def _normalize_tool_calls(self, tool_calls) -> list[dict]:
        """Ensure each tool call has a valid structure with dict parameters."""
        normalized = []
        for call in tool_calls:
            if not isinstance(call, dict):
                logger.warning("Skipping non-dict tool call", call=call)
                continue
            tool_name = call.get("tool")
            if not tool_name or not isinstance(tool_name, str):
                logger.warning("Skipping tool call without valid name", call=call)
                continue
            params = call.get("parameters") or {}
            # If parameters is a string, wrap it as a prompt
            if isinstance(params, str):
                params = {"prompt": params}
            if not isinstance(params, dict):
                params = {}
            normalized.append({"tool": tool_name, "parameters": params})
        return normalized

    def _build_response(self, results: list[ToolResult], planning_message: str = "") -> str:
        """Build user-facing response from tool results.

        Priority chain:
        1. ContentAgent's response_message (for update_text — knows actual result)
        2. Tool planning LLM's response_message (for other tools — knows what's planned)
        3. Hardcoded template (fallback if both missing or tools failed)
        """
        all_success = all(r.success for r in results)

        if all_success:
            # Prefer ContentAgent's natural response for update_text
            for r in results:
                if r.tool_name == "update_text" and r.message and r.message != "Updated post text":
                    return r.message

            # Fall back to tool planning LLM's response_message
            if planning_message:
                return planning_message

        # Final fallback: template
        parts = []
        for result in results:
            if result.success:
                parts.append(f"- {result.message}")
            else:
                parts.append(f"- Failed: {result.error}")

        if not parts:
            return "No changes were made."

        header = "Done! Here's what I changed:\n" if all_success else "Completed with some issues:\n"
        return header + "\n".join(parts)

    def _trace(self, state: ConversationState, start_time: float, reason: str, extra: dict = None):
        """Log execution trace."""
        execution_time = int((time.time() - start_time) * 1000)
        metadata = {"reason": reason}
        if extra:
            metadata.update(extra)
        add_execution_trace(state, "artifact_modifier", "completed", execution_time, metadata=metadata)
        add_stream_event(state, "node_completed", node="artifact_modifier")


# Create node instance
artifact_modifier_node = ArtifactModifierNode()
