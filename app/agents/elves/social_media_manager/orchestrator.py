"""
Social Media Manager Elf Orchestrator.
LangGraph workflow with intelligent planning and conditional agent execution.
"""

import asyncio
import time
from typing import Any, Optional

import structlog
from langgraph.graph import END, StateGraph

from app.agents.elves.base import BaseElf
from app.agents.elves.social_media_manager.state import (
    ContentVariation,
    CreatePostRequest,
    SocialMediaState,
)
from app.agents.elves.social_media_manager.mini_agents import (
    ContentAgent,
    ExpertPersonaAgent,
    OptimizationAgent,
    PlannerAgent,
    VideoAgent,
    VisualAgent,
)

logger = structlog.get_logger(__name__)


class SocialMediaManagerElf(BaseElf):
    """
    Social Media Manager Elf - Creates and optimizes social media content.

    Mini-Agents (5 total):
    1. Planner Agent - Decides which agents to run (runs FIRST)
    2. Content Agent - Generates post content (runs SECOND)
    3. Optimization Agent - Generates hashtags and timing (conditional)
    4. Visual Agent - Generates image descriptions (conditional)
    5. Video Agent - Generates video scripts and recommendations (conditional)

    Workflow Pattern:
    - Planner runs first to decide what's needed
    - Content runs second to generate the post
    - Downstream agents (Optimization, Visual, Video) run in parallel based on planner decision
    - Synthesis combines all outputs

    Target: <10 second response time
    """

    name = "social_media_manager"
    description = "Creates and optimizes social media content"
    version = "4.0"

    def __init__(self):
        # Initialize mini-agents
        self.planner_agent = PlannerAgent()
        self.expert_persona_agent = ExpertPersonaAgent()
        self.content_agent = ContentAgent()
        self.optimization_agent = OptimizationAgent()
        self.visual_agent = VisualAgent()
        self.video_agent = VideoAgent()

        super().__init__()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow with planner-first, then parallel execution."""
        
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("parallel_agents", self._run_parallel_agents)
        workflow.add_node("synthesize", self._synthesize_results)
        
        # Define edges: planner → parallel_agents → synthesize → END
        # After planner decides, Content + Optimization + Visual run in PARALLEL
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "parallel_agents")
        workflow.add_edge("parallel_agents", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self._workflow = workflow.compile()
    
    async def execute(self, request: dict, context: dict) -> dict:
        """
        Execute the Social Media Manager workflow.
        
        Args:
            request: Task request with topic, platform, etc.
            context: Execution context with user info
            
        Returns:
            Generated content with optimizations
        """
        start_time = time.time()
        
        # Extract entities from context
        entities = context.get("entities", {})
        
        # Build enriched request
        enriched_request = {
            **request,
            "topic": request.get("topic") or entities.get("topic") or self._extract_topic_from_message(request.get("message", "")),
            "platform": request.get("platform") or entities.get("platform") or "linkedin",
            "content_type": request.get("content_type", "thought_leadership"),
            "goals": request.get("goals", ["engagement"]),
            "message": request.get("message", ""),
        }
        
        # Initialize state
        initial_state = {
            "user_request": enriched_request,
            "context": context,
            "plan": None,  # Planner output
            "content": None,
            "hashtags": [],
            "timing": None,
            "visual_advice": None,
            "video_advice": None,
            "final_output": None,
            "errors": [],
            "execution_trace": [],
            # Modification fields — populated when route == "modification"
            "previous_content": enriched_request.get("previous_content", ""),
            "modification_feedback": enriched_request.get("modification_feedback", ""),
        }
        
        logger.info(
            "Social Media Manager executing",
            platform=enriched_request.get("platform"),
            topic=enriched_request.get("topic", "")[:50],
        )
        
        try:
            # Run workflow
            final_state = await self._workflow.ainvoke(initial_state)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            return self._build_response(final_state, execution_time_ms)
            
        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "error": str(e),
                "execution_time_ms": execution_time_ms,
                "post_variations": [],
                "recommendations": {
                    "best_variation": None,
                    "reason": "Execution failed",
                    "suggested_action": "Please try again",
                },
            }
    
    def _extract_topic_from_message(self, message: str) -> str:
        """Extract topic from user's message as fallback."""
        import re
        
        if not message:
            return ""
        
        patterns = [
            r'about\s+(.+?)(?:\s+for|\s+on\s+(?:linkedin|twitter|instagram|facebook)|\.|\?|$)',
            r'on\s+(?!linkedin|twitter|instagram|facebook)(.+?)(?:\s+for|\.|\?|$)',
            r'post\s+about\s+(.+?)(?:\s+for|\s+on|\.|\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3:
                    return topic
        
        # Fallback: clean action words
        cleaned = re.sub(
            r'^(?:create|write|generate|make)\s+(?:a\s+)?(?:linkedin|twitter|instagram|facebook)?\s*(?:post|content|article)?\s*',
            '',
            message,
            flags=re.IGNORECASE
        ).strip()
        
        return cleaned if len(cleaned) > 3 else message
    
    async def _run_planner(self, state: dict) -> dict:
        """Run Planner + ExpertPersona agents in parallel to decide strategy and build expert identity."""
        logger.debug("Running planner agent")

        context = state.get("context", {})
        request = state.get("user_request", {})

        # Run planner and expert persona generation concurrently (zero latency overhead)
        planner_result, expert_persona = await asyncio.gather(
            self.planner_agent.execute(state, context),
            self.expert_persona_agent.generate(
                topic=request.get("topic", ""),
                platform=request.get("platform", "linkedin"),
                content_type=request.get("content_type", "thought_leadership"),
            ),
            return_exceptions=True,
        )

        # Process planner result
        if isinstance(planner_result, Exception):
            logger.error("Planner agent failed", error=str(planner_result))
            image_requested = context.get("image", False)
            video_requested = context.get("video", False)
            state["plan"] = {
                "include_hashtags": True,
                "include_visual": image_requested or video_requested,
                "reasoning": "Fallback due to planner error",
            }
            state["errors"].append(f"planner: {str(planner_result)}")
            state["execution_trace"].append({
                "agent": "planner",
                "status": "failed",
                "error": str(planner_result),
            })
        else:
            state["plan"] = planner_result.get("plan", {})
            state["execution_trace"].append({
                "agent": "planner",
                "status": "completed",
                "decision": {
                    "include_hashtags": state["plan"].get("include_hashtags"),
                    "include_visual": state["plan"].get("include_visual"),
                    "include_video": state["plan"].get("include_video"),
                },
            })
            logger.info(
                "Planner completed",
                include_hashtags=state["plan"].get("include_hashtags"),
                include_visual=state["plan"].get("include_visual"),
                include_video=state["plan"].get("include_video"),
            )

        # Inject expert persona into plan (empty string → content.py falls back to static prompt)
        if isinstance(expert_persona, Exception):
            logger.warning("Expert persona agent failed", error=str(expert_persona))
            state["plan"]["expert_persona"] = ""
        else:
            state["plan"]["expert_persona"] = expert_persona or ""

        return state
    
    async def _run_parallel_agents(self, state: dict) -> dict:
        """Run Content, Optimization, Visual, and Video agents in parallel based on planner decision."""
        plan = state.get("plan") or {}
        context = state.get("context") or {}

        include_hashtags = plan.get("include_hashtags", True)

        # Check if image/video content is requested and allowed by user flags
        image_requested = context.get("image", False)
        video_requested = context.get("video", False)

        # Include visual only if planner suggests it AND user allows it (image=true)
        include_visual = plan.get("include_visual", False) and image_requested

        # Include video only if planner suggests it AND user allows it (video=true)
        include_video = plan.get("include_video", False) and video_requested

        logger.debug(
            "Running parallel agents",
            include_hashtags=include_hashtags,
            include_visual=include_visual,
            include_video=include_video,
            image_requested=image_requested,
            video_requested=video_requested,
        )

        # Content agent ALWAYS runs
        agents_to_run = [
            ("content", self.content_agent.execute(state, context)),
        ]

        # Optimization runs based on planner decision
        if include_hashtags:
            agents_to_run.append((
                "optimization",
                self.optimization_agent.execute(state, context, target_audience=None)
            ))

        # Visual runs based on planner decision AND image flag
        if include_visual:
            agents_to_run.append((
                "visual",
                self.visual_agent.execute(state, context)
            ))

        # Video runs based on planner decision AND video flag
        if include_video:
            agents_to_run.append((
                "video",
                self.video_agent.execute(state, context)
            ))
        
        # Run ALL in parallel
        agent_names = [a[0] for a in agents_to_run]
        agent_tasks = [a[1] for a in agents_to_run]
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                logger.error(f"{name} agent failed", error=str(result))
                state["errors"].append(f"{name}: {str(result)}")
                state["execution_trace"].append({
                    "agent": name,
                    "status": "failed",
                    "error": str(result),
                })
            else:
                if name == "content":
                    state["content"] = result.get("content")
                    state["content_output"] = result
                elif name == "optimization":
                    state["hashtags"] = result.get("hashtags", [])
                    state["timing"] = result.get("timing")
                elif name == "visual":
                    state["visual_advice"] = result.get("visual_advice")
                elif name == "video":
                    state["video_advice"] = result.get("video_advice")

                state["execution_trace"].append({
                    "agent": name,
                    "status": "completed",
                })
        
        return state
    
    async def _synthesize_results(self, state: dict) -> dict:
        """Synthesize all agent outputs into final result."""
        logger.debug("Synthesizing results")

        content = state.get("content") or {}
        hashtags = state.get("hashtags") or []
        timing = state.get("timing") or {}
        visual_advice = state.get("visual_advice")
        video_advice = state.get("video_advice")

        # Build visual recommendations list (includes both images and videos)
        visual_recommendations = []
        if visual_advice:
            visual_recommendations.append(visual_advice)
        if video_advice:
            visual_recommendations.append(video_advice)

        # Build complete post
        complete_variation = ContentVariation(
            version="optimized",
            content=content,
            hashtags=hashtags,
            posting_schedule=timing or {},
            visual_recommendations=visual_recommendations,
            complete_preview=self._build_preview(content, hashtags),
            estimated_engagement=self._estimate_engagement(
                content,
                hashtags,
                timing,
                state.get("user_request", {}).get("platform", "linkedin")
            ),
        )
        
        state["final_output"] = {
            "post_variations": [complete_variation.model_dump()],
            "recommendations": {
                "best_variation": "optimized",
                "reason": "AI-optimized content with strategic hashtags and timing",
                "suggested_action": "Review and customize before scheduling",
            },
        }
        
        state["execution_trace"].append({
            "agent": "synthesizer",
            "status": "completed",
        })
        
        return state
    
    def _build_preview(self, content: dict, hashtags: list[dict]) -> str:
        """Build complete preview of post with hashtags."""
        post_text = content.get("post_text", "") if content else ""
        
        if hashtags:
            tags = " ".join(h.get("tag", "") for h in hashtags[:5])
            return f"{post_text}\n\n{tags}"
        
        return post_text
    
    def _estimate_engagement(
        self,
        content: dict,
        hashtags: list[dict],
        timing: dict,
        platform: str,
    ) -> dict:
        """Estimate engagement metrics."""
        base_reach = {
            "linkedin": 500,
            "twitter": 1000,
            "instagram": 800,
            "facebook": 600,
        }
        
        base_engagement = {
            "linkedin": 0.03,
            "twitter": 0.015,
            "instagram": 0.04,
            "facebook": 0.02,
        }
        
        reach = base_reach.get(platform, 500)
        engagement_rate = base_engagement.get(platform, 0.02)
        
        # Boost for hashtags
        if hashtags:
            hashtag_boost = min(len(hashtags) * 0.05, 0.3)
            reach = int(reach * (1 + hashtag_boost))
        
        # Boost for optimal timing
        if timing and timing.get("confidence", 0) > 0.8:
            reach = int(reach * 1.2)
        
        return {
            "reach": reach,
            "engagement_rate": round(engagement_rate, 4),
            "confidence": 0.7,
        }
    
    def _build_response(self, state: dict, execution_time_ms: int) -> dict:
        """Build final response from state."""
        final_output = state.get("final_output", {})
        
        return {
            "post_variations": final_output.get("post_variations", []),
            "recommendations": final_output.get("recommendations", {
                "best_variation": None,
                "reason": "Unable to generate recommendations",
                "suggested_action": "Please try again",
            }),
            "execution_time_ms": execution_time_ms,
            "execution_trace": state.get("execution_trace", []),
            "errors": state.get("errors", []),
        }
