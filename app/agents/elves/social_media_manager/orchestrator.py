"""
Social Media Manager Elf Orchestrator.
LangGraph-based workflow coordinating all mini-agents.
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
    ContentGeneratorAgent,
    HashtagResearchAgent,
    StrategyAgent,
    TimingOptimizerAgent,
    VisualAdvisorAgent,
)

logger = structlog.get_logger(__name__)


class SocialMediaManagerElf(BaseElf):
    """
    Social Media Manager Elf - Creates, optimizes, and schedules social media content.
    
    Mini-Agents:
    1. Strategy Agent - Creates content strategy brief
    2. Content Generator Agent - Generates content variations
    3. Hashtag Research Agent - Researches optimal hashtags
    4. Timing Optimizer Agent - Calculates best posting times
    5. Visual Advisor Agent - Recommends visual content
    
    Workflow Pattern:
    - Sequential: Strategy runs first
    - Parallel: Content, Hashtags, Timing, Visual run simultaneously
    - Synthesis: Combines all outputs
    """
    
    name = "social_media_manager"
    description = "Creates and optimizes social media content"
    version = "1.0"
    
    def __init__(self):
        # Initialize mini-agents
        self.strategy_agent = StrategyAgent()
        self.content_agent = ContentGeneratorAgent()
        self.hashtag_agent = HashtagResearchAgent()
        self.timing_agent = TimingOptimizerAgent()
        self.visual_agent = VisualAdvisorAgent()
        
        super().__init__()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow."""
        
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("strategy", self._run_strategy)
        workflow.add_node("parallel_agents", self._run_parallel_agents)
        workflow.add_node("synthesize", self._synthesize_results)
        workflow.add_node("validate", self._validate_output)
        
        # Define edges
        workflow.set_entry_point("strategy")
        workflow.add_edge("strategy", "parallel_agents")
        workflow.add_edge("parallel_agents", "synthesize")
        workflow.add_edge("synthesize", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "retry": "parallel_agents",
                "complete": END,
            }
        )
        
        self._workflow = workflow.compile()
    
    async def execute(self, request: dict, context: dict) -> dict:
        """
        Execute the Social Media Manager workflow.
        
        Args:
            request: Task request with topic, platform, etc.
            context: Execution context with user info
            
        Returns:
            Generated content with all optimizations
        """
        start_time = time.time()
        
        # Extract entities from context and merge into request
        entities = context.get("entities", {})
        
        # Build enriched request with extracted entities
        enriched_request = {
            **request,
            # Use entities if not already in request
            "topic": request.get("topic") or entities.get("topic") or self._extract_topic_from_message(request.get("message", "")),
            "platform": request.get("platform") or entities.get("platform") or "linkedin",
            "content_type": request.get("content_type", "thought_leadership"),
            "goals": request.get("goals", ["engagement"]),
        }
        
        # Initialize state
        initial_state = {
            "user_request": enriched_request,
            "context": context,
            "strategy": None,
            "content_variations": [],
            "hashtags": [],
            "timing": None,
            "visual_advice": [],
            "final_output": None,
            "retry_count": 0,
            "errors": [],
            "execution_trace": [],
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
        
        # Common patterns for topic extraction
        patterns = [
            r'about\s+(.+?)(?:\s+for|\s+on\s+(?:linkedin|twitter|instagram|facebook)|\.|$)',
            r'on\s+(?!linkedin|twitter|instagram|facebook)(.+?)(?:\s+for|\.|$)',
            r'post\s+about\s+(.+?)(?:\s+for|\s+on|\.|$)',
            r'content\s+about\s+(.+?)(?:\s+for|\s+on|\.|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3:
                    return topic
        
        # Last resort: Take the message after removing action words
        cleaned = re.sub(
            r'^(?:create|write|generate|make)\s+(?:a\s+)?(?:linkedin|twitter|instagram|facebook)?\s*(?:post|content|article)?\s*',
            '',
            message,
            flags=re.IGNORECASE
        ).strip()
        
        return cleaned if len(cleaned) > 3 else message
    
    async def _run_strategy(self, state: dict) -> dict:
        """Run strategy agent."""
        logger.debug("Running strategy agent")
        
        result = await self.strategy_agent.execute(
            state, state.get("context", {})
        )
        
        state["strategy"] = result.get("strategy")
        state["execution_trace"].append({
            "agent": "strategy",
            "status": "completed",
        })
        
        return state
    
    async def _run_parallel_agents(self, state: dict) -> dict:
        """Run content, hashtag, timing, and visual agents in parallel."""
        logger.debug("Running parallel agents")
        
        context = state.get("context", {})
        
        # Run all agents in parallel
        results = await asyncio.gather(
            self.content_agent.execute(state, context),
            self.hashtag_agent.execute(state, context),
            self.timing_agent.execute(state, context),
            self.visual_agent.execute(state, context),
            return_exceptions=True,
        )
        
        # Process results
        agent_names = ["content", "hashtags", "timing", "visual"]
        
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
                # Merge result into state
                if name == "content":
                    state["content_variations"] = result.get("content_variations", [])
                elif name == "hashtags":
                    state["hashtags"] = result.get("hashtags", [])
                elif name == "timing":
                    state["timing"] = result.get("timing")
                elif name == "visual":
                    state["visual_advice"] = result.get("visual_advice", [])
                
                state["execution_trace"].append({
                    "agent": name,
                    "status": "completed",
                })
        
        return state
    
    async def _synthesize_results(self, state: dict) -> dict:
        """Synthesize all agent outputs into final result."""
        logger.debug("Synthesizing results")
        
        content_variations = state.get("content_variations", [])
        hashtags = state.get("hashtags", [])
        timing = state.get("timing", {})
        visual_advice = state.get("visual_advice", [])
        
        # Build complete variations
        complete_variations = []
        
        for var in content_variations:
            complete_var = ContentVariation(
                version=var.get("version", "variation"),
                content=var.get("content", {}),
                hashtags=hashtags,
                posting_schedule=timing or {},
                visual_recommendations=visual_advice,
                complete_preview=self._build_preview(var, hashtags),
                estimated_engagement=self._estimate_engagement(
                    var, hashtags, timing, state.get("user_request", {}).get("platform", "linkedin")
                ),
            )
            complete_variations.append(complete_var)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(complete_variations)
        
        state["final_output"] = {
            "post_variations": [v.model_dump() for v in complete_variations],
            "recommendations": recommendations,
        }
        
        state["execution_trace"].append({
            "agent": "synthesizer",
            "status": "completed",
        })
        
        return state
    
    async def _validate_output(self, state: dict) -> dict:
        """Validate the final output meets requirements."""
        logger.debug("Validating output")
        
        final_output = state.get("final_output", {})
        variations = final_output.get("post_variations", [])
        
        # Check we have content
        if not variations:
            state["errors"].append("No content variations generated")
            state["execution_trace"].append({
                "agent": "validator",
                "status": "failed",
                "error": "No variations",
            })
        else:
            # Validate each variation has content
            for i, var in enumerate(variations):
                content = var.get("content", {})
                if not content.get("post_text"):
                    state["errors"].append(f"Variation {i} has no content")
            
            state["execution_trace"].append({
                "agent": "validator",
                "status": "completed",
            })
        
        return state
    
    def _should_retry(self, state: dict) -> str:
        """Determine if workflow should retry."""
        retry_count = state.get("retry_count", 0)
        errors = state.get("errors", [])
        final_output = state.get("final_output", {})
        
        # Check if we have valid output
        has_content = bool(final_output.get("post_variations"))
        
        # Retry if no content and under retry limit
        if not has_content and retry_count < 2 and errors:
            state["retry_count"] = retry_count + 1
            logger.warning(
                "Retrying workflow",
                retry_count=state["retry_count"],
                errors=errors,
            )
            return "retry"
        
        return "complete"
    
    def _build_preview(self, variation: dict, hashtags: list[dict]) -> str:
        """Build complete preview of post with hashtags."""
        content = variation.get("content", {})
        post_text = content.get("post_text", "")
        
        if hashtags:
            tags = " ".join(h.get("tag", "") for h in hashtags[:5])
            return f"{post_text}\n\n{tags}"
        
        return post_text
    
    def _estimate_engagement(
        self,
        variation: dict,
        hashtags: list[dict],
        timing: dict,
        platform: str,
    ) -> dict:
        """Estimate engagement metrics for variation."""
        # Base estimates by platform
        base_reach = {
            "linkedin": 500,
            "twitter": 1000,
            "instagram": 800,
            "facebook": 600,
        }
        
        base_engagement = {
            "linkedin": 0.03,  # 3%
            "twitter": 0.015,  # 1.5%
            "instagram": 0.04,  # 4%
            "facebook": 0.02,  # 2%
        }
        
        reach = base_reach.get(platform, 500)
        engagement_rate = base_engagement.get(platform, 0.02)
        
        # Adjust for hashtags
        if hashtags:
            hashtag_boost = min(len(hashtags) * 0.05, 0.3)  # Up to 30% boost
            reach = int(reach * (1 + hashtag_boost))
        
        # Adjust for optimal timing
        if timing and timing.get("confidence", 0) > 0.8:
            reach = int(reach * 1.2)
        
        return {
            "reach": reach,
            "engagement_rate": round(engagement_rate, 4),
            "confidence": 0.6,  # These are estimates
        }
    
    def _generate_recommendations(self, variations: list[ContentVariation]) -> dict:
        """Generate recommendations for best variation."""
        if not variations:
            return {
                "best_variation": None,
                "reason": "No variations available",
                "suggested_action": "Try generating content again",
            }
        
        # Score each variation
        scores = []
        for i, var in enumerate(variations):
            score = 0
            
            # Has content
            if var.content.get("post_text"):
                score += 5
            
            # Has hashtags
            if var.hashtags:
                score += 2
            
            # Has timing
            if var.posting_schedule:
                score += 2
            
            # Has visual recommendations
            if var.visual_recommendations:
                score += 1
            
            # Engagement estimate
            engagement = var.estimated_engagement.get("engagement_rate", 0)
            score += engagement * 10
            
            scores.append((i, score, var.version))
        
        # Get best
        best_idx, best_score, best_version = max(scores, key=lambda x: x[1])
        
        # Version descriptions
        version_descriptions = {
            "hook_focused": "attention-grabbing approach",
            "story_focused": "narrative engagement style",
            "value_focused": "educational content approach",
        }
        
        return {
            "best_variation": best_version,
            "reason": f"The {version_descriptions.get(best_version, best_version)} "
                     f"is recommended based on the content type and platform patterns.",
            "suggested_action": "Review and customize before scheduling",
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

