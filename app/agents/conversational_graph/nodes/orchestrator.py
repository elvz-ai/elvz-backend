"""
Multi-Platform Orchestrator Node.

Coordinates artifact generation across multiple platforms.
"""

import asyncio
import time
import uuid
from typing import Any

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    GeneratedArtifact,
    add_execution_trace,
    add_stream_event,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)


class MultiPlatformOrchestratorNode:
    """
    Orchestrates content generation across platforms.

    Calls the existing SocialMediaManagerElf for each platform
    and collects results.
    """

    def __init__(self):
        self._elf = None

    async def _get_elf(self):
        """Lazy load the SocialMediaManagerElf."""
        if self._elf is None:
            try:
                from app.agents.elves.social_media_manager.orchestrator import SocialMediaManagerElf
                self._elf = SocialMediaManagerElf()
            except ImportError as e:
                logger.error(f"Failed to import SocialMediaManagerElf: {e}")
                raise
        return self._elf

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Generate content for all platforms.

        Args:
            state: Current conversation state

        Returns:
            Updated state with generated artifacts
        """
        start_time = time.time()
        state["current_node"] = "multi_platform_orchestrator"

        add_stream_event(state, "node_started", node="multi_platform_orchestrator")

        try:
            queries = state.get("decomposed_queries", [])

            if not queries:
                logger.warning("No queries to process")
                state["final_response"] = "I couldn't understand what content to generate. Could you please be more specific?"
                return state

            # Create batch ID for this generation
            batch_id = str(uuid.uuid4())
            state["artifact_batch_id"] = batch_id

            # Initialize progress tracking
            state["generation_progress"] = {q["platform"]: 0.0 for q in queries}

            # Generate for each platform
            if settings.enable_parallel_generation and len(queries) > 1:
                # Parallel generation
                artifacts = await self._generate_parallel(state, queries, batch_id)
            else:
                # Sequential generation
                artifacts = await self._generate_sequential(state, queries, batch_id)

            state["artifacts"] = artifacts

            # Organize by platform
            state["artifacts_by_platform"] = {
                a["platform"]: a for a in artifacts if a.get("platform")
            }

            # If no artifacts generated, ask the user for more details via HITL
            if not artifacts:
                platforms = [q.get("platform", "unknown") for q in queries]
                state["needs_follow_up"] = True
                state["follow_up_questions"] = [
                    f"I wasn't able to generate content for {', '.join(platforms)}. "
                    "Could you give me more details? For example:\n"
                    "- What topic should the post be about?\n"
                    "- What tone do you want (professional, casual, inspirational)?\n"
                    "- Any specific message or call-to-action to include?"
                ]
                state["final_response"] = state["follow_up_questions"][0]
                return state

            # Build final response
            state["final_response"] = self._build_response(artifacts, queries)
            state["suggestions"] = self._get_suggestions(artifacts)

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "multi_platform_orchestrator",
                "completed",
                execution_time,
                metadata={
                    "artifact_count": len(artifacts),
                    "batch_id": batch_id,
                },
            )
            add_stream_event(
                state,
                "node_completed",
                content={"artifact_count": len(artifacts)},
                node="multi_platform_orchestrator",
            )

            logger.info(
                "Content generation complete",
                artifact_count=len(artifacts),
                batch_id=batch_id,
            )

        except Exception as e:
            logger.error("Content generation failed", error=str(e))
            state["errors"].append(f"Generation error: {str(e)}")
            state["final_response"] = (
                "I encountered an error while generating content. "
                "Please try again or rephrase your request."
            )

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "multi_platform_orchestrator",
                "failed",
                execution_time,
                str(e),
            )

        return state

    async def _generate_parallel(
        self,
        state: ConversationState,
        queries: list,
        batch_id: str,
    ) -> list[GeneratedArtifact]:
        """Generate content for all platforms in parallel."""
        tasks = []

        for query in queries:
            task = self._generate_for_platform(state, query, batch_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        artifacts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Platform generation failed: {result}")
                state["errors"].append(f"Failed for {queries[i]['platform']}: {str(result)}")
            elif result:
                artifacts.append(result)

        return artifacts

    async def _generate_sequential(
        self,
        state: ConversationState,
        queries: list,
        batch_id: str,
    ) -> list[GeneratedArtifact]:
        """Generate content for platforms sequentially."""
        artifacts = []

        for query in queries:
            try:
                artifact = await self._generate_for_platform(state, query, batch_id)
                if artifact:
                    artifacts.append(artifact)
            except Exception as e:
                logger.error(f"Failed for {query['platform']}: {e}")
                state["errors"].append(f"Failed for {query['platform']}: {str(e)}")

        return artifacts

    async def _generate_for_platform(
        self,
        state: ConversationState,
        query: dict,
        batch_id: str,
    ) -> GeneratedArtifact:
        """
        Generate content for a single platform.

        Args:
            state: Conversation state
            query: Decomposed query for this platform
            batch_id: Batch identifier

        Returns:
            Generated artifact
        """
        platform = query.get("platform", "linkedin")
        topic = query.get("topic") or self._extract_topic(query.get("query", ""))

        # Update progress
        add_stream_event(
            state,
            "platform_started",
            content={"platform": platform},
            node="multi_platform_orchestrator",
        )

        start_time = time.time()

        try:
            # Get the Elf
            elf = await self._get_elf()

            # Build request for Elf
            elf_request = {
                "topic": topic,
                "platform": platform,
                "content_type": "thought_leadership",
                "message": query.get("query", state["current_input"]),
            }

            # Build context for Elf
            context = {
                "user_id": state["user_id"],
                "image": False,  # Can be enabled based on user preferences
                "video": False,
                "brand_info": self._get_brand_info(state),
            }

            # Add RAG context if available
            if state.get("rag_context"):
                context["rag_context"] = state["rag_context"]

            # Execute Elf
            result = await elf.execute(elf_request, context)

            generation_time = int((time.time() - start_time) * 1000)

            # Build artifact from result
            artifact = GeneratedArtifact(
                id=str(uuid.uuid4()),
                platform=platform,
                artifact_type="social_post",
                content=self._extract_content(result),
                status="completed",
                generation_time_ms=generation_time,
                tokens_used=result.get("tokens_used", 0),
            )

            # Update progress
            state["generation_progress"][platform] = 1.0
            add_stream_event(
                state,
                "platform_completed",
                content={"platform": platform, "artifact": artifact},
                node="multi_platform_orchestrator",
            )

            return artifact

        except Exception as e:
            logger.error(f"Generation failed for {platform}", error=str(e))
            state["generation_progress"][platform] = -1  # Mark as failed
            raise

    def _extract_content(self, result: dict) -> dict:
        """Extract content from Elf result."""
        # Handle different result structures
        if not result:
            return {}

        content = {}

        # Direct content
        if "content" in result:
            content.update(result["content"])
        elif "final_output" in result:
            output = result["final_output"]
            if isinstance(output, dict):
                content.update(output)

        # Extract specific fields
        content.setdefault("text", result.get("post_text", ""))
        content.setdefault("hashtags", result.get("hashtags", []))
        content.setdefault("schedule", result.get("timing", {}))

        # Visual content
        if "visual_advice" in result:
            content["visual"] = result["visual_advice"]

        return content

    def _extract_topic(self, query: str) -> str:
        """Extract topic from query text."""
        # Simple extraction - take key words
        stopwords = {"generate", "create", "write", "post", "content", "for", "about", "a", "an", "the"}
        words = query.lower().split()
        topic_words = [w for w in words if w not in stopwords]
        return " ".join(topic_words[:5]) if topic_words else "general content"

    def _get_brand_info(self, state: ConversationState) -> dict:
        """Extract brand info from user profile."""
        profile = state.get("user_profile", {})

        if not profile:
            return {}

        user_profile = profile.get("profile", {})

        return {
            "brand_name": user_profile.get("brand_name", ""),
            "industry": user_profile.get("industry", ""),
            "brand_voice": user_profile.get("brand_voice_description", ""),
        }

    def _build_response(self, artifacts: list, queries: list) -> str:
        """Build response message from generated artifacts."""
        if not artifacts:
            return "I couldn't generate any content. Please try again."

        if len(artifacts) == 1:
            artifact = artifacts[0]
            content = artifact.get("content", {})
            text = content.get("text", "")

            response = f"Here's your {artifact['platform'].title()} post:\n\n"
            response += f"**Post:**\n{text}\n\n"

            if content.get("hashtags"):
                hashtags = " ".join(content["hashtags"][:10])
                response += f"**Hashtags:** {hashtags}\n"

            return response

        else:
            # Multiple platforms
            response = f"Here's your content for {len(artifacts)} platforms:\n\n"

            for artifact in artifacts:
                platform = artifact.get("platform", "unknown").title()
                content = artifact.get("content", {})
                text = content.get("text", "")[:200]

                response += f"**{platform}:**\n{text}...\n\n"

            response += "Would you like me to show the full content for any platform?"
            return response

    def _get_suggestions(self, artifacts: list) -> list[str]:
        """Generate follow-up suggestions."""
        suggestions = []

        if artifacts:
            suggestions.append("Generate an image for this post")
            suggestions.append("Adjust the tone")
            suggestions.append("Create a variation")

        if len(artifacts) == 1:
            suggestions.append("Create for another platform")

        return suggestions[:4]


# Create node instance
multi_platform_orchestrator_node = MultiPlatformOrchestratorNode()
