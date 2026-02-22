"""
Data Checker Node.

Verifies that required user data is available in the social_memory_test
Qdrant collection for content generation. Blocks artifact generation
when the user has not connected their social media.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)

logger = structlog.get_logger(__name__)


class DataCheckerNode:
    """
    Checks availability of user data in the social_memory_test collection.

    When ALL requested platforms have no data, sets social_not_connected=True
    and writes a final_response telling the user to connect their social media.
    The graph's conditional edge then skips artifact generation.
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        start_time = time.time()
        state["current_node"] = "data_checker"

        add_stream_event(state, "node_started", node="data_checker")

        try:
            user_id = state["user_id"]
            platforms = self._get_platforms(state)

            data_available = {}
            missing_platforms = []

            for platform in platforms:
                has_data = await self._check_platform_data(user_id, platform)
                data_available[platform] = has_data
                if not has_data:
                    missing_platforms.append(platform)

            state["data_available"] = data_available
            state["missing_data_platforms"] = missing_platforms

            # Check user profile
            has_profile = state.get("user_profile") is not None
            has_brand_voice = (
                (state.get("user_profile") or {}).get("brand_voice") is not None
                if has_profile else False
            )
            state["working_memory"]["has_user_profile"] = has_profile
            state["working_memory"]["has_brand_voice"] = has_brand_voice

            if missing_platforms and len(missing_platforms) == len(platforms):
                # No social media data at all — user hasn't connected
                state["social_not_connected"] = True
                platform_names = ", ".join(p.title() for p in missing_platforms)
                state["final_response"] = (
                    f"You don't have {platform_names} connected yet. "
                    "Please connect your social media account at "
                    "https://www.elvz.ai/elves/social-media-manager to get started."
                )
                logger.info(
                    "Social media not connected — blocking artifact generation",
                    user_id=user_id,
                    missing=missing_platforms,
                )
            elif missing_platforms:
                # Some platforms connected, some not — proceed with note
                state["working_memory"]["reduced_personalization"] = True
                connected = [p for p in platforms if p not in missing_platforms]
                state["working_memory"]["personalization_note"] = (
                    f"Note: Limited personalization for {', '.join(missing_platforms)}. "
                    f"Data available for: {', '.join(connected)}."
                )

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(
                state,
                "data_checker",
                "completed",
                execution_time,
                metadata={
                    "platforms_checked": platforms,
                    "missing": missing_platforms,
                    "social_not_connected": state.get("social_not_connected", False),
                },
            )
            add_stream_event(
                state,
                "node_completed",
                content={
                    "data_available": data_available,
                    "missing": missing_platforms,
                },
                node="data_checker",
            )

            logger.info(
                "Data check complete",
                platforms=platforms,
                data_available=data_available,
                social_not_connected=state.get("social_not_connected", False),
            )

        except Exception as e:
            logger.error("Data check failed", error=str(e))
            state["data_available"] = {}
            state["missing_data_platforms"] = []
            state["errors"].append(f"Data check error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "data_checker", "failed", execution_time, str(e))

        return state

    def _get_platforms(self, state: ConversationState) -> list[str]:
        """Extract platforms from state."""
        platforms = []

        queries = state.get("decomposed_queries", [])
        for q in queries:
            platform = q.get("platform")
            if platform and platform != "general":
                platforms.append(platform)

        if not platforms:
            intent = (state.get("current_intent") or {})
            entities = (intent.get("entities") or {})

            if entities.get("platforms"):
                platforms.extend(entities["platforms"])
            elif entities.get("platform"):
                platforms.append(entities["platform"])

        if not platforms:
            platforms = ["linkedin"]

        return list(set(platforms))

    async def _check_platform_data(self, user_id: str, platform: str) -> bool:
        """Check if user has scraped social data in social_memory_test collection."""
        from app.core.vector_store import vector_store

        try:
            return await vector_store.has_social_content(
                user_id=user_id, platform=platform
            )
        except Exception as e:
            logger.warning(f"Error checking social content: {e}")
            return False


# Create node instance
data_checker_node = DataCheckerNode()
