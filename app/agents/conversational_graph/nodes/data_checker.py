"""
Data Checker Node.

Verifies that required user data is available for content generation.
"""

import time

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.services.memory_manager import memory_manager

logger = structlog.get_logger(__name__)


class DataCheckerNode:
    """
    Checks availability of user data for content generation.

    Verifies:
    - User has content history for personalization
    - Social media handles are configured
    - Brand voice profile exists
    """

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Check data availability for platforms.

        Args:
            state: Current conversation state

        Returns:
            Updated state with data availability info
        """
        start_time = time.time()
        state["current_node"] = "data_checker"

        add_stream_event(state, "node_started", node="data_checker")

        try:
            user_id = state["user_id"]

            # Get platforms to check
            platforms = self._get_platforms(state)

            # Check data availability for each platform
            data_available = {}
            missing_platforms = []

            for platform in platforms:
                # Check if user has content for this platform
                has_data = await self._check_platform_data(user_id, platform)
                data_available[platform] = has_data

                if not has_data:
                    missing_platforms.append(platform)

            state["data_available"] = data_available
            state["missing_data_platforms"] = missing_platforms

            # Check if user profile exists
            has_profile = state.get("user_profile") is not None
            has_brand_voice = (
                state.get("user_profile", {}).get("brand_voice") is not None
                if has_profile else False
            )

            state["working_memory"]["has_user_profile"] = has_profile
            state["working_memory"]["has_brand_voice"] = has_brand_voice

            # Determine if we can proceed
            # We can proceed without data, but with reduced personalization
            can_proceed = True  # Always allow proceeding

            if missing_platforms:
                logger.info(
                    "Missing data for platforms",
                    missing=missing_platforms,
                    proceeding=can_proceed,
                )
                # Add note about reduced personalization
                state["working_memory"]["reduced_personalization"] = True
                state["working_memory"]["personalization_note"] = (
                    f"Note: Limited personalization for {', '.join(missing_platforms)} "
                    "due to missing content history."
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
            )

        except Exception as e:
            logger.error("Data check failed", error=str(e))
            # On error, assume data is available and proceed
            state["data_available"] = {}
            state["missing_data_platforms"] = []
            state["errors"].append(f"Data check error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "data_checker", "failed", execution_time, str(e))

        return state

    def _get_platforms(self, state: ConversationState) -> list[str]:
        """Extract platforms from state."""
        platforms = []

        # From decomposed queries
        queries = state.get("decomposed_queries", [])
        for q in queries:
            platform = q.get("platform")
            if platform and platform != "general":
                platforms.append(platform)

        # From intent entities
        if not platforms:
            intent = state.get("current_intent", {})
            entities = intent.get("entities", {})

            if entities.get("platforms"):
                platforms.extend(entities["platforms"])
            elif entities.get("platform"):
                platforms.append(entities["platform"])

        # Default to LinkedIn
        if not platforms:
            platforms = ["linkedin"]

        return list(set(platforms))

    async def _check_platform_data(self, user_id: str, platform: str) -> bool:
        """
        Check if user has data for a specific platform.

        Args:
            user_id: User identifier
            platform: Platform to check

        Returns:
            True if data exists
        """
        try:
            # Try to retrieve user content for this platform
            content = await memory_manager.retrieve_user_content(
                query="",  # Empty query to just check existence
                user_id=user_id,
                platform=platform,
                top_k=1,
            )

            return len(content) > 0

        except Exception as e:
            logger.warning(f"Error checking platform data: {e}")
            return False


# Create node instance
data_checker_node = DataCheckerNode()
