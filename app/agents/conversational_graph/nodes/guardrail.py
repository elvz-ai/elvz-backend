"""
Guardrail Node - Content Safety Check.

Uses AWS Bedrock Guardrails to check input content for safety.
"""

import time
from typing import Any

import structlog

from app.agents.conversational_graph.state import (
    ConversationState,
    add_execution_trace,
    add_stream_event,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)


class GuardrailNode:
    """
    Content safety check using AWS Bedrock Guardrails.

    Checks user input for:
    - Harmful content
    - Policy violations
    - PII exposure
    - Prompt injection attempts
    """

    def __init__(self):
        self.enabled = bool(settings.bedrock_guardrail_id)
        self.client = None

        if self.enabled:
            try:
                import boto3
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id or None,
                    aws_secret_access_key=settings.aws_secret_access_key or None,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self.enabled = False

    async def __call__(self, state: ConversationState) -> ConversationState:
        """
        Check content safety using AWS Bedrock Guardrails.

        Args:
            state: Current conversation state

        Returns:
            Updated state with guardrail results
        """
        start_time = time.time()
        state["current_node"] = "guardrail_check"

        add_stream_event(state, "node_started", node="guardrail_check")

        try:
            if not self.enabled:
                # Guardrails not configured, pass through
                state["guardrail_passed"] = True
                state["guardrail_action"] = "pass"
                logger.debug("Guardrails not enabled, passing through")

            else:
                # Apply Bedrock guardrail
                result = await self._apply_guardrail(state["current_input"])

                state["guardrail_passed"] = result["passed"]
                state["guardrail_violations"] = result.get("violations", [])
                state["guardrail_action"] = result.get("action", "pass")

                if not result["passed"]:
                    logger.warning(
                        "Guardrail blocked content",
                        violations=result["violations"],
                    )
                    # Set error response for blocked content
                    state["final_response"] = self._get_blocked_response(result["violations"])

            execution_time = int((time.time() - start_time) * 1000)
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
            add_stream_event(
                state,
                "node_completed",
                content={"passed": state["guardrail_passed"]},
                node="guardrail_check",
            )

        except Exception as e:
            logger.error("Guardrail check failed", error=str(e))
            # On error, pass through but log warning
            state["guardrail_passed"] = True
            state["guardrail_action"] = "error_passthrough"
            state["errors"].append(f"Guardrail error: {str(e)}")

            execution_time = int((time.time() - start_time) * 1000)
            add_execution_trace(state, "guardrail_check", "failed", execution_time, str(e))

        return state

    async def _apply_guardrail(self, content: str) -> dict:
        """
        Apply AWS Bedrock guardrail to content.

        Args:
            content: Content to check

        Returns:
            Dict with passed, violations, action
        """
        if not self.client:
            return {"passed": True, "action": "pass", "violations": []}

        try:
            response = self.client.apply_guardrail(
                guardrailIdentifier=settings.bedrock_guardrail_id,
                guardrailVersion=settings.bedrock_guardrail_version,
                source='INPUT',
                content=[{'text': {'text': content}}]
            )

            action = response.get('action', 'NONE')
            passed = action != 'GUARDRAIL_INTERVENED'

            violations = []
            if not passed:
                # Extract violations from assessments
                for assessment in response.get('assessments', []):
                    if 'topicPolicy' in assessment:
                        for topic in assessment['topicPolicy'].get('topics', []):
                            violations.append(topic.get('name', 'unknown'))
                    if 'contentPolicy' in assessment:
                        for filter_result in assessment['contentPolicy'].get('filters', []):
                            if filter_result.get('action') == 'BLOCKED':
                                violations.append(filter_result.get('type', 'unknown'))

            return {
                "passed": passed,
                "action": "block" if not passed else "pass",
                "violations": violations,
            }

        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            # On API error, use fallback
            return await self._fallback_check(content)

    async def _fallback_check(self, content: str) -> dict:
        """
        Fallback content check using simple keyword filtering.

        Args:
            content: Content to check

        Returns:
            Dict with passed, violations, action
        """
        # Simple keyword blocklist
        blocklist = [
            "hack", "exploit", "inject", "attack",
            "password", "credentials", "api_key",
        ]

        content_lower = content.lower()
        violations = [word for word in blocklist if word in content_lower]

        return {
            "passed": len(violations) == 0,
            "action": "block" if violations else "pass",
            "violations": violations,
        }

    def _get_blocked_response(self, violations: list[str]) -> str:
        """Get response message for blocked content."""
        if violations:
            return (
                "I'm sorry, but I can't process that request. "
                "Your message may contain content that violates our usage policies. "
                "Please rephrase your request and try again."
            )
        return (
            "I'm sorry, but I can't process that request at this time. "
            "Please try rephrasing your message."
        )


# Create node instance
guardrail_node = GuardrailNode()
