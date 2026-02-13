"""
AWS Bedrock Guardrails Middleware.

Provides content safety checking using AWS Bedrock Guardrails
with fallback to keyword-based filtering.
"""

from typing import Optional

import boto3
import structlog
from botocore.exceptions import ClientError

from app.core.config import settings

logger = structlog.get_logger(__name__)


class GuardrailsMiddleware:
    """
    AWS Bedrock Guardrails wrapper for content safety.
    
    Features:
    - Harmful content detection
    - PII detection
    - Prompt injection detection
    - Policy violation detection
    - Fallback to keyword filtering
    """

    def __init__(self):
        self.enabled = bool(settings.bedrock_guardrail_id)
        self.client = None
        
        if self.enabled:
            try:
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id or None,
                    aws_secret_access_key=settings.aws_secret_access_key or None,
                )
                logger.info("AWS Bedrock Guardrails initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self.enabled = False

    async def check_content(
        self,
        content: str,
        source: str = "INPUT",
    ) -> dict:
        """
        Check content against guardrails.
        
        Args:
            content: Text content to check
            source: "INPUT" or "OUTPUT"
        
        Returns:
            dict with:
                - passed: bool
                - action: "BLOCKED" or "NONE"
                - violations: list of violation types
                - assessments: detailed assessment results
        """
        if not self.enabled:
            return await self._fallback_check(content)
        
        try:
            return await self._apply_guardrail(content, source)
        except Exception as e:
            logger.warning(f"Guardrail check failed, using fallback: {e}")
            return await self._fallback_check(content)

    async def _apply_guardrail(self, content: str, source: str) -> dict:
        """Apply AWS Bedrock Guardrail."""
        try:
            response = self.client.apply_guardrail(
                guardrailIdentifier=settings.bedrock_guardrail_id,
                guardrailVersion=settings.bedrock_guardrail_version,
                source=source,
                content=[
                    {
                        "text": {
                            "text": content
                        }
                    }
                ]
            )
            
            action = response.get("action", "NONE")
            passed = action != "GUARDRAIL_INTERVENED"
            
            violations = []
            assessments = response.get("assessments", [])
            
            for assessment in assessments:
                # Check for harmful content
                if "contentPolicy" in assessment:
                    for filter_result in assessment["contentPolicy"].get("filters", []):
                        if filter_result.get("action") == "BLOCKED":
                            violations.append(f"harmful_content_{filter_result.get('type', 'unknown')}")
                
                # Check for PII
                if "sensitiveInformationPolicy" in assessment:
                    for pii_entity in assessment["sensitiveInformationPolicy"].get("piiEntities", []):
                        if pii_entity.get("action") == "BLOCKED":
                            violations.append(f"pii_{pii_entity.get('type', 'unknown')}")
                
                # Check for prompt injection
                if "contextualGroundingPolicy" in assessment:
                    for filter_result in assessment["contextualGroundingPolicy"].get("filters", []):
                        if filter_result.get("action") == "BLOCKED":
                            violations.append("prompt_injection")
            
            logger.info(
                "Guardrail check completed",
                passed=passed,
                action=action,
                violations=violations,
            )
            
            return {
                "passed": passed,
                "action": action,
                "violations": violations,
                "assessments": assessments,
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"Bedrock API error: {error_code} - {str(e)}")
            raise

    async def _fallback_check(self, content: str) -> dict:
        """
        Fallback keyword-based content check.
        
        Simple check for obviously harmful content.
        """
        content_lower = content.lower()
        
        # Basic harmful keywords (this is a minimal set)
        harmful_keywords = [
            "hack", "exploit", "malware", "phishing",
            "illegal", "fraud", "scam",
        ]
        
        violations = []
        for keyword in harmful_keywords:
            if keyword in content_lower:
                violations.append(f"keyword_{keyword}")
        
        passed = len(violations) == 0
        action = "BLOCKED" if not passed else "NONE"
        
        return {
            "passed": passed,
            "action": action,
            "violations": violations,
            "assessments": [],
            "fallback": True,
        }

    async def check_input(self, user_input: str) -> dict:
        """Check user input content."""
        return await self.check_content(user_input, source="INPUT")

    async def check_output(self, model_output: str) -> dict:
        """Check model output content."""
        return await self.check_content(model_output, source="OUTPUT")


# Global instance
guardrails_middleware = GuardrailsMiddleware()
