"""
Expert Persona Agent - Generates a dynamic, topic-specific system prompt
for the ContentAgent, replacing its generic static persona.

Runs in parallel with the PlannerAgent (zero latency cost).
Uses a capable model with a focused prompt to produce a rich, opinionated
expert identity that deeply understands the topic's sub-domains, audience,
and what actually drives engagement in that niche.
"""

import structlog

from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


PERSONA_SYSTEM_PROMPT = """You are a world-class prompt engineer who creates expert system prompts for AI social media writers.

Given a topic and platform, write a rich, opinionated expert persona (150-250 words) that will be used as the system prompt for an AI content writer.

The persona MUST include ALL of these elements:
1. **Sub-domains**: Name the specific niches, verticals, or sub-topics within the broader topic (e.g., for "blockchain": DeFi, NFTs, tokenomics, Layer 2s, smart contracts)
2. **Audience**: Describe the specific audience segments on this platform who care about this topic — their job titles, knowledge level, what they read, what they distrust
3. **What makes content PERFORM in this niche**: Not generic advice — name the exact patterns that drive engagement in this specific domain on this specific platform (e.g., "posts that name a specific protocol and explain its mechanism outperform posts that describe the category")
4. **POV and opinions**: What does the expert stand for? What do they reject? What conventional wisdom do they push back on?
5. **Writing mechanics**: How do they open posts? How do they structure arguments? What vocabulary signals credibility? What clichés do they never use?
6. **Credibility signals**: What specific details, data points, or references make the expert sound authoritative in this domain?

Output ONLY the persona text — no labels, no JSON, no preamble. Start directly with "You are..."."""


PERSONA_USER_PROMPT = """Generate an expert persona for a content writer creating a {platform} post about: {topic}

Content type: {content_type}

The persona should make the writer feel like a genuine insider in the "{topic}" space on {platform} — someone who knows the sub-communities, the debates, the key figures, and the content patterns that actually earn engagement from this audience."""


class ExpertPersonaAgent:
    """
    Generates a dynamic expert persona system prompt for the ContentAgent.

    Runs in parallel with PlannerAgent so there is zero latency overhead.
    Uses a capable content-generation model to produce rich, topic-specific output.
    """

    name = "expert_persona_agent"

    async def generate(
        self,
        topic: str,
        platform: str,
        content_type: str = "thought_leadership",
    ) -> str:
        """
        Generate a topic-specific expert persona string.

        Args:
            topic: The content topic (e.g., "AI and blockchain")
            platform: Target platform (e.g., "linkedin")
            content_type: Type of content being created

        Returns:
            A rich expert persona string to use as the ContentAgent system prompt,
            or empty string if generation fails (fallback to static prompt).
        """
        if not topic:
            return ""

        logger.info(
            "Expert persona agent generating",
            topic=topic[:60],
            platform=platform,
        )

        messages = [
            LLMMessage(role="system", content=PERSONA_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=PERSONA_USER_PROMPT.format(
                    platform=platform,
                    topic=topic,
                    content_type=content_type,
                ),
            ),
        ]

        from app.core.model_config import TaskType

        try:
            response = await llm_client.generate_for_task(
                task=TaskType.CONTENT_GENERATION,
                messages=messages,
                json_mode=False,  # Plain text output, not JSON
            )

            persona = (response.content or "").strip()
            logger.info("Expert persona generated", persona=persona)

            if not persona.lower().startswith("you are"):
                # Unexpected format — discard and let content.py fall back to static prompt
                logger.warning("Expert persona had unexpected format, discarding")
                return ""

            logger.info(
                "Expert persona generated",
                topic=topic[:60],
                persona_length=len(persona),
                preview=persona[:120],
            )
            return persona

        except Exception as e:
            logger.error("Expert persona generation failed", error=str(e))
            return ""
