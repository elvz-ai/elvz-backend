"""
New Post wizard API routes.

Two endpoints:
1. POST /posts/generate — content generation (fresh, draft, refine)
2. POST /posts/{artifact_id}/generate-image — standalone AI image generation

Optimization: All DB writes are deferred to FastAPI BackgroundTasks.
The LLM calls don't need anything from the database, so the response
is built from in-memory data and returned immediately.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select

from app.agents.elves.social_media_manager import SocialMediaManagerElf
from app.agents.elves.social_media_manager.mini_agents.visual import VisualAgent
from app.api.deps import get_user_id
from app.core.database import get_db_context
from app.models.artifact import Artifact, ArtifactBatch, ArtifactStatus
from app.models.conversation import Conversation, ConversationStatus
from app.models.user import User
from app.schemas.posts import (
    GenerateImageRequest,
    GenerateImageResponse,
    GeneratePostArtifact,
    GeneratePostRequest,
    GeneratePostResponse,
)
from app.services.artifact_service import artifact_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/posts", tags=["posts"])

# Reuse the existing elf — no graph, direct call
_social_media_elf = SocialMediaManagerElf()
_visual_agent = VisualAgent()

# Tone chip → brand_voice mapping
TONE_MAP = {
    "professional": "professional, authoritative, polished",
    "engaging": "conversational, compelling, relatable",
    "concise": "concise, direct, minimal, no fluff",
    "witty": "witty, clever, light-hearted",
}


def _build_modification_feedback(tone: str, length: str) -> str:
    """Build modification instructions from wizard parameters."""
    parts = []

    if length == "shorter":
        parts.append(
            "Make this post significantly shorter and more concise "
            "while keeping the key message intact."
        )
    elif length == "longer":
        parts.append(
            "Expand this post with more detail, examples, and depth "
            "while keeping it engaging."
        )

    tone_label = TONE_MAP.get(tone, tone)
    parts.append(f"Apply a {tone_label} tone throughout.")

    return " ".join(parts)


async def _persist_all(
    user_id: str,
    conversation_id: str,
    batch_id: str,
    topic: Optional[str],
    platforms: list[str],
    artifacts_data: list[dict],
    execution_time_ms: int,
):
    """Background: persist everything in a single transaction after response is sent."""
    try:
        async with get_db_context() as session:
            # 1. Ensure user exists
            result = await session.execute(select(User).where(User.id == user_id))
            if result.scalar_one_or_none() is None:
                session.add(User(
                    id=user_id,
                    email=f"{user_id}@elvz.local",
                    name=user_id,
                ))

            # 2. Create headless conversation
            session.add(Conversation(
                id=conversation_id,
                user_id=user_id,
                thread_id=str(uuid.uuid4()),
                title=(topic or "New Post")[:255],
                status=ConversationStatus.ACTIVE.value,
                extra_metadata={"source": "wizard", "platforms": platforms},
            ))

            # 3. Create batch (already complete)
            session.add(ArtifactBatch(
                id=batch_id,
                conversation_id=conversation_id,
                platforms=platforms,
                topic=topic,
                status="complete",
                execution_strategy="parallel",
                execution_time_ms=execution_time_ms,
                completed_at=datetime.now(timezone.utc),
            ))

            # 4. Create all artifacts
            for data in artifacts_data:
                session.add(Artifact(
                    id=data["id"],
                    conversation_id=conversation_id,
                    batch_id=batch_id,
                    artifact_type="social_post",
                    platform=data["platform"],
                    content=data["content"],
                    status=ArtifactStatus.DRAFT.value,
                    generation_metadata=data["metadata"],
                ))

            # 5. Single COMMIT (get_db_context auto-commits on exit)

        logger.info(
            "Background persist completed",
            batch_id=batch_id,
            artifact_count=len(artifacts_data),
        )
    except Exception as e:
        logger.error(
            "Background persist failed",
            batch_id=batch_id,
            error=str(e),
        )


@router.post("/generate", response_model=GeneratePostResponse)
async def generate_post(
    request: GeneratePostRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id),
) -> GeneratePostResponse:
    """
    Generate social media content for the New Post wizard.

    Handles three scenarios via the same endpoint:
    - **Fresh generate**: idea + tone + length, no draft
    - **Use Draft**: draft = user's pasted text
    - **Refine**: draft = previously generated text, new tone/length

    All DB writes are deferred to background — response returns immediately
    after LLM generation.
    """
    start_time = time.time()

    # Generate all IDs upfront — no DB needed
    conversation_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())

    logger.info(
        "Wizard generate request",
        user_id=user_id,
        platforms=request.platforms,
        tone=request.tone,
        length=request.length,
        has_draft=request.draft is not None,
        batch_id=batch_id,
    )

    # Build base request for the elf
    brand_voice = TONE_MAP.get(request.tone, request.tone)
    base_request = {
        "topic": request.idea,
        "content_type": "thought_leadership",
        "brand_voice": brand_voice,
        "goals": ["engagement"],
    }

    # If draft provided, set up modification fields
    if request.draft:
        base_request["previous_content"] = request.draft
        base_request["modification_feedback"] = _build_modification_feedback(
            request.tone, request.length
        )
    elif request.length != "keep":
        # Fresh generation with length preference
        length_hint = {
            "shorter": "Keep the post brief and punchy — short paragraphs, minimal words.",
            "longer": "Write a detailed, comprehensive post with examples and depth.",
        }
        base_request["additional_context"] = length_hint.get(request.length, "")

    # Generate content for each platform in parallel
    async def _generate_for_platform(platform: str) -> Optional[dict]:
        try:
            elf_request = {
                **base_request,
                "platform": platform,
            }
            result = await _social_media_elf.execute(
                request=elf_request,
                context={
                    "user_id": user_id,
                    "image": "false",
                    "video": "false",
                },
            )
            return {"platform": platform, "result": result}
        except Exception as e:
            logger.error("Generation failed for platform", platform=platform, error=str(e))
            return None

    platform_results = await asyncio.gather(
        *[_generate_for_platform(p) for p in request.platforms]
    )

    # Build response from in-memory data — no DB needed
    artifacts_response: list[GeneratePostArtifact] = []
    artifacts_to_persist: list[dict] = []

    for pr in platform_results:
        if pr is None:
            continue

        platform = pr["platform"]
        result = pr["result"]

        # Extract content from the elf's response
        variations = result.get("post_variations", [])
        if not variations:
            continue

        variation = variations[0]
        content_data = variation.get("content", {})

        # Build artifact content
        artifact_content = {
            "post_text": content_data.get("post_text", ""),
            "hook": content_data.get("hook", ""),
            "cta": content_data.get("cta", ""),
            "hashtags": [
                h.get("tag", "") for h in variation.get("hashtags", [])
            ],
            "posting_schedule": variation.get("posting_schedule", {}),
        }

        artifact_id = str(uuid.uuid4())

        artifacts_response.append(
            GeneratePostArtifact(
                id=artifact_id,
                platform=platform,
                content=artifact_content,
                status="draft",
            )
        )

        artifacts_to_persist.append({
            "id": artifact_id,
            "platform": platform,
            "content": artifact_content,
            "metadata": {
                "source": "wizard",
                "tone": request.tone,
                "length": request.length,
                "had_draft": request.draft is not None,
                "execution_time_ms": result.get("execution_time_ms", 0),
            },
        })

    execution_time_ms = int((time.time() - start_time) * 1000)

    if not artifacts_response:
        raise HTTPException(
            status_code=500,
            detail="Content generation failed for all platforms",
        )

    # Defer ALL DB writes to background
    background_tasks.add_task(
        _persist_all,
        user_id=user_id,
        conversation_id=conversation_id,
        batch_id=batch_id,
        topic=request.idea[:500] if request.idea else None,
        platforms=request.platforms,
        artifacts_data=artifacts_to_persist,
        execution_time_ms=execution_time_ms,
    )

    return GeneratePostResponse(
        batch_id=batch_id,
        artifacts=artifacts_response,
        execution_time_ms=execution_time_ms,
    )


@router.post("/{artifact_id}/generate-image", response_model=GenerateImageResponse)
async def generate_image(
    artifact_id: str,
    request: GenerateImageRequest,
    user_id: str = Depends(get_user_id),
) -> GenerateImageResponse:
    """
    Generate an AI image and attach it to an existing artifact.

    Called from Step 3 of the wizard when user clicks "AI Generate".
    """
    logger.info(
        "Wizard image generation",
        user_id=user_id,
        artifact_id=artifact_id,
        prompt_length=len(request.prompt),
    )

    # Verify artifact exists
    artifact = await artifact_service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Generate image directly via the visual agent
    image_result = await _visual_agent._generate_image(
        description=request.prompt,
        style="Professional, high-quality social media visual",
        dimensions="1200 x 630",
    )

    if not image_result or not image_result.get("url"):
        raise HTTPException(
            status_code=500,
            detail="Image generation failed",
        )

    image_url = image_result["url"]

    # Update the artifact's content with the image URL
    updated_content = {**(artifact.content or {}), "image_url": image_url}
    await artifact_service.update_artifact(
        artifact_id=artifact_id,
        content=updated_content,
    )

    return GenerateImageResponse(
        image_url=image_url,
        artifact_id=artifact_id,
        credits_used=5,
    )
