"""
Artifact API routes.

Endpoints for managing generated artifacts.
"""

import time
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.agents.elves.social_media_manager.mini_agents.content import ContentAgent
from app.api.deps import get_user_id
from app.services.artifact_service import artifact_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


class ArtifactFeedbackRequest(BaseModel):
    """Request for artifact feedback."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None
    was_published: bool = False


class ArtifactContentUpdate(BaseModel):
    """Request for updating editable artifact fields.

    Only send the fields that changed. Omitted fields keep their current values.
    The server merges updates into the existing content JSONB and tracks the diff.
    """
    text: Optional[str] = None
    hashtags: Optional[list[str]] = None
    image_url: Optional[str] = None
    schedule: Optional[dict] = None


class OptimizeArtifactRequest(BaseModel):
    """Request for AI re-optimization of an artifact."""
    prompt: str = Field(..., min_length=3, max_length=2000)
    elf: str = Field(..., min_length=1, description="Elf agent slug")


_content_agent = ContentAgent()


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Get artifact details.

    Args:
        artifact_id: Artifact identifier
        user_id: Current user ID (from auth)

    Returns:
        Artifact details
    """
    try:
        # Returns latest version content, resolves transparently
        artifact = await artifact_service.get_current_artifact(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if artifact.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return artifact.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get artifact", error=str(e), artifact_id=artifact_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artifact_id}/feedback")
async def submit_feedback(
    artifact_id: str,
    feedback_request: ArtifactFeedbackRequest,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Submit feedback for an artifact.

    Args:
        artifact_id: Artifact identifier
        feedback_request: Feedback data
        user_id: Current user ID (from auth)

    Returns:
        Updated artifact
    """
    try:
        artifact = await artifact_service.get_artifact(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if artifact.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update artifact with feedback (operates on original row)
        updated_artifact = await artifact_service.update_artifact_feedback(
            artifact_id=artifact_id,
            rating=feedback_request.rating,
            feedback=feedback_request.feedback,
            was_published=feedback_request.was_published,
        )

        return updated_artifact.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to submit feedback",
            error=str(e),
            artifact_id=artifact_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{artifact_id}/content")
async def update_content(
    artifact_id: str,
    request: ArtifactContentUpdate,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Update artifact content by creating a new version row.

    Send only the fields that changed (text, hashtags, image_url, schedule).
    Omitted fields keep their current values. The server merges updates into
    the current content, computes a field-level diff, and creates a new
    version row with parent_artifact_id linking back to the original.

    Args:
        artifact_id: Artifact identifier
        request: Editable fields (text, hashtags, image_url, schedule)
        user_id: Current user ID (from auth)

    Returns:
        Updated artifact
    """
    try:
        # Build partial update from only non-None fields
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        artifact = await artifact_service.get_current_artifact(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if artifact.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        updated_artifact = await artifact_service.update_artifact_content(
            artifact_id=artifact_id,
            updates=updates,
        )

        return updated_artifact.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update artifact content",
            error=str(e),
            artifact_id=artifact_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artifact_id}/optimize")
async def optimize_artifact(
    artifact_id: str,
    request: OptimizeArtifactRequest,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Re-optimize an artifact via AI using a user prompt.

    Calls ContentAgent directly (no planner/persona/optimization) with the
    artifact's current content as previous_content and the user's prompt
    as modification_feedback. Creates a new version row with source="regeneration".
    """
    try:
        artifact = await artifact_service.get_current_artifact(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if artifact.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        current_content = artifact.content or {}
        current_text = current_content.get("text") or current_content.get("post_text") or ""

        start = time.monotonic()

        # Call ContentAgent directly (skip planner/persona/optimization)
        agent_state = {
            "user_request": {
                "platform": artifact.platform or "linkedin",
                "topic": current_text[:100],
                "content_type": "thought_leadership",
                "goals": ["engagement"],
            },
            "previous_content": current_text,
            "modification_feedback": request.prompt,
            "plan": {
                "content_strategy": f"Modify the existing post based on user instruction: {request.prompt}",
            },
        }
        agent_result = await _content_agent.execute(agent_state, {"user_id": user_id})

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Extract content from ContentAgent response — may be a string if JSON parsing failed
        content = agent_result.get("content") or {}
        if isinstance(content, str):
            updates = {"text": content, "hook": "", "cta": ""}
        else:
            updates = {
                "text": content.get("post_text") or content.get("text") or "",
                "hook": content.get("hook") or "",
                "cta": content.get("cta") or "",
            }
        if not updates.get("text"):
            raise HTTPException(
                status_code=500,
                detail="Optimization produced no content",
            )

        # Persist with source="regeneration" so the brain can distinguish
        updated_artifact = await artifact_service.update_artifact_content(
            artifact_id=artifact_id,
            updates=updates,
            source="regeneration",
            prompt=request.prompt,
        )

        # Return same shape as POST /posts/generate (id = original)
        return {
            "batch_id": updated_artifact.batch_id,
            "artifacts": [
                {
                    "id": updated_artifact.parent_artifact_id or updated_artifact.id,
                    "platform": updated_artifact.platform,
                    "content": updated_artifact.content,
                    "status": updated_artifact.status,
                }
            ],
            "execution_time_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to optimize artifact",
            error=str(e),
            artifact_id=artifact_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


def _extract_content(result: dict) -> dict:
    """Extract normalized content fields from a SocialMediaManagerElf response."""
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

    # Fallback: direct keys at result root
    content = {}
    if "content" in result:
        content.update(result["content"])
    elif "final_output" in result:
        output = result["final_output"]
        if isinstance(output, dict):
            content.update(output)

    content.setdefault("text", result.get("post_text") or "")
    content.setdefault("hashtags", result.get("hashtags") or [])
    content.setdefault("schedule", result.get("timing") or {})
    return content


@router.get("/{artifact_id}/versions")
async def get_artifact_versions(
    artifact_id: str,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Get version history for an artifact.

    Returns all versions ordered by created_at, including the original as version 0.
    Each version includes content, source, prompt, diff, message_id, and created_at.
    """
    try:
        artifact = await artifact_service.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Resolve original ID
        original_id = artifact.parent_artifact_id or artifact.id
        original = artifact if artifact.parent_artifact_id is None else await artifact_service.get_artifact(original_id)

        if not original or original.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        versions = await artifact_service.get_artifact_versions(original_id)

        version_list = []
        for i, v in enumerate(versions):
            entry = {
                "version": i,
                "version_id": v.id,
                "content": v.content,
                "edit_diff": v.edit_diff,
                "message_id": v.message_id,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            version_list.append(entry)

        return {
            "artifact_id": original_id,
            "versions": version_list,
            "current_version": len(version_list) - 1,
            "total_versions": len(version_list),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get versions", error=str(e), artifact_id=artifact_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/{batch_id}")
async def get_artifact_batch(
    batch_id: str,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Get artifact batch with all artifacts.

    Args:
        batch_id: Batch identifier
        user_id: Current user ID (from auth)

    Returns:
        Batch details with artifacts
    """
    try:
        batch = await artifact_service.get_batch(batch_id)

        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

        if batch.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get original artifacts in batch, resolve latest content for each
        originals = await artifact_service.get_batch_artifacts(batch_id)
        resolved = []
        for a in originals:
            current = await artifact_service.get_current_artifact(a.id)
            resolved.append((current or a).to_dict())

        return {
            **batch.to_dict(),
            "artifacts": resolved,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get batch", error=str(e), batch_id=batch_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def list_conversation_artifacts(
    conversation_id: str,
    user_id: str = Depends(get_user_id),
    platform: Optional[str] = Query(None),
    artifact_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """
    List artifacts for a conversation.

    Args:
        conversation_id: Conversation identifier
        user_id: Current user ID (from auth)
        platform: Filter by platform
        artifact_type: Filter by type
        limit: Max artifacts to return

    Returns:
        List of artifacts
    """
    try:
        # Get artifacts scoped to this user + conversation
        artifacts = await artifact_service.list_artifacts(
            conversation_id=conversation_id,
            platform=platform,
            artifact_type=artifact_type,
            limit=limit,
        )

        # Filter to only this user's artifacts, resolve latest content
        artifacts = [a for a in artifacts if a.user_id == user_id]
        resolved = []
        for a in artifacts:
            current = await artifact_service.get_current_artifact(a.id)
            resolved.append((current or a).to_dict())

        return {
            "conversation_id": conversation_id,
            "artifacts": resolved,
            "total": len(resolved),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to list artifacts",
            error=str(e),
            conversation_id=conversation_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{artifact_id}")
async def delete_artifact(
    artifact_id: str,
    user_id: str = Depends(get_user_id),
) -> dict:
    """
    Delete an artifact.

    Args:
        artifact_id: Artifact identifier
        user_id: Current user ID (from auth)

    Returns:
        Success message
    """
    try:
        artifact = await artifact_service.get_artifact(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if artifact.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        await artifact_service.delete_artifact(artifact_id)

        return {
            "message": "Artifact deleted",
            "artifact_id": artifact_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete artifact",
            error=str(e),
            artifact_id=artifact_id,
        )
        raise HTTPException(status_code=500, detail=str(e))
