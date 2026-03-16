"""
Artifact API routes.

Endpoints for managing generated artifacts.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_user_id
from app.services.artifact_service import artifact_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


class ArtifactFeedbackRequest(BaseModel):
    """Request for artifact feedback."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None
    was_edited: bool = False
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
        artifact = await artifact_service.get_artifact(artifact_id)

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

        # Update artifact with feedback
        updated_artifact = await artifact_service.update_artifact_feedback(
            artifact_id=artifact_id,
            rating=feedback_request.rating,
            feedback=feedback_request.feedback,
            was_edited=feedback_request.was_edited,
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
    Update artifact content and track the edit for brain training.

    Send only the fields that changed (text, hashtags, image_url, schedule).
    Omitted fields keep their current values. The server merges updates into
    the existing content JSONB, computes a field-level diff, and appends it
    to edit_history. The original AI-generated content is preserved in
    original_content.

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

        artifact = await artifact_service.get_artifact(artifact_id)

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

        # Get all artifacts in batch
        artifacts = await artifact_service.get_batch_artifacts(batch_id)

        return {
            **batch.to_dict(),
            "artifacts": [a.to_dict() for a in artifacts],
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

        # Filter to only this user's artifacts
        artifacts = [a for a in artifacts if a.user_id == user_id]

        return {
            "conversation_id": conversation_id,
            "artifacts": [a.to_dict() for a in artifacts],
            "total": len(artifacts),
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
