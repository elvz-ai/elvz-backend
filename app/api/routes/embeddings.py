"""
Embedding Ingestion API Routes.

Provides endpoints for frontend to send embedding blocks
for processing and storage in Vertex AI Vector Search.
"""

import json
from fastapi import APIRouter, HTTPException, Request, status
import structlog

from app.services.embedding_ingestion.schemas import (
    EmbeddingIngestionRequest,
    EmbeddingIngestionResponse,
)
from app.services.embedding_ingestion.service import get_ingestion_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/debug-ingest",
    summary="Debug: Log raw request body",
    description="Debug endpoint to see raw request before validation"
)
async def debug_ingest(request: Request):
    """
    Debug endpoint - logs the raw request body to help identify validation issues.
    """
    try:
        # Get raw body
        body = await request.body()
        body_str = body.decode("utf-8")
        
        # Try to parse as JSON
        try:
            body_json = json.loads(body_str)
            logger.info("=" * 60)
            logger.info("RAW REQUEST BODY RECEIVED:")
            logger.info("=" * 60)
            print("\n" + "=" * 60)
            print("RAW REQUEST BODY:")
            print("=" * 60)
            print(json.dumps(body_json, indent=2))
            print("=" * 60 + "\n")
            
            # Log key fields
            logger.info(
                "Request fields",
                has_company_id="company_id" in body_json,
                has_user_id="user_id" in body_json,
                has_region="region" in body_json,
                has_source="source" in body_json,
                has_blocks="blocks" in body_json,
                blocks_count=len(body_json.get("blocks", [])) if isinstance(body_json.get("blocks"), list) else "NOT_A_LIST",
            )
            
            # Check blocks structure
            if "blocks" in body_json and isinstance(body_json["blocks"], list):
                for i, block in enumerate(body_json["blocks"]):
                    logger.info(
                        f"Block {i} structure",
                        has_block_id="block_id" in block,
                        has_block_type="block_type" in block,
                        block_type=block.get("block_type"),
                        has_modality="modality" in block,
                        modality=block.get("modality"),
                        has_platform="platform" in block,
                        platform=block.get("platform"),
                        has_content="content" in block,
                        content_keys=list(block.get("content", {}).keys()) if isinstance(block.get("content"), dict) else "NOT_A_DICT",
                        has_metadata="metadata" in block,
                    )
            
            return {
                "status": "received",
                "body_length": len(body_str),
                "parsed": True,
                "fields": list(body_json.keys()),
                "blocks_count": len(body_json.get("blocks", [])) if isinstance(body_json.get("blocks"), list) else 0,
            }
            
        except json.JSONDecodeError as e:
            logger.error("JSON parse error", error=str(e), raw_body=body_str[:500])
            return {
                "status": "error",
                "error": "Invalid JSON",
                "details": str(e),
                "raw_body_preview": body_str[:500]
            }
            
    except Exception as e:
        logger.error("Debug endpoint error", error=str(e))
        return {"status": "error", "error": str(e)}


@router.post(
    "/ingest",
    response_model=EmbeddingIngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest embedding blocks",
    description="""
    Ingest embedding blocks from frontend for processing.
    
    **Supports:**
    - Text embeddings (mission, captions, descriptions)
    - Image embeddings (from Firebase URLs)
    - Video embeddings (with optional segment boundaries)
    
    **Features:**
    - Idempotent: Same payload produces same vector IDs (upsert)
    - Partial success: Continues processing on block failures
    - Multi-modal: Text, image, and video in single request
    
    **Block Types:**
    - Company: company_profile, company_positioning, company_voice, etc.
    - Competitors: competitor_profile, competitor_swot, competitor_strategy
    - Instagram: ig_profile_text, ig_post_text, ig_post_image, ig_post_video_segment
    - Documents: document_text_chunk
    """
)
async def ingest_embeddings(
    request: EmbeddingIngestionRequest
) -> EmbeddingIngestionResponse:
    """
    Process embedding ingestion request.
    
    This endpoint:
    1. Validates the request payload
    2. Generates embeddings using Vertex AI multimodal model
    3. Stores vectors in Vertex AI Vector Search with metadata
    4. Returns success/failure counts
    
    Args:
        request: EmbeddingIngestionRequest with blocks to process
        
    Returns:
        EmbeddingIngestionResponse with processing results
        
    Raises:
        HTTPException 400: Invalid request payload
        HTTPException 500: Server error during processing
    """
    try:
        logger.info(
            "Embedding ingestion request received",
            company_id=request.company_id,
            user_id=request.user_id,
            blocks=len(request.blocks),
            source=request.source.value
        )
        
        # Get the ingestion service
        service = get_ingestion_service()
        
        # Process the request
        response = await service.ingest(request)
        
        # Log summary
        if response.failed_blocks > 0:
            logger.warning(
                "Ingestion completed with failures",
                company_id=response.company_id,
                embedded=response.embedded_blocks,
                failed=response.failed_blocks
            )
        else:
            logger.info(
                "Ingestion completed successfully",
                company_id=response.company_id,
                embedded=response.embedded_blocks
            )
        
        return response
        
    except ValueError as e:
        # Validation errors
        logger.warning("Invalid ingestion request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Server errors
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the embedding service is ready"
)
async def health_check():
    """
    Health check endpoint for the embedding service.
    
    Returns:
        Dict with status and service readiness
    """
    try:
        # Just check if the service can be instantiated
        service = get_ingestion_service()
        return {
            "status": "healthy",
            "service": "embedding_ingestion",
            "ready": True
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "embedding_ingestion",
            "ready": False,
            "error": str(e)
        }

