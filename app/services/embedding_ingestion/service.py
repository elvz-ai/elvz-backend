"""
Embedding Ingestion Service.

Main orchestrator for processing embedding ingestion requests.
Handles validation, embedding generation, and vector storage.
"""

import time
from typing import Optional
import structlog

from .schemas import (
    EmbeddingIngestionRequest,
    EmbeddingIngestionResponse,
    EmbeddingBlock,
    BlockFailure,
    TextContent,
    ImageContent,
    VideoContent,
    Modality,
)
from .vertex_embeddings import VertexEmbeddingService, get_embedding_service
from .vertex_vector_store import VertexVectorStore, get_vector_store

logger = structlog.get_logger(__name__)


class EmbeddingIngestionService:
    """
    Main service for processing embedding ingestion requests.
    
    Responsibilities:
    - Validate incoming requests
    - Route blocks by modality
    - Generate embeddings via Vertex AI
    - Store vectors with metadata
    - Handle partial failures gracefully
    """
    
    def __init__(
        self,
        embedding_service: Optional[VertexEmbeddingService] = None,
        vector_store: Optional[VertexVectorStore] = None
    ):
        """
        Initialize the ingestion service.
        
        Args:
            embedding_service: Optional custom embedding service (for testing)
            vector_store: Optional custom vector store (for testing)
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
    
    def _generate_vector_id(self, company_id: str, block_id: str) -> str:
        """
        Generate deterministic vector ID for upsert idempotency.
        
        Format: {company_id}:{block_id}
        
        This ensures:
        - Same content always maps to same ID
        - Re-sending overwrites instead of duplicating
        """
        return f"{company_id}:{block_id}"
    
    def _parse_content(self, block: EmbeddingBlock):
        """
        Parse and validate content based on modality.
        
        Args:
            block: The embedding block
            
        Returns:
            Properly typed content object
            
        Raises:
            ValueError: If content doesn't match modality
        """
        content = block.content
        
        # Content may already be properly typed from Pydantic
        if block.modality == Modality.TEXT:
            if isinstance(content, TextContent):
                return content
            elif isinstance(content, dict) and "text" in content:
                return TextContent(**content)
            else:
                raise ValueError(f"Invalid text content: {content}")
        
        elif block.modality == Modality.IMAGE:
            if isinstance(content, ImageContent):
                return content
            elif isinstance(content, dict) and "image_url" in content:
                return ImageContent(**content)
            else:
                raise ValueError(f"Invalid image content: {content}")
        
        elif block.modality == Modality.VIDEO:
            if isinstance(content, VideoContent):
                return content
            elif isinstance(content, dict) and "video_url" in content:
                return VideoContent(**content)
            else:
                raise ValueError(f"Invalid video content: {content}")
        
        else:
            raise ValueError(f"Unknown modality: {block.modality}")
    
    async def process_block(
        self, 
        block: EmbeddingBlock, 
        company_id: str
    ) -> Optional[str]:
        """
        Process a single embedding block.
        
        Args:
            block: The embedding block to process
            company_id: Company ID for vector ID generation
            
        Returns:
            None if successful, error message string if failed
        """
        start_time = time.time()
        
        try:
            # 1. Parse and validate content
            content = self._parse_content(block)
            
            # 2. Generate embedding
            embedding = await self.embedding_service.embed_block(
                modality=block.modality,
                content=content
            )
            
            # 3. Prepare metadata for storage
            # Convert Pydantic model to dict, handling aliases
            metadata_dict = block.metadata.model_dump(by_alias=True)
            
            # Flatten and ensure all values are serializable
            # Note: modality and platform are copied from BLOCK level (not metadata)
            # This allows frontend to skip these in metadata
            stored_metadata = {
                "company_id": metadata_dict.get("company_id", ""),
                "user_id": metadata_dict.get("user_id", ""),
                "block_type": block.block_type,  # Already a string
                "block_id": block.block_id,
                "modality": block.modality.value,  # From block level
                "platform": block.platform.value,  # From block level
                "source_collection": metadata_dict.get("source_collection", ""),
                "doc_id": metadata_dict.get("doc_id") or "",
                "post_type": metadata_dict.get("post_type") or "",
                "region": metadata_dict.get("region", ""),
                "language": metadata_dict.get("language", "en"),
                "topic_tags": metadata_dict.get("topicTags") or [],
                "updated_at": metadata_dict.get("updated_at", 0),
            }
            
            # 4. Generate deterministic vector ID
            vector_id = self._generate_vector_id(company_id, block.block_id)
            
            # 5. Upsert to vector store
            await self.vector_store.upsert(
                vector_id=vector_id,
                embedding=embedding,
                metadata=stored_metadata
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                "Block processed successfully",
                block_id=block.block_id,
                modality=block.modality.value,
                block_type=block.block_type,  # block_type is already a string
                vector_id=vector_id,
                elapsed_ms=elapsed_ms
            )
            
            return None  # Success
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Block processing failed",
                block_id=block.block_id,
                modality=block.modality.value,
                error=str(e),
                elapsed_ms=elapsed_ms
            )
            return str(e)
    
    async def ingest(
        self, 
        request: EmbeddingIngestionRequest
    ) -> EmbeddingIngestionResponse:
        """
        Process full ingestion request.
        
        Processes all blocks, collecting successes and failures.
        Supports partial success - continues processing on block failures.
        
        Args:
            request: The ingestion request from frontend
            
        Returns:
            Response with counts and failure details
        """
        start_time = time.time()
        
        logger.info(
            "Starting embedding ingestion",
            company_id=request.company_id,
            user_id=request.user_id,
            source=request.source.value,
            total_blocks=len(request.blocks),
            modalities=self._count_modalities(request.blocks)
        )
        
        failures: list[BlockFailure] = []
        embedded_count = 0
        
        # Process each block
        for i, block in enumerate(request.blocks):
            logger.debug(
                f"Processing block {i+1}/{len(request.blocks)}",
                block_id=block.block_id,
                modality=block.modality.value
            )
            
            error = await self.process_block(block, request.company_id)
            
            if error:
                failures.append(BlockFailure(
                    block_id=block.block_id, 
                    reason=error
                ))
            else:
                embedded_count += 1
        
        # Build response
        response = EmbeddingIngestionResponse(
            company_id=request.company_id,
            total_blocks=len(request.blocks),
            embedded_blocks=embedded_count,
            failed_blocks=len(failures),
            failures=failures if failures else None
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "Embedding ingestion complete",
            company_id=request.company_id,
            total=len(request.blocks),
            embedded=embedded_count,
            failed=len(failures),
            elapsed_ms=elapsed_ms
        )
        
        return response
    
    def _count_modalities(self, blocks: list[EmbeddingBlock]) -> dict[str, int]:
        """Count blocks by modality for logging."""
        counts = {"text": 0, "image": 0, "video": 0}
        for block in blocks:
            modality = block.modality.value
            if modality in counts:
                counts[modality] += 1
        return counts


# Singleton instance
_ingestion_service: EmbeddingIngestionService | None = None


def get_ingestion_service() -> EmbeddingIngestionService:
    """Get or create the ingestion service singleton."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = EmbeddingIngestionService()
    return _ingestion_service

