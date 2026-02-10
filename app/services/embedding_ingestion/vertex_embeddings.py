"""
Vertex AI Multimodal Embeddings Client.

Generates embeddings for text, images, and videos using
Vertex AI's multimodal embedding model.
"""

import asyncio
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Union
import structlog

from app.core.config import settings
from .schemas import Modality, TextContent, ImageContent, VideoContent

logger = structlog.get_logger(__name__)

# Vertex AI Multimodal embedding has a 1024 character limit for text
MAX_TEXT_LENGTH = 1024


def _setup_google_credentials():
    """Set up Google Cloud credentials from service account JSON."""
    creds_path = settings.google_application_credentials
    
    # Handle relative paths (relative to project root)
    if not os.path.isabs(creds_path):
        # Get project root (assuming this file is in app/services/embedding_ingestion/)
        project_root = Path(__file__).parent.parent.parent.parent
        creds_path = str(project_root / creds_path)
    
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        logger.info("Google credentials configured", path=creds_path)
    else:
        logger.warning(
            "Google credentials file not found",
            path=creds_path,
            hint="Set GOOGLE_APPLICATION_CREDENTIALS env var or place credentials file"
        )


class VertexEmbeddingService:
    """
    Generate embeddings using Vertex AI Multimodal Embeddings.
    
    Supports:
    - Text embeddings
    - Image embeddings (from URLs)
    - Video embeddings (with optional segment boundaries)
    """
    
    def __init__(self):
        self._initialized = False
        self._model = None
        # Multimodal embedding dimension (1408 for multimodalembedding@001)
        self.dimension = 1408
    
    def _ensure_initialized(self):
        """Lazy initialization of Vertex AI."""
        if self._initialized:
            return
        
        try:
            # Set up credentials from service account JSON
            _setup_google_credentials()
            
            import vertexai
            from vertexai.vision_models import MultiModalEmbeddingModel
            
            vertexai.init(
                project=settings.google_cloud_project,
                location=settings.google_cloud_location
            )
            
            self._model = MultiModalEmbeddingModel.from_pretrained(
                settings.vertex_embedding_model
            )
            self._initialized = True
            
            logger.info(
                "Vertex AI initialized",
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                model=settings.vertex_embedding_model
            )
        except Exception as e:
            logger.error("Failed to initialize Vertex AI", error=str(e))
            raise
    
    async def embed_text(self, content: TextContent) -> list[float]:
        """
        Generate embedding for text content.
        
        Args:
            content: TextContent with text field
            
        Returns:
            List of floats representing the embedding vector
            
        Note:
            Vertex AI multimodal embedding has a 1024 character limit.
            Text exceeding this limit is truncated.
        """
        self._ensure_initialized()
        
        try:
            # Truncate text if it exceeds the limit
            text = content.text
            was_truncated = False
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
                was_truncated = True
                logger.debug(
                    "Text truncated for embedding",
                    original_length=len(content.text),
                    truncated_length=MAX_TEXT_LENGTH
                )
            
            # Run in thread pool since Vertex AI SDK is synchronous
            def _embed():
                embeddings = self._model.get_embeddings(
                    contextual_text=text
                )
                return list(embeddings.text_embedding)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _embed)
            
            logger.debug(
                "Text embedding generated",
                text_length=len(content.text),
                was_truncated=was_truncated,
                embedding_dim=len(result)
            )
            return result
            
        except Exception as e:
            logger.error("Text embedding failed", error=str(e))
            raise
    
    async def embed_image(self, content: ImageContent) -> list[float]:
        """
        Generate embedding for image from URL.
        
        Supports:
        - GCS URLs (gs://bucket/path) - loaded directly
        - HTTP/HTTPS URLs - downloaded to temp file first
        
        Args:
            content: ImageContent with image_url field
            
        Returns:
            List of floats representing the embedding vector
        """
        self._ensure_initialized()
        
        try:
            from vertexai.vision_models import Image
            
            def _embed():
                url = content.image_url
                
                # GCS URLs can be loaded directly
                if url.startswith("gs://"):
                    image = Image.load_from_file(url)
                else:
                    # HTTP URLs need to be downloaded first
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        try:
                            # Download the image
                            urllib.request.urlretrieve(url, tmp.name)
                            image = Image.load_from_file(tmp.name)
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp.name):
                                os.unlink(tmp.name)
                
                embeddings = self._model.get_embeddings(image=image)
                return list(embeddings.image_embedding)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _embed)
            
            logger.debug(
                "Image embedding generated",
                image_url=content.image_url[:50] + "...",
                embedding_dim=len(result)
            )
            return result
            
        except Exception as e:
            logger.error(
                "Image embedding failed",
                image_url=content.image_url[:50] + "...",
                error=str(e)
            )
            raise
    
    async def embed_video(self, content: VideoContent) -> list[float]:
        """
        Generate embedding for video segment.
        
        Supports:
        - GCS URLs (gs://bucket/path) - loaded directly
        - HTTP/HTTPS URLs - downloaded to temp file first
        
        If start_ms and end_ms are provided, embeds only that segment.
        Otherwise, embeds the first segment (up to 2 minutes).
        
        Args:
            content: VideoContent with video_url and optional segment bounds
            
        Returns:
            List of floats representing the embedding vector
        """
        self._ensure_initialized()
        
        try:
            from vertexai.vision_models import Video, VideoSegmentConfig
            
            def _embed():
                url = content.video_url
                temp_path = None
                
                try:
                    # GCS URLs can be loaded directly
                    if url.startswith("gs://"):
                        video = Video.load_from_file(url)
                    else:
                        # HTTP URLs need to be downloaded first
                        # Determine file extension from URL
                        ext = ".mp4"  # Default to mp4
                        for video_ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                            if video_ext in url.lower():
                                ext = video_ext
                                break
                        
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            temp_path = tmp.name
                            # Download the video
                            urllib.request.urlretrieve(url, temp_path)
                            video = Video.load_from_file(temp_path)
                    
                    # Configure video segment
                    video_segment_config = None
                    if content.start_ms is not None and content.end_ms is not None:
                        video_segment_config = VideoSegmentConfig(
                            start_offset_sec=content.start_ms / 1000.0,
                            end_offset_sec=content.end_ms / 1000.0,
                        )
                    else:
                        # Default: first 1 second (minimal segment for faster processing)
                        video_segment_config = VideoSegmentConfig(
                            end_offset_sec=1
                        )
                    
                    embeddings = self._model.get_embeddings(
                        video=video,
                        video_segment_config=video_segment_config
                    )
                    
                    # Return first segment embedding
                    if embeddings.video_embeddings:
                        return list(embeddings.video_embeddings[0].embedding)
                    raise ValueError("No video embeddings generated")
                    
                finally:
                    # Clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _embed)
            
            logger.debug(
                "Video embedding generated",
                video_url=content.video_url[:50] + "...",
                embedding_dim=len(result)
            )
            return result
            
        except Exception as e:
            logger.error(
                "Video embedding failed",
                video_url=content.video_url[:50] + "...",
                error=str(e)
            )
            raise
    
    async def embed_block(
        self, 
        modality: Modality, 
        content: Union[TextContent, ImageContent, VideoContent]
    ) -> list[float]:
        """
        Route to correct embedding method based on modality.
        
        Args:
            modality: The content modality (text, image, video)
            content: The content payload matching the modality
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If modality is unknown or content type mismatches
        """
        if modality == Modality.TEXT:
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent for text modality, got {type(content)}")
            return await self.embed_text(content)
        
        elif modality == Modality.IMAGE:
            if not isinstance(content, ImageContent):
                raise ValueError(f"Expected ImageContent for image modality, got {type(content)}")
            return await self.embed_image(content)
        
        elif modality == Modality.VIDEO:
            if not isinstance(content, VideoContent):
                raise ValueError(f"Expected VideoContent for video modality, got {type(content)}")
            return await self.embed_video(content)
        
        else:
            raise ValueError(f"Unknown modality: {modality}")


# Singleton instance
_embedding_service: VertexEmbeddingService | None = None


def get_embedding_service() -> VertexEmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = VertexEmbeddingService()
    return _embedding_service

