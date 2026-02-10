"""
Embedding Ingestion Service.

Handles ingestion of text, image, and video embeddings
using Vertex AI and stores them in Vertex AI Vector Search.
"""

from .service import EmbeddingIngestionService
from .schemas import (
    EmbeddingIngestionRequest,
    EmbeddingIngestionResponse,
    EmbeddingBlock,
    BlockMetadata,
    TextContent,
    ImageContent,
    VideoContent,
    Modality,
    Platform,
)

__all__ = [
    "EmbeddingIngestionService",
    "EmbeddingIngestionRequest",
    "EmbeddingIngestionResponse",
    "EmbeddingBlock",
    "BlockMetadata",
    "TextContent",
    "ImageContent",
    "VideoContent",
    "Modality",
    "Platform",
]

