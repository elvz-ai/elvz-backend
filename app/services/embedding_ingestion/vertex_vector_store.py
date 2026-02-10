"""
Vertex AI Vector Search Client.

Handles storage and retrieval of vectors in Vertex AI Vector Search
(formerly Matching Engine).
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Optional
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


def _setup_google_credentials():
    """Set up Google Cloud credentials from service account JSON."""
    # Skip if already set
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    
    creds_path = settings.google_application_credentials
    
    # Handle relative paths (relative to project root)
    if not os.path.isabs(creds_path):
        project_root = Path(__file__).parent.parent.parent.parent
        creds_path = str(project_root / creds_path)
    
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        logger.debug("Google credentials configured", path=creds_path)


class VertexVectorStore:
    """
    Client for Vertex AI Vector Search.
    
    Handles:
    - Upserting vectors with metadata
    - Searching for similar vectors
    - Deleting vectors
    """
    
    def __init__(self):
        self._initialized = False
        self._index_endpoint = None
        self._deployed_index_id = None
    
    def _ensure_initialized(self):
        """Lazy initialization of Vector Search client."""
        if self._initialized:
            return
        
        try:
            # Set up credentials from service account JSON
            _setup_google_credentials()
            
            from google.cloud import aiplatform
            from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
            
            aiplatform.init(
                project=settings.google_cloud_project,
                location=settings.google_cloud_location
            )
            
            # Get the index endpoint
            endpoint_name = settings.vertex_vector_index_endpoint
            if endpoint_name:
                self._index_endpoint = MatchingEngineIndexEndpoint(
                    index_endpoint_name=endpoint_name
                )
                self._deployed_index_id = settings.vertex_deployed_index_id
                
                logger.info(
                    "Vertex Vector Search initialized",
                    endpoint=endpoint_name,
                    deployed_index=self._deployed_index_id
                )
            else:
                logger.warning(
                    "Vertex Vector Search not configured - running in dry-run mode"
                )
            
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize Vector Search", error=str(e))
            raise
    
    async def upsert(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any]
    ) -> None:
        """
        Upsert a single vector with metadata.
        
        Args:
            vector_id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Metadata to store with the vector
        """
        self._ensure_initialized()
        
        if self._index_endpoint is None:
            logger.warning(
                "Dry-run mode: would upsert vector",
                vector_id=vector_id,
                embedding_dim=len(embedding),
                metadata_keys=list(metadata.keys())
            )
            return
        
        try:
            def _upsert():
                # Convert metadata to restricts format for filtering
                restricts = self._metadata_to_restricts(metadata)
                
                # Prepare datapoint
                datapoint = {
                    "datapoint_id": vector_id,
                    "feature_vector": embedding,
                    "restricts": restricts,
                }
                
                # Upsert to the deployed index
                self._index_endpoint.upsert_datapoints(
                    deployed_index_id=self._deployed_index_id,
                    datapoints=[datapoint]
                )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _upsert)
            
            logger.debug(
                "Vector upserted",
                vector_id=vector_id,
                embedding_dim=len(embedding)
            )
            
        except Exception as e:
            logger.error(
                "Vector upsert failed",
                vector_id=vector_id,
                error=str(e)
            )
            raise
    
    async def upsert_batch(
        self,
        vectors: list[dict[str, Any]]
    ) -> None:
        """
        Upsert multiple vectors in batch.
        
        Args:
            vectors: List of dicts with 'id', 'embedding', 'metadata' keys
        """
        self._ensure_initialized()
        
        if self._index_endpoint is None:
            logger.warning(
                "Dry-run mode: would upsert batch",
                count=len(vectors)
            )
            return
        
        try:
            def _upsert_batch():
                datapoints = []
                for v in vectors:
                    restricts = self._metadata_to_restricts(v["metadata"])
                    datapoints.append({
                        "datapoint_id": v["id"],
                        "feature_vector": v["embedding"],
                        "restricts": restricts,
                    })
                
                self._index_endpoint.upsert_datapoints(
                    deployed_index_id=self._deployed_index_id,
                    datapoints=datapoints
                )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _upsert_batch)
            
            logger.debug("Batch upserted", count=len(vectors))
            
        except Exception as e:
            logger.error("Batch upsert failed", error=str(e))
            raise
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with 'id', 'score', 'metadata'
        """
        self._ensure_initialized()
        
        if self._index_endpoint is None:
            logger.warning("Dry-run mode: search not available")
            return []
        
        try:
            def _search():
                restricts = None
                if filter_metadata:
                    restricts = self._metadata_to_restricts(filter_metadata)
                
                response = self._index_endpoint.find_neighbors(
                    deployed_index_id=self._deployed_index_id,
                    queries=[query_embedding],
                    num_neighbors=top_k,
                    filter=restricts
                )
                
                results = []
                if response and len(response) > 0:
                    for neighbor in response[0]:
                        results.append({
                            "id": neighbor.id,
                            "score": neighbor.distance,
                            "metadata": {}  # Metadata not returned by default
                        })
                return results
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _search)
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            raise
    
    async def delete(self, vector_ids: list[str]) -> None:
        """
        Delete vectors by ID.
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        self._ensure_initialized()
        
        if self._index_endpoint is None:
            logger.warning(
                "Dry-run mode: would delete vectors",
                count=len(vector_ids)
            )
            return
        
        try:
            def _delete():
                self._index_endpoint.delete_datapoints(
                    deployed_index_id=self._deployed_index_id,
                    datapoint_ids=vector_ids
                )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _delete)
            
            logger.debug("Vectors deleted", count=len(vector_ids))
            
        except Exception as e:
            logger.error("Vector deletion failed", error=str(e))
            raise
    
    def _metadata_to_restricts(self, metadata: dict[str, Any]) -> list[dict]:
        """
        Convert metadata dict to Vertex AI restricts format.
        
        Vertex AI Vector Search uses "restricts" for filtering:
        - namespace: the field name
        - allow_list: allowed values (for equality)
        - deny_list: denied values (for exclusion)
        """
        restricts = []
        
        for key, value in metadata.items():
            if value is None:
                continue
            
            if isinstance(value, list):
                # For list values, allow any of the values
                if len(value) > 0:
                    restricts.append({
                        "namespace": key,
                        "allow_list": [str(v) for v in value]
                    })
            else:
                # For scalar values, single-item allow list
                restricts.append({
                    "namespace": key,
                    "allow_list": [str(value)]
                })
        
        return restricts


# Singleton instance
_vector_store: VertexVectorStore | None = None


def get_vector_store() -> VertexVectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VertexVectorStore()
    return _vector_store

