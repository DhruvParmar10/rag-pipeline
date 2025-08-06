import asyncio
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, Range, MatchValue, PayloadSchemaType
)
from loguru import logger

from config import get_settings
from models import DocumentChunk, SearchResult

settings = get_settings()

class VectorStore:
    """Qdrant vector database service for semantic search"""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.embedding_dimension
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client and create collection if needed"""
        try:
            settings = get_settings()
            
            # Check if it's a cloud URL (contains https://)
            if settings.qdrant_host.startswith('https://'):
                # For Qdrant Cloud, use url parameter
                self.client = QdrantClient(
                    url=settings.qdrant_host,
                    api_key=settings.qdrant_api_key
                )
                logger.info(f"Connected to Qdrant Cloud at {settings.qdrant_host}")
            elif settings.qdrant_api_key:
                # For self-hosted with API key
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key
                )
                logger.info(f"Connected to Qdrant with API key at {settings.qdrant_host}:{settings.qdrant_port}")
            else:
                # For local instance without API key
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port
                )
                logger.info(f"Connected to local Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload index for document_url field to enable filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_url",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"Collection {self.collection_name} created successfully with payload index")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]], document_url: str):
        """Store document chunks with their embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "document_url": document_url,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)
            
            # Store points in batch - run in executor for non-blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            )
            
            logger.info(f"Stored {len(points)} chunks for document: {document_url}")
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise

    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_url: Optional[str] = None,
        top_k: int = None,
        threshold: float = None
    ) -> List[SearchResult]:
        """Enhanced search for similar chunks using vector similarity"""
        if top_k is None:
            top_k = settings.top_k_chunks
        if threshold is None:
            threshold = settings.similarity_threshold
        
        try:
            # Create filter for specific document if provided
            query_filter = None
            if document_url:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_url",
                            match=MatchValue(value=document_url)
                        )
                    ]
                )
            
            # Search with a higher limit initially to allow for better filtering
            search_limit = min(top_k * 2, 100)
            
            # Perform vector search - run in executor for non-blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=search_limit,
                    score_threshold=0.0,  # No threshold - get all results
                    with_payload=True
                )
            )
            
            # Convert to SearchResult objects and apply final filtering
            results = []
            for result in search_results:
                # Accept all results regardless of threshold for better debugging
                search_result = SearchResult(
                    chunk_id=result.payload["chunk_id"],
                    content=result.payload["content"],
                    score=result.score,
                    metadata={
                        **result.payload.get("metadata", {}),
                        "document_url": result.payload.get("document_url"),
                        "chunk_index": result.payload.get("chunk_index")
                    }
                )
                results.append(search_result)
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            final_results = results[:top_k]
            
            logger.info(f"Enhanced search found {len(final_results)} chunks (threshold: {threshold:.3f})")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []

    async def delete_document_chunks(self, document_url: str) -> int:
        """Delete all chunks for a specific document"""
        try:
            loop = asyncio.get_event_loop()
            
            # Search for all points with the document URL
            search_results = await loop.run_in_executor(
                None,
                lambda: self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_url",
                                match=MatchValue(value=document_url)
                            )
                        ]
                    ),
                    limit=10000  # Adjust based on expected document size
                )
            )
            
            point_ids = [point.id for point in search_results[0]]
            
            if point_ids:
                await loop.run_in_executor(
                    None,
                    lambda: self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                )
                logger.info(f"Deleted {len(point_ids)} chunks for document: {document_url}")
            
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            # Use a simpler approach to avoid validation issues
            loop = asyncio.get_event_loop()
            
            # Get basic collection count first
            count_info = await loop.run_in_executor(
                None,
                lambda: self.client.count(self.collection_name)
            )
            
            result = {
                "collection_name": self.collection_name,
                "points_count": count_info.count if hasattr(count_info, 'count') else 0,
                "status": "available"
            }
            
            # Try to get additional info if possible
            try:
                collections = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_collections()
                )
                
                # Find our collection in the list
                for collection in collections.collections:
                    if collection.name == self.collection_name:
                        result["vector_size"] = getattr(collection, 'vectors_count', 'N/A')
                        break
                        
            except Exception as detail_error:
                logger.warning(f"Could not get detailed collection info: {detail_error}")
                # Continue with basic info
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "status": "unknown"
            }

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and responsive"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.get_collections()
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
