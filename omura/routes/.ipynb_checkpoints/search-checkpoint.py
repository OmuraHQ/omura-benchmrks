"""Search API endpoints for vector similarity search."""

from __future__ import annotations

import threading
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from omura.utils.imagebind_embeddings import generate_text_embedding
from omura.utils.vector_store import VectorStore


router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        503: {"description": "Service Unavailable - Vector store not initialized"},
        500: {"description": "Internal Server Error"},
    },
)

# Global shared vector store instance
_shared_vector_store: Optional[VectorStore] = None
_vector_store_lock = threading.Lock()


def set_shared_vector_store(store: VectorStore) -> None:
    """Set the shared vector store instance (called from API startup)."""
    global _shared_vector_store
    with _vector_store_lock:
        _shared_vector_store = store
        print("[Search] Shared vector store set")


def get_vector_store() -> VectorStore:
    """Get the shared vector store instance.
    
    The vector store is initialized at startup and kept in memory.
    It's shared between the API and indexer, so new embeddings are
    immediately visible without reloading.
    
    Returns:
        VectorStore instance
    """
    global _shared_vector_store
    
    with _vector_store_lock:
        if _shared_vector_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store not initialized. The service may still be starting up.",
            )
        return _shared_vector_store


class SearchRequest(BaseModel):
    """Search request model for text-to-image queries."""

    query: str = Field(
        ...,
        description="Natural language text query to search for images",
        examples=[
            "a cat playing with a ball",
            "sunset over mountains",
            "modern architecture building",
            "cute puppy in a garden",
        ],
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to return (1-100)",
        example=5,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "a cat playing with a ball",
                "top_k": 5,
            }
        }
    }


class SearchResult(BaseModel):
    """Individual search result with blob metadata and similarity score."""

    blob_id: str = Field(
        ...,
        description="Base64-encoded Walrus blob ID",
        example="DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
    )
    mime_type: str = Field(
        ...,
        description="MIME type of the blob",
        examples=["image/png", "image/jpeg", "image/gif"],
    )
    size: int = Field(..., description="Size of the blob in bytes", example=123456)
    similarity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Similarity score (0-100, where 100 is identical). Scores use *1000 for granularity then cap at 100. For text-to-image search, typical max is ~87.",
        example=87.3,
    )
    extension: Optional[str] = Field(
        None, description="File extension", examples=["png", "jpg", "gif"]
    )
    kind: Optional[str] = Field(
        None, description="Content kind", examples=["image"]
    )
    is_nsfw: bool = Field(
        False, description="Flag indicating potential NSFW content based on zero-shot classification"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "blob_id": "DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
                "mime_type": "image/png",
                "size": 123456,
                                "similarity": 85.0,
                "extension": "png",
                "kind": "image",
                "is_nsfw": False,
            }
        }
    }


class SearchResponse(BaseModel):
    """Search response containing results and metadata."""

    results: List[SearchResult] = Field(
        ..., description="List of search results sorted by similarity (descending)"
    )
    total: int = Field(..., description="Total number of results returned", example=5)

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "blob_id": "DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
                        "mime_type": "image/png",
                        "size": 123456,
                        "similarity": 92.5,
                        "extension": "png",
                        "kind": "image",
                    },
                    {
                        "blob_id": "ABC123...",
                        "mime_type": "image/jpeg",
                        "size": 98765,
                        "similarity": 0.78,
                        "extension": "jpg",
                        "kind": "image",
                    },
                ],
                "total": 2,
            }
        }
    }


class VectorStoreStats(BaseModel):
    """Vector store statistics and status."""

    total_embeddings: int = Field(
        ..., description="Total number of embeddings in the vector store", example=1234
    )
    index_built: bool = Field(
        ..., description="Whether the vector index has been built", example=True
    )
    status: Optional[str] = Field(
        None,
        description="Status message (e.g., 'not_initialized' if empty)",
        example="ready",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_embeddings": 1234,
                "index_built": True,
                "status": "ready",
            }
        }
    }


@router.post(
    "/",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search images using text query",
    description="""
    Perform a text-to-image search using ImageBind embeddings.

    This endpoint:
    1. Converts the text query into an embedding using ImageBind
    2. Searches the vector store for similar image embeddings
    3. Returns the top-k most similar images with similarity scores

    The similarity score ranges from 0 to 100, where 100 indicates perfect match. 
    Scores are multiplied by 1000 for granularity then capped at 100. For text-to-image search, typical max is ~87.
    Results are sorted by similarity in descending order.

    **Note**: The vector store must be populated by the indexer before searching.
    Check `/search/stats` to see if embeddings are available.
    """,
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "blob_id": "DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
                                "mime_type": "image/png",
                                "size": 123456,
                                "similarity": 0.92,
                                "extension": "png",
                                "kind": "image",
                            }
                        ],
                        "total": 1,
                    }
                }
            },
        },
        503: {"description": "Vector store is empty or not initialized"},
        500: {"description": "Internal server error during search"},
    },
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for images using a text query.

    Uses ImageBind to generate a text embedding, then searches the vector store
    for similar image embeddings using cosine similarity.
    
    The vector store is kept in memory and shared with the indexer, so new
    embeddings are immediately visible as they are indexed.
    """
    # Get shared vector store (always up-to-date, no reload needed)
    store = get_vector_store()
    
    # Check if store is empty
    if store.size() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is empty. The indexer may still be running. Check /search/stats for progress.",
        )
    
    # Index will be built automatically by store.search() if needed (like FAISS/Chroma)
    # No need to check or build explicitly here

    try:
        # Generate text embedding
        query_embedding = generate_text_embedding(request.query.strip().lower())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {e}",
        )

    # Search vector store
    try:
        results = store.search(query_embedding, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )

    # Check if query is NSFW-related
    nsfw_keywords = [
        "nsfw", "nudity", "naked", "nude", "explicit content", "pornography", "porn", 
        "sexual content", "sex", "adult content", "erotic", "gore", "violence", 
        "disturbing content", "sexual", "breasts", "genitals", "explicit", "xxx", 
        "mature content", "adult", "sexually explicit"
    ]
    query_lower = request.query.lower()
    is_nsfw_query = any(keyword in query_lower for keyword in nsfw_keywords)
    
    # Format results
    search_results = []
    for meta, similarity in results:
        # Convert cosine similarity (-1.0 to 1.0) to 0-100 scale
        # We treat anything <= 0 (orthogonal or opposite) as 0% match
        # Multiply by 1000 for better granularity, then cap at 100
        # For text-to-image search, max similarity might be ~0.087 (87 when *1000)
        # Values like 87 stay 87, but values like 106 get capped at 100
        raw_score = max(0.0, similarity) * 1000.0
        # Cap at 100 (so 87 stays 87, but 106 becomes 100)
        percentage_score = min(raw_score, 100.0)
        
        # Check NSFW status from metadata (calculated during indexing)
        is_nsfw = meta.get("is_nsfw", False)
        
        # If query is NSFW-related and similarity > 55%, flag as NSFW
        if is_nsfw_query and percentage_score > 55.0:
            is_nsfw = True
        
        search_results.append(
            SearchResult(
                blob_id=meta["blob_id"],
                mime_type="image/jpeg",  # Default since we only store essential metadata
                size=meta["size"],
                similarity=percentage_score,
                extension="jpg",  # Default since we only store essential metadata
                kind="image",  # Default since we only store essential metadata
                is_nsfw=is_nsfw,
            )
        )

    return SearchResponse(results=search_results, total=len(search_results))


@router.get(
    "/stats",
    response_model=VectorStoreStats,
    summary="Get vector store statistics",
    description="""
    Retrieve statistics about the vector store, including:
    - Total number of embeddings indexed
    - Whether the index has been built
    - Current status

    Use this endpoint to check if the indexer has populated the vector store
    and if it's ready for searching.
    """,
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_embeddings": 1234,
                        "index_built": True,
                        "status": "ready",
                    }
                }
            },
        },
    },
)
async def get_stats() -> VectorStoreStats:
    """Get statistics about the vector store.
    
    Automatically reloads the vector store to show current statistics,
    including newly indexed embeddings.
    """
    try:
        # Get shared vector store (always up-to-date)
        store = get_vector_store()
        return VectorStoreStats(
            total_embeddings=store.size(),
            index_built=store._is_built,
            status="ready" if store.size() > 0 else "indexing",
        )
    except HTTPException as e:
        # Vector store not initialized yet or error
        return VectorStoreStats(
            total_embeddings=0,
            index_built=False,
            status="not_initialized",
        )
    except Exception as e:
        # Any other error
        print(f"[Search] Error getting stats: {e}")
        return VectorStoreStats(
            total_embeddings=0,
            index_built=False,
            status="error",
        )