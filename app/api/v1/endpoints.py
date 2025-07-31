"""
Main API Endpoints
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    ExplanationRequest,
    ExplanationResponse,
    SystemStatsResponse,
    HealthCheckResponse,
    ErrorResponse
)
from app.services.query_processor import QueryProcessingService
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global service instance
query_service = QueryProcessingService(use_pinecone=bool(settings.PINECONE_API_KEY))


@router.post(
    "/hackrx/run",
    response_model=QueryResponse,
    summary="Process documents and answer questions",
    description="Main endpoint that processes documents and answers questions using LLM and vector search"
)
async def run_query_processing(request: QueryRequest) -> QueryResponse:
    """
    Process documents and answer questions
    
    This is the main API endpoint that:
    1. Downloads and processes documents from the provided URL
    2. Creates embeddings and indexes them for semantic search
    3. Analyzes each question using LLM
    4. Performs vector search to find relevant document sections
    5. Generates answers using the LLM with retrieved context
    
    Args:
        request: QueryRequest containing document URL and list of questions
        
    Returns:
        QueryResponse with answers and processing metadata
    """
    try:
        logger.info(f"Processing query with document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Convert single document URL to list for processing
        document_urls = [request.documents] if isinstance(request.documents, str) else [request.documents]
        
        # Process documents and questions
        result = await query_service.process_documents_and_queries(
            document_urls=document_urls,
            questions=request.questions
        )
        
        # Handle errors in processing
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result['error']}"
            )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_query_processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and process a document",
    description="Upload a document and process it for later querying"
)
async def upload_document(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """
    Upload and process a document
    
    Args:
        request: DocumentUploadRequest with document URL
        
    Returns:
        DocumentUploadResponse with processing results
    """
    try:
        start_time = time.time()
        
        # Process the document
        processed_doc = await query_service.document_processor.process_document(request.document_url)
        
        # Add to vector index
        success = query_service.vector_search.add_documents([processed_doc])
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to vector index"
            )
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            success=True,
            document_id=request.document_id or processed_doc["url"],
            file_type=processed_doc["file_type"],
            num_chunks=processed_doc["num_chunks"],
            total_length=processed_doc["total_length"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents",
    description="Perform semantic search across indexed documents"
)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search documents using vector similarity
    
    Args:
        request: SearchRequest with query and parameters
        
    Returns:
        SearchResponse with search results
    """
    try:
        start_time = time.time()
        
        # Perform search
        results = query_service.vector_search.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Generate explanation",
    description="Generate detailed explanation for an answer"
)
async def generate_explanation(request: ExplanationRequest) -> ExplanationResponse:
    """
    Generate explanation for an answer
    
    Args:
        request: ExplanationRequest with question, answer, and clauses
        
    Returns:
        ExplanationResponse with detailed explanation
    """
    try:
        explanation = await query_service.get_explanation(
            question=request.question,
            answer=request.answer,
            relevant_clauses=request.relevant_clauses
        )
        
        return ExplanationResponse(
            explanation=explanation,
            reasoning_steps=[],
            cited_clauses=request.relevant_clauses
        )
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics",
    description="Get comprehensive system statistics and health information"
)
async def get_system_stats() -> SystemStatsResponse:
    """
    Get system statistics
    
    Returns:
        SystemStatsResponse with system statistics
    """
    try:
        stats = query_service.get_system_stats()
        return SystemStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system stats: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of all system components"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        HealthCheckResponse with system health status
    """
    try:
        # Check component health
        components = {}
        
        # Check LLM service
        try:
            await query_service.llm_service.generate_response("Hello", max_tokens=10)
            components["llm"] = "healthy"
        except Exception as e:
            components["llm"] = f"unhealthy: {str(e)}"
        
        # Check vector search
        try:
            stats = query_service.vector_search.get_stats()
            components["vector_search"] = "healthy" if "error" not in stats else "unhealthy"
        except Exception as e:
            components["vector_search"] = f"unhealthy: {str(e)}"
        
        # Check document processor
        try:
            # Simple check - create embedding for test text
            query_service.document_processor.create_embeddings(["test"])
            components["document_processor"] = "healthy"
        except Exception as e:
            components["document_processor"] = f"unhealthy: {str(e)}"
        
        # Overall status
        overall_status = "healthy" if all(
            status == "healthy" for status in components.values()
        ) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            components={"error": str(e)}
        )


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get comprehensive cache usage statistics and performance metrics"
)
async def get_cache_stats():
    """
    Get comprehensive cache statistics
    
    Returns:
        Cache statistics including hit rate, size, entries, and provider info
    """
    try:
        stats = query_service.llm_service.get_cache_stats()
        return {
            "cache_enabled": settings.ENABLE_CACHE,
            "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post(
    "/cache/clear",
    summary="Clear cache",
    description="Clear all cached responses and reset cache statistics"
)
async def clear_cache():
    """
    Clear all cached responses and reset statistics
    
    Returns:
        Success message with cache stats before clearing
    """
    try:
        # Get stats before clearing
        stats_before = query_service.llm_service.get_cache_stats()
        
        # Clear the cache
        query_service.llm_service.clear_cache()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "stats_before_clearing": stats_before,
            "entries_cleared": stats_before.get("total_entries", 0)
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get(
    "/provider/status",
    summary="Get provider status",
    description="Get current provider configuration and availability"
)
async def get_provider_status():
    """
    Get current LLM provider status and configuration
    
    Returns:
        Provider status and availability information
    """
    try:
        stats = query_service.llm_service.get_cache_stats()
        
        return {
            "primary_provider": query_service.llm_service.primary_provider,
            "providers_available": {
                "gemini": {
                    "available": query_service.llm_service.gemini_client is not None,
                    "model": settings.GEMINI_MODEL if hasattr(settings, 'GEMINI_MODEL') else None
                }
            },
            "cache_stats": {
                "enabled": settings.ENABLE_CACHE,
                "hit_rate": stats.get("hit_rate_percentage", 0),
                "total_entries": stats.get("total_entries", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting provider status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get provider status: {str(e)}"
        )
