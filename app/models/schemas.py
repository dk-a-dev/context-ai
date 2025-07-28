"""
Pydantic Models for API Request/Response
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for the main query processing endpoint"""
    
    documents: str = Field(
        ...,
        description="Document URL (blob URL, HTTP URL, etc.)",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of questions to answer based on the documents",
        example=[
            "What is the grace period for premium payment?",
            "Does this policy cover maternity expenses?"
        ]
    )


class QueryResponse(BaseModel):
    """Response model for the main query processing endpoint"""
    
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions"
    )
    # processing_time: Optional[float] = Field(
    #     None,
    #     description="Total processing time in seconds"
    # )
    # documents_processed: Optional[int] = Field(
    #     None,
    #     description="Number of documents successfully processed"
    # )
    # questions_processed: Optional[int] = Field(
    #     None,
    #     description="Number of questions processed"
    # )
    # vector_index_stats: Optional[Dict[str, Any]] = Field(
    #     None,
    #     description="Statistics about the vector index"
    # )
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )


class DocumentUploadRequest(BaseModel):
    """Request model for document upload and processing"""
    
    document_url: str = Field(
        ...,
        description="URL of the document to process"
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document identifier"
    )


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    
    success: bool = Field(..., description="Whether the upload was successful")
    document_id: str = Field(..., description="Unique identifier for the document")
    file_type: str = Field(..., description="Detected file type")
    num_chunks: int = Field(..., description="Number of text chunks created")
    total_length: int = Field(..., description="Total character length of the document")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if upload failed")


class SearchRequest(BaseModel):
    """Request model for vector search"""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )


class SearchResult(BaseModel):
    """Individual search result"""
    
    id: str = Field(..., description="Unique identifier for the result")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Matched text content")
    document_url: str = Field(..., description="Source document URL")
    chunk_id: str = Field(..., description="Chunk identifier within the document")
    file_type: str = Field(..., description="File type of the source document")
    chunk_length: int = Field(..., description="Length of the text chunk")


class SearchResponse(BaseModel):
    """Response model for vector search"""
    
    results: List[SearchResult] = Field(..., description="List of search results")
    query: str = Field(..., description="The original search query")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Search execution time in seconds")


class ExplanationRequest(BaseModel):
    """Request model for generating explanations"""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    relevant_clauses: List[str] = Field(..., description="Relevant document clauses")


class ExplanationResponse(BaseModel):
    """Response model for explanations"""
    
    explanation: str = Field(..., description="Detailed explanation of the answer")
    reasoning_steps: List[str] = Field(
        default=[],
        description="Step-by-step reasoning process"
    )
    cited_clauses: List[str] = Field(
        default=[],
        description="Specific clauses cited in the explanation"
    )


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    
    vector_search: Dict[str, Any] = Field(..., description="Vector search statistics")
    document_cache_size: int = Field(..., description="Number of cached documents")
    embedding_model: str = Field(..., description="Name of the embedding model")
    llm_model: str = Field(..., description="Name of the LLM model")
    config: Dict[str, Any] = Field(..., description="System configuration")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Status of individual components")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
