"""Pydantic schemas for REST API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class IndexRepositoryRequest(BaseModel):
    """Request to index a repository."""

    url: str = Field(..., description="GitHub repository URL")
    branch: Optional[str] = Field(None, description="Branch name (default: main)")
    include_patterns: Optional[list[str]] = Field(None, description="File patterns to include")
    exclude_patterns: Optional[list[str]] = Field(None, description="File patterns to exclude")


class IndexRepositoryResponse(BaseModel):
    """Response from indexing request."""

    repo_id: str = Field(..., description="Repository ID")
    status: str = Field(..., description="Indexing status")
    message: str = Field(..., description="Status message")


class QueryRequest(BaseModel):
    """Request to query a repository."""

    question: str = Field(..., description="Question about the repository")
    repo_id: str = Field(..., description="Repository ID to query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class CitationResponse(BaseModel):
    """Citation information."""

    file_path: str
    start_line: int
    end_line: int

    class Config:
        from_attributes = True


class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk information."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    chunk_type: str
    name: Optional[str] = None
    content: str

    class Config:
        from_attributes = True


class QueryResponse(BaseModel):
    """Response from a query."""

    answer: str = Field(..., description="Generated answer")
    citations: list[CitationResponse] = Field(..., description="Citations in the answer")
    retrieved_chunks: list[RetrievedChunkResponse] = Field(..., description="Evidence chunks")
    grounded: bool = Field(..., description="Whether response is grounded in evidence")
    query_id: str = Field(..., description="Query ID")


class RepositoryInfo(BaseModel):
    """Repository information."""

    id: str
    url: str
    branch: str
    chunk_count: int
    status: str
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ListRepositoriesResponse(BaseModel):
    """List of repositories."""

    repositories: list[RepositoryInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app: str
    version: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
