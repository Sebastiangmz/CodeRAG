"""Pydantic schemas for REST API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IndexRepositoryRequest(BaseModel):
    """Request to index a repository."""

    url: str = Field(..., description="GitHub repository URL")
    branch: str | None = Field(None, description="Branch name (default: main)")
    include_patterns: list[str] | None = Field(None, description="File patterns to include")
    exclude_patterns: list[str] | None = Field(None, description="File patterns to exclude")


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

    model_config = ConfigDict(from_attributes=True)

class CitationVerificationResponse(BaseModel):
    """Verification status for a parsed citation."""

    file_path: str
    start_line: int
    end_line: int
    verified: bool
    reason: str
    chunk_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk information."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    chunk_type: str
    name: str | None = None
    content: str
    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    """Response from a query."""

    answer: str = Field(..., description="Generated answer")
    citations: list[CitationResponse] = Field(..., description="Citations in the answer")
    citation_verifications: list[CitationVerificationResponse] = Field(
        default_factory=list,
        description="Verification status for each parsed citation",
    )
    retrieved_chunks: list[RetrievedChunkResponse] = Field(..., description="Evidence chunks")
    grounded: bool = Field(..., description="Whether every parsed citation is verified against retrieved evidence")
    query_id: str = Field(..., description="Query ID")

class RepositoryInfo(BaseModel):
    """Repository information."""

    id: str
    url: str
    branch: str
    chunk_count: int
    status: str
    indexed_at: datetime | None = None
    error_message: str | None = None


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
    detail: str | None = None
