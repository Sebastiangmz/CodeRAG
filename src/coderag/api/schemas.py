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
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    retrieval_sources: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    ranking_reason: str | None = None
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


class ContextPackRequest(BaseModel):
    """Request to build a retrieval-only context pack."""

    query: str = Field(..., description="Search query for the context pack")
    repo_id: str = Field(..., description="Repository ID to search")
    top_k: int = Field(10, ge=1, le=20, description="Maximum snippets to include")
    max_tokens: int = Field(4000, ge=1, description="Maximum context token estimate")
    max_chunks_per_file: int = Field(3, ge=1, description="Maximum snippets per file")


class ContextSnippetResponse(BaseModel):
    """A retrieval-only context snippet."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    citation: str
    chunk_type: str
    name: str | None = None
    content: str
    relevance_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    retrieval_sources: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    ranking_reason: str | None = None


class ContextPackResponse(BaseModel):
    """Retrieval-only context pack response."""

    repo_id: str
    query: str
    snippets: list[ContextSnippetResponse]
    token_estimate: int
    budget: dict[str, int]
    capabilities: dict[str, object]


class GraphQueryRequest(BaseModel):
    """Request for graph symbol queries."""

    repo_id: str = Field(..., description="Repository ID to query")
    symbol: str = Field(..., description="Symbol name")


class GraphSymbolResponse(BaseModel):
    repo_id: str
    file_path: str
    name: str
    kind: str
    language: str | None = None
    start_line: int
    end_line: int
    container: str | None = None


class GraphReferenceResponse(BaseModel):
    repo_id: str
    file_path: str
    symbol_name: str
    reference_kind: str
    start_line: int
    end_line: int
    source_name: str | None = None


class FindSymbolResponse(BaseModel):
    symbols: list[GraphSymbolResponse]


class FindReferencesResponse(BaseModel):
    references: list[GraphReferenceResponse]


class BlastRadiusResponse(BaseModel):
    repo_id: str
    symbol: str
    impacted_files: list[str]
    impacted_symbols: list[str]
    associated_tests: list[str]
    edges: list[dict[str, object]]
    reasons: list[str]
    capabilities: dict[str, object]

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
