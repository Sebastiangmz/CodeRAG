"""REST API routes."""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from coderag.api.schemas import (
    CitationResponse,
    CitationVerificationResponse,
    ContextPackRequest,
    ContextPackResponse,
    ContextSnippetResponse,
    IndexRepositoryRequest,
    IndexRepositoryResponse,
    ListRepositoriesResponse,
    QueryRequest,
    QueryResponse,
    RepositoryInfo,
    RetrievedChunkResponse,
)
from coderag.logging import get_logger
from coderag.models.repository import Repository
from coderag.services.indexing import IndexingOptions, IndexingService
from coderag.services.registry import RepositoryRegistry
from coderag.services.retrieval import RetrievalService

logger = get_logger(__name__)
router = APIRouter()

registry = RepositoryRegistry()
indexing_service = IndexingService(registry=registry)
retrieval_service = RetrievalService(registry=registry)


def get_repo_or_404(repo_id: str) -> Repository:
    """Get a repository by full ID or unambiguous prefix, raising 404 if not found."""
    repo = registry.get_unique(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


def _read_field(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _citation_verification_response(item: Any) -> CitationVerificationResponse:
    return CitationVerificationResponse(
        file_path=_read_field(item, "file_path"),
        start_line=_read_field(item, "start_line"),
        end_line=_read_field(item, "end_line"),
        verified=_read_field(item, "verified"),
        reason=_read_field(item, "reason"),
        chunk_id=_read_field(item, "chunk_id"),
    )


async def index_repository_task(
    url: str,
    repo_id: str,
    branch: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    """Background task to index a repository through the shared service seam."""
    try:
        indexing_service.index_repository_record(
            repo_id,
            url,
            IndexingOptions(
                branch=branch,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            ),
        )
    except Exception as exc:
        logger.error("Indexing failed", repo_id=repo_id, error=str(exc))


@router.post("/repos/index", response_model=IndexRepositoryResponse, status_code=202)
async def index_repository(
    request: IndexRepositoryRequest,
    background_tasks: BackgroundTasks,
) -> IndexRepositoryResponse:
    """Index a GitHub repository."""
    try:
        repo = indexing_service.create_repository_record(
            request.url,
            IndexingOptions(
                branch=request.branch,
                include_patterns=request.include_patterns,
                exclude_patterns=request.exclude_patterns,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(
        index_repository_task,
        request.url,
        repo.id,
        request.branch,
        request.include_patterns,
        request.exclude_patterns,
    )

    return IndexRepositoryResponse(
        repo_id=repo.id,
        status=repo.status.value,
        message="Repository indexing started",
    )


@router.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest) -> QueryResponse:
    """Query a repository by full ID or prefix."""
    repo = get_repo_or_404(request.repo_id)
    result = retrieval_service.query_code(repo.id, request.question, request.top_k)
    if result.error is not None or result.response is None:
        if result.error and result.error.startswith("Repository not found"):
            raise HTTPException(status_code=404, detail="Repository not found")
        if result.error and result.error.startswith("Repository not ready"):
            raise HTTPException(
                status_code=400,
                detail=f"Repository not ready (status: {result.repo.status.value})" if result.repo else result.error,
            )
        if result.error and (
            result.error.startswith("LLM generation is disabled") or result.error.startswith("Private profile requires")
        ):
            raise HTTPException(status_code=400, detail=result.error)
        raise HTTPException(status_code=500, detail=result.error or "Query failed")

    response = result.response
    return QueryResponse(
        answer=response.answer,
        citations=[
            CitationResponse(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
            )
            for c in response.citations
        ],
        citation_verifications=[
            _citation_verification_response(verification)
            for verification in getattr(response, "citation_verifications", [])
        ],
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                relevance_score=c.relevance_score,
                chunk_type=c.chunk_type,
                name=c.name,
                content=c.content,
                score_breakdown=c.score_breakdown,
                retrieval_sources=c.retrieval_sources,
                token_estimate=c.token_estimate,
                ranking_reason=c.ranking_reason,
            )
            for c in response.retrieved_chunks
        ],
        grounded=response.grounded,
        query_id=response.query_id,
    )


@router.post("/context-pack", response_model=ContextPackResponse)
async def context_pack(request: ContextPackRequest) -> ContextPackResponse:
    """Build a hybrid retrieval context pack without LLM generation."""
    repo = get_repo_or_404(request.repo_id)
    try:
        pack = retrieval_service.get_context_pack(
            repo.id,
            request.query,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            max_chunks_per_file=request.max_chunks_per_file,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ContextPackResponse(
        repo_id=pack.repo_id,
        query=pack.query,
        snippets=[
            ContextSnippetResponse(
                chunk_id=snippet.chunk_id,
                file_path=snippet.file_path,
                start_line=snippet.start_line,
                end_line=snippet.end_line,
                citation=snippet.citation,
                chunk_type=snippet.chunk_type,
                name=snippet.name,
                content=snippet.content,
                relevance_score=snippet.relevance_score,
                score_breakdown=snippet.score_breakdown,
                retrieval_sources=snippet.retrieval_sources,
                token_estimate=snippet.token_estimate,
                ranking_reason=snippet.ranking_reason,
            )
            for snippet in pack.snippets
        ],
        token_estimate=pack.token_estimate,
        budget=pack.budget,
        capabilities=pack.capabilities,
    )


@router.get("/repos", response_model=ListRepositoriesResponse)
async def list_repositories() -> ListRepositoriesResponse:
    """List all repositories."""
    return ListRepositoriesResponse(
        repositories=[
            RepositoryInfo(
                id=repo.id,
                url=repo.url,
                branch=repo.branch,
                chunk_count=repo.chunk_count,
                status=repo.status.value,
                indexed_at=repo.indexed_at,
                error_message=repo.error_message,
            )
            for repo in registry.list()
        ]
    )


@router.get("/repos/{repo_id}", response_model=RepositoryInfo)
async def get_repository(repo_id: str) -> RepositoryInfo:
    """Get repository details by full ID or prefix."""
    repo = get_repo_or_404(repo_id)
    return RepositoryInfo(
        id=repo.id,
        url=repo.url,
        branch=repo.branch,
        chunk_count=repo.chunk_count,
        status=repo.status.value,
        indexed_at=repo.indexed_at,
        error_message=repo.error_message,
    )


@router.delete("/repos/{repo_id}")
async def delete_repository(repo_id: str) -> dict[str, str]:
    """Delete a repository by full ID or prefix."""
    repo = get_repo_or_404(repo_id)
    try:
        result = indexing_service.delete_repository(repo.id)
    except Exception as exc:
        logger.error("Delete failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return {"message": f"Repository {result.repo.full_name} deleted"}
