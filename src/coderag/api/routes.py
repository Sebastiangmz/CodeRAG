"""REST API routes."""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from coderag.api.schemas import (
    IndexRepositoryRequest,
    IndexRepositoryResponse,
    QueryRequest,
    QueryResponse,
    ListRepositoriesResponse,
    RepositoryInfo,
    CitationResponse,
    RetrievedChunkResponse,
    ErrorResponse,
)
from coderag.config import get_settings
from coderag.generation.generator import ResponseGenerator
from coderag.indexing.embeddings import EmbeddingGenerator
from coderag.indexing.vectorstore import VectorStore
from coderag.ingestion.chunker import CodeChunker
from coderag.ingestion.filter import FileFilter
from coderag.ingestion.loader import RepositoryLoader
from coderag.ingestion.validator import GitHubURLValidator, ValidationError
from coderag.logging import get_logger
from coderag.models.document import Document
from coderag.models.query import Query as QueryModel
from coderag.models.repository import Repository, RepositoryStatus

logger = get_logger(__name__)
router = APIRouter()

# Global state (in production, use a proper database)
settings = get_settings()
repos_file = settings.data_dir / "repositories.json"
repositories: dict[str, Repository] = {}


def load_repositories() -> None:
    """Load repositories from disk."""
    global repositories
    if repos_file.exists():
        try:
            data = json.loads(repos_file.read_text())
            repositories = {r["id"]: Repository.from_dict(r) for r in data}
        except Exception as e:
            logger.error("Failed to load repositories", error=str(e))


def save_repositories() -> None:
    """Save repositories to disk."""
    repos_file.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in repositories.values()]
    repos_file.write_text(json.dumps(data, indent=2))


# Load on startup
load_repositories()


def resolve_repo_id(partial_id: str) -> Optional[str]:
    """Resolve a partial repository ID to a full ID.

    Supports both full UUIDs and partial IDs (first 8+ characters).
    Returns None if no match or multiple matches found.
    """
    # First try exact match
    if partial_id in repositories:
        return partial_id

    # Try prefix match (minimum 8 characters recommended)
    matches = [rid for rid in repositories.keys() if rid.startswith(partial_id)]

    if len(matches) == 1:
        return matches[0]

    return None


def get_repo_or_404(repo_id: str) -> Repository:
    """Get a repository by ID (full or partial), raising 404 if not found."""
    full_id = resolve_repo_id(repo_id)
    if full_id is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repositories[full_id]


async def index_repository_task(
    url: str,
    repo_id: str,
    branch: Optional[str],
    include_patterns: Optional[list[str]],
    exclude_patterns: Optional[list[str]],
) -> None:
    """Background task to index a repository."""
    repo = repositories[repo_id]

    try:
        # Validate and clone
        validator = GitHubURLValidator()
        repo_info = await validator.validate_repository(url)
        branch = branch or repo_info.branch or "main"

        loader = RepositoryLoader()
        repo_path = loader.clone_repository(repo_info, branch)

        repo.clone_path = repo_path
        repo.status = RepositoryStatus.INDEXING
        save_repositories()

        # Filter files
        file_filter = FileFilter(
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        files = list(file_filter.filter_files(repo_path))

        # Load documents
        documents = []
        for file_path in files:
            try:
                doc = Document.from_file(file_path, repo_path, repo.id)
                documents.append(doc)
            except Exception as e:
                logger.warning("Failed to load file", path=str(file_path), error=str(e))

        # Chunk
        chunker = CodeChunker()
        chunks = []
        for doc in documents:
            for chunk in chunker.chunk_document(doc):
                chunks.append(chunk)

        # Embed and store
        if chunks:
            vectorstore = VectorStore()
            vectorstore.delete_repo_chunks(repo.id)

            embedder = EmbeddingGenerator()
            embedded_chunks = embedder.embed_chunks(chunks)
            vectorstore.add_chunks(embedded_chunks)

        # Update status
        repo.chunk_count = len(chunks)
        repo.indexed_at = datetime.now()
        repo.status = RepositoryStatus.READY
        save_repositories()

        logger.info("Repository indexed", repo_id=repo_id, chunks=len(chunks))

    except Exception as e:
        logger.error("Indexing failed", repo_id=repo_id, error=str(e))
        repo.status = RepositoryStatus.ERROR
        repo.error_message = str(e)
        save_repositories()


@router.post("/repos/index", response_model=IndexRepositoryResponse, status_code=202)
async def index_repository(
    request: IndexRepositoryRequest,
    background_tasks: BackgroundTasks,
) -> IndexRepositoryResponse:
    """Index a GitHub repository."""
    # Create repository record
    repo = Repository(
        url=request.url,
        branch=request.branch or "main",
        status=RepositoryStatus.PENDING,
    )
    repositories[repo.id] = repo
    save_repositories()

    # Start background indexing
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
    """Query a repository.

    Supports both full repository IDs and partial IDs (first 8+ characters).
    """
    # Check repository exists (supports partial IDs)
    repo = get_repo_or_404(request.repo_id)

    if repo.status != RepositoryStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Repository not ready (status: {repo.status.value})",
        )

    try:
        # Generate response (use resolved repo.id for consistency)
        generator = ResponseGenerator()
        query = QueryModel(
            question=request.question,
            repo_id=repo.id,  # Use resolved full ID
            top_k=request.top_k,
        )
        response = generator.generate(query)

        # Convert to API schema
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
                )
                for c in response.retrieved_chunks
            ],
            grounded=response.grounded,
            query_id=response.query_id,
        )

    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
            for repo in repositories.values()
        ]
    )


@router.get("/repos/{repo_id}", response_model=RepositoryInfo)
async def get_repository(repo_id: str) -> RepositoryInfo:
    """Get repository details.

    Supports both full repository IDs and partial IDs (first 8+ characters).
    """
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
async def delete_repository(repo_id: str) -> dict:
    """Delete a repository.

    Supports both full repository IDs and partial IDs (first 8+ characters).
    """
    repo = get_repo_or_404(repo_id)

    try:
        # Delete from vector store (use resolved full ID)
        vectorstore = VectorStore()
        vectorstore.delete_repo_chunks(repo.id)

        # Delete from records (use resolved full ID)
        del repositories[repo.id]
        save_repositories()

        return {"message": f"Repository {repo.full_name} deleted"}

    except Exception as e:
        logger.error("Delete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
