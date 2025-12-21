"""MCP tool definitions for CodeRAG."""

from typing import Optional

from coderag.mcp.handlers import get_mcp_handlers
from coderag.mcp.server import mcp


@mcp.tool()
async def index_repository(
    url: str,
    branch: str = "",
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> dict:
    """Index a GitHub repository for code Q&A.

    Args:
        url: GitHub repository URL (e.g., https://github.com/owner/repo)
        branch: Branch to index (defaults to main/master)
        include_patterns: File patterns to include (e.g., ["*.py", "*.ts"])
        exclude_patterns: File patterns to exclude (e.g., ["*_test.py"])

    Returns:
        dict with success status, repo_id, files_processed, and chunks_indexed
    """
    handlers = get_mcp_handlers()
    return await handlers.index_repository(
        url=url,
        branch=branch,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )


@mcp.tool()
async def query_code(
    repo_id: str,
    question: str,
    top_k: int = 5,
) -> dict:
    """Ask questions about indexed code with citations.

    Args:
        repo_id: Repository ID (full or first 8 characters)
        question: Question about the code
        top_k: Number of code chunks to retrieve for context (default: 5)

    Returns:
        dict with answer, citations, evidence, and grounded flag
    """
    handlers = get_mcp_handlers()
    return await handlers.query_code(
        repo_id=repo_id,
        question=question,
        top_k=top_k,
    )


@mcp.tool()
async def list_repositories() -> dict:
    """List all indexed repositories.

    Returns:
        dict with repositories array and count
    """
    handlers = get_mcp_handlers()
    return await handlers.list_repositories()


@mcp.tool()
async def get_repository_info(repo_id: str) -> dict:
    """Get detailed repository information.

    Args:
        repo_id: Repository ID (full or first 8 characters)

    Returns:
        dict with repository metadata including name, url, branch, chunk_count, status, and indexed_files
    """
    handlers = get_mcp_handlers()
    return await handlers.get_repository_info(repo_id=repo_id)


@mcp.tool()
async def delete_repository(repo_id: str) -> dict:
    """Remove an indexed repository.

    Args:
        repo_id: Repository ID (full or first 8 characters)

    Returns:
        dict with success status and chunks_deleted
    """
    handlers = get_mcp_handlers()
    return await handlers.delete_repository(repo_id=repo_id)


@mcp.tool()
async def update_repository(repo_id: str) -> dict:
    """Incremental update of a repository (only changed files).

    Args:
        repo_id: Repository ID (full or first 8 characters)

    Returns:
        dict with success status, files_changed, chunks_added, chunks_deleted
    """
    handlers = get_mcp_handlers()
    return await handlers.update_repository(repo_id=repo_id)


@mcp.tool()
async def search_code(
    repo_id: str,
    query: str,
    top_k: int = 10,
    file_filter: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> dict:
    """Semantic code search without LLM generation.

    Args:
        repo_id: Repository ID (full or first 8 characters)
        query: Search query
        top_k: Maximum number of results (default: 10)
        file_filter: File pattern filter (e.g., "*.py")
        chunk_type: Filter by chunk type (e.g., "function", "class")

    Returns:
        dict with results array containing file_path, start_line, end_line, content, and relevance_score
    """
    handlers = get_mcp_handlers()
    return await handlers.search_code(
        repo_id=repo_id,
        query=query,
        top_k=top_k,
        file_filter=file_filter,
        chunk_type=chunk_type,
    )
