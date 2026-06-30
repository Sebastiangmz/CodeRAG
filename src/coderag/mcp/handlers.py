"""MCP handlers for CodeRAG."""

from typing import Any, cast

from coderag.logging import get_logger
from coderag.models.repository import Repository
from coderag.services.graph import CodeGraphService
from coderag.services.indexing import IndexingOptions, IndexingService
from coderag.services.registry import RepositoryRegistry
from coderag.services.retrieval import RetrievalService

logger = get_logger(__name__)


def _read_field(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _citation_dict(item: Any) -> dict[str, Any]:
    return {
        "file_path": _read_field(item, "file_path"),
        "start_line": _read_field(item, "start_line"),
        "end_line": _read_field(item, "end_line"),
    }


def _citation_verification_dict(item: Any) -> dict[str, Any]:
    return {
        "file_path": _read_field(item, "file_path"),
        "start_line": _read_field(item, "start_line"),
        "end_line": _read_field(item, "end_line"),
        "verified": _read_field(item, "verified"),
        "reason": _read_field(item, "reason"),
        "chunk_id": _read_field(item, "chunk_id"),
    }

def _retrieved_chunk_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": _read_field(chunk, "chunk_id"),
        "file_path": _read_field(chunk, "file_path"),
        "start_line": _read_field(chunk, "start_line"),
        "end_line": _read_field(chunk, "end_line"),
        "citation": f"[{_read_field(chunk, 'file_path')}:{_read_field(chunk, 'start_line')}-{_read_field(chunk, 'end_line')}]",
        "chunk_type": _read_field(chunk, "chunk_type"),
        "name": _read_field(chunk, "name"),
        "content": _read_field(chunk, "content"),
        "relevance_score": round(_read_field(chunk, "relevance_score", 0.0) or 0.0, 3),
        "score_breakdown": _read_field(chunk, "score_breakdown", {}),
        "retrieval_sources": _read_field(chunk, "retrieval_sources", []),
        "token_estimate": _read_field(chunk, "token_estimate", 0),
        "ranking_reason": _read_field(chunk, "ranking_reason"),
    }

def _model_dict(item: Any) -> dict[str, Any]:
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        return cast("dict[str, Any]", to_dict())
    if isinstance(item, dict):
        return cast("dict[str, Any]", item)
    return dict(getattr(item, "__dict__", {}))


class MCPHandlers:
    """Handlers for MCP tools backed by shared application services."""

    def __init__(
        self,
        registry: RepositoryRegistry | None = None,
        indexing_service: IndexingService | None = None,
        retrieval_service: RetrievalService | None = None,
        graph_service: CodeGraphService | None = None,
    ) -> None:
        self.registry = registry or RepositoryRegistry()
        self.indexing_service = indexing_service or IndexingService(registry=self.registry)
        self.retrieval_service = retrieval_service or RetrievalService(registry=self.registry)
        self.graph_service = graph_service or CodeGraphService(registry=self.registry)

    @property
    def repositories(self) -> dict[str, Repository]:
        """Compatibility view for existing tests and adapters."""
        return {repo.id: repo for repo in self.registry.list()}

    @repositories.setter
    def repositories(self, repositories: dict[str, Repository]) -> None:
        self.registry.save(repositories)

    def _reload_repositories(self) -> None:
        """Compatibility no-op; registry reads latest data on demand."""
        self.registry.load()

    def _find_repository(self, repo_id: str) -> Repository | None:
        return self.registry.get(repo_id)

    async def index_repository(
        self,
        url: str,
        branch: str = "",
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index a GitHub repository."""
        try:
            result = await self.indexing_service.index_repository(
                url,
                IndexingOptions(
                    branch=branch or None,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                ),
            )
            return {
                "success": True,
                "repo_id": result.repo.id,
                "name": result.repo.full_name,
                "branch": result.repo.branch,
                "files_processed": result.files_processed,
                "chunks_indexed": result.chunks_indexed,
            }
        except Exception as exc:
            logger.error("MCP: Indexing failed", error=str(exc), exc_info=True)
            return {"success": False, "error": str(exc)}

    async def query_code(
        self,
        repo_id: str,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Ask a question about a repository."""
        result = self.retrieval_service.query_code(repo_id, question, top_k)
        if result.error is not None or result.response is None:
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "citation_verifications": [],
                "grounded": False,
                "error": result.error or "Query failed",
            }

        response = result.response
        return {
            "answer": response.answer,
            "citations": [_citation_dict(citation) for citation in response.citations],
            "evidence": [
                {
                    "file": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content[:500],
                    "relevance": round(chunk.relevance_score or 0, 3),
                }
                for chunk in response.retrieved_chunks
            ],
            "grounded": response.grounded,
            "citation_verifications": [
                _citation_verification_dict(verification)
                for verification in getattr(response, "citation_verifications", [])
            ],
        }

    async def list_repositories(self) -> dict[str, Any]:
        """List all indexed repositories."""
        repos = [
            {
                "id": repo.id,
                "name": repo.full_name,
                "branch": repo.branch,
                "status": repo.status.value,
                "chunk_count": repo.chunk_count,
                "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            }
            for repo in self.registry.list()
        ]
        return {"repositories": repos, "count": len(repos)}

    async def get_repository_info(self, repo_id: str) -> dict[str, Any]:
        """Get detailed repository information."""
        repo = self.registry.get(repo_id)
        if not repo:
            return {"error": f"Repository not found: {repo_id}"}

        indexed_files: list[str] = []
        try:
            files = self.indexing_service.get_indexed_files(repo.id)
            indexed_files = list(files) if files else []
        except Exception:
            pass

        return {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "url": repo.url,
            "branch": repo.branch,
            "status": repo.status.value,
            "chunk_count": repo.chunk_count,
            "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            "last_commit": repo.last_commit,
            "indexed_files": indexed_files,
            "error_message": repo.error_message,
        }

    async def delete_repository(self, repo_id: str) -> dict[str, Any]:
        """Delete an indexed repository."""
        result = self.indexing_service.delete_repository(repo_id)
        if result is None:
            return {"success": False, "error": f"Repository not found: {repo_id}"}
        return {
            "success": True,
            "repo_id": result.repo.id,
            "name": result.repo.full_name,
            "chunks_deleted": result.chunks_deleted,
        }

    async def update_repository(self, repo_id: str) -> dict[str, Any]:
        """Incremental update of a repository."""
        try:
            result = self.indexing_service.update_repository(repo_id)
            if result.already_up_to_date:
                return {
                    "success": True,
                    "message": "Repository is already up to date",
                    "files_changed": 0,
                    "chunks_added": 0,
                    "chunks_deleted": 0,
                }
            return {
                "success": True,
                "files_changed": result.files_changed,
                "files_added": result.files_added,
                "files_modified": result.files_modified,
                "files_deleted": result.files_deleted,
                "chunks_added": result.chunks_added,
                "chunks_deleted": result.chunks_deleted,
                "total_chunks": result.total_chunks,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def search_code(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        file_filter: str | None = None,
        chunk_type: str | None = None,
    ) -> dict[str, Any]:
        """Semantic code search without LLM generation."""
        try:
            results = self.retrieval_service.search_code(repo_id, query, top_k, file_filter, chunk_type)
            return {
                "results": [
                    {
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "chunk_type": chunk.chunk_type,
                        "content": chunk.content,
                        "relevance_score": round(score, 3),
                    }
                    for chunk, score in results
                ],
                "count": len(results),
            }
        except Exception as exc:
            logger.error("MCP: Search failed", error=str(exc))
            return {"results": [], "error": str(exc)}

    async def search_hybrid(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        max_chunks_per_file: int = 3,
    ) -> dict[str, Any]:
        """Hybrid code search without LLM generation."""
        try:
            results = self.retrieval_service.search_hybrid(repo_id, query, top_k, max_tokens, max_chunks_per_file)
            return {
                "results": [_retrieved_chunk_dict(chunk) for chunk in results],
                "count": len(results),
            }
        except Exception as exc:
            logger.error("MCP: Hybrid search failed", error=str(exc))
            return {"results": [], "error": str(exc)}

    async def get_context_pack(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        max_chunks_per_file: int = 3,
    ) -> dict[str, Any]:
        """Build a hybrid retrieval context pack without LLM generation."""
        try:
            pack = self.retrieval_service.get_context_pack(repo_id, query, top_k, max_tokens, max_chunks_per_file)
            return pack.to_dict()
        except Exception as exc:
            logger.error("MCP: Context pack failed", error=str(exc))
            return {"snippets": [], "error": str(exc)}

    def find_symbol(self, repo_id: str, symbol_name: str) -> dict[str, Any]:
        """Find symbols by name using the code graph."""
        try:
            symbols = self.graph_service.find_symbol(repo_id, symbol_name)
            return {"symbols": [_model_dict(symbol) for symbol in symbols], "count": len(symbols)}
        except Exception as exc:
            logger.error("MCP: Find symbol failed", error=str(exc))
            return {"symbols": [], "error": str(exc)}

    def find_references(self, repo_id: str, symbol_name: str) -> dict[str, Any]:
        """Find references by symbol name using the code graph."""
        try:
            references = self.graph_service.find_references(repo_id, symbol_name)
            return {"references": [_model_dict(reference) for reference in references], "count": len(references)}
        except Exception as exc:
            logger.error("MCP: Find references failed", error=str(exc))
            return {"references": [], "error": str(exc)}

    def get_blast_radius(self, repo_id: str, symbol_name: str) -> dict[str, Any]:
        """Return graph-backed blast radius for a symbol."""
        try:
            return self.graph_service.get_blast_radius(repo_id, symbol_name).to_dict()
        except Exception as exc:
            logger.error("MCP: Blast radius failed", error=str(exc))
            return {"impacted_files": [], "associated_tests": [], "error": str(exc)}


_mcp_handlers: MCPHandlers | None = None


def get_mcp_handlers() -> MCPHandlers:
    """Get the singleton MCPHandlers instance."""
    global _mcp_handlers
    if _mcp_handlers is None:
        _mcp_handlers = MCPHandlers()
    return _mcp_handlers
