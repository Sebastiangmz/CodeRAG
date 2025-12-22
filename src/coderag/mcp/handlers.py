"""MCP handlers for CodeRAG - non-streaming versions of UI handlers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from coderag.config import get_settings
from coderag.generation.generator import ResponseGenerator
from coderag.indexing.embeddings import EmbeddingGenerator
from coderag.indexing.vectorstore import VectorStore
from coderag.ingestion.chunker import CodeChunker
from coderag.ingestion.filter import FileFilter
from coderag.ingestion.loader import RepositoryLoader
from coderag.ingestion.validator import GitHubURLValidator, ValidationError
from coderag.logging import get_logger
from coderag.models.chunk import Chunk
from coderag.models.document import Document
from coderag.models.query import Query
from coderag.models.repository import Repository, RepositoryStatus

logger = get_logger(__name__)


class MCPHandlers:
    """Handlers for MCP tools - non-streaming versions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = GitHubURLValidator()
        self.loader = RepositoryLoader()
        self.filter = FileFilter()
        self.chunker = CodeChunker()
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()
        self.generator: Optional[ResponseGenerator] = None

        # Repository metadata storage (shared with UIHandlers)
        self.repos_file = self.settings.data_dir / "repositories.json"
        self.repositories: dict[str, Repository] = self._load_repositories()

    def _load_repositories(self) -> dict[str, Repository]:
        """Load repositories from JSON file."""
        if self.repos_file.exists():
            try:
                data = json.loads(self.repos_file.read_text())
                return {r["id"]: Repository.from_dict(r) for r in data}
            except Exception as e:
                logger.error("Failed to load repositories", error=str(e))
        return {}

    def _save_repositories(self) -> None:
        """Save repositories to JSON file."""
        self.repos_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.repositories.values()]
        self.repos_file.write_text(json.dumps(data, indent=2))

    def _reload_repositories(self) -> None:
        """Reload repositories from disk (for consistency with UIHandlers)."""
        self.repositories = self._load_repositories()

    def _find_repository(self, repo_id: str) -> Optional[Repository]:
        """Find repository by full or partial ID."""
        self._reload_repositories()
        for rid, repo in self.repositories.items():
            if rid == repo_id or rid.startswith(repo_id):
                return repo
        return None

    def _process_batch(self, chunks: list[Chunk]) -> int:
        """Process a batch: embed + store + release memory."""
        if not chunks:
            return 0

        embedded = self.embedder.embed_chunks(chunks, show_progress=False)
        self.vectorstore.add_chunks(embedded)

        # Release memory
        del embedded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return len(chunks)

    def _get_current_commit(self, repo_path: Path) -> str:
        """Get the SHA of the current commit."""
        from git import Repo

        git_repo = Repo(repo_path)
        return git_repo.head.commit.hexsha

    def _get_changed_files(
        self,
        repo_path: Path,
        last_commit: str,
        current_commit: str,
    ) -> tuple[set[str], set[str], set[str]]:
        """Get files that were added, modified, or deleted between commits."""
        from git import Repo

        git_repo = Repo(repo_path)
        diff = git_repo.commit(last_commit).diff(current_commit)

        added: set[str] = set()
        modified: set[str] = set()
        deleted: set[str] = set()

        for d in diff:
            if d.new_file:
                added.add(d.b_path)
            elif d.deleted_file:
                deleted.add(d.a_path)
            elif d.renamed:
                deleted.add(d.a_path)
                added.add(d.b_path)
            else:
                modified.add(d.b_path or d.a_path)

        return added, modified, deleted

    async def index_repository(
        self,
        url: str,
        branch: str = "",
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Index a GitHub repository (non-streaming version)."""
        try:
            logger.info("MCP: Starting indexing", url=url, branch=branch)
            repo_info = self.validator.parse_url(url)
            branch = branch.strip() if branch else repo_info.branch or "main"

            # Create repository record
            repo = Repository(
                url=repo_info.url,
                branch=branch,
                status=RepositoryStatus.CLONING,
            )
            self.repositories[repo.id] = repo
            self._save_repositories()

            # Clone repository
            logger.info("MCP: Cloning repository", url=url, branch=branch)
            repo_path = self.loader.clone_repository(repo_info, branch)
            repo.clone_path = repo_path
            repo.status = RepositoryStatus.INDEXING
            self._save_repositories()

            # Setup filter with custom patterns
            file_filter = FileFilter(
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )

            # Process files
            logger.info("MCP: Filtering files", repo_path=str(repo_path))
            files = list(file_filter.filter_files(repo_path))
            file_count = len(files)
            logger.info("MCP: Files to process", count=file_count)

            # Delete existing chunks for this repo (re-indexing)
            self.vectorstore.delete_repo_chunks(repo.id)

            # Index all files
            total_chunks = 0
            batch: list[Chunk] = []
            batch_size = self.settings.ingestion.batch_size

            for file_path in files:
                try:
                    doc = Document.from_file(file_path, repo_path, repo.id)
                    for chunk in self.chunker.chunk_document(doc):
                        chunk.repo_id = repo.id
                        batch.append(chunk)

                        if len(batch) >= batch_size:
                            total_chunks += self._process_batch(batch)
                            batch = []
                except Exception as e:
                    logger.warning("Failed to process file", path=str(file_path), error=str(e))

            # Process final batch
            if batch:
                total_chunks += self._process_batch(batch)

            # Save commit for incremental updates
            try:
                repo.last_commit = self._get_current_commit(repo_path)
            except Exception:
                repo.last_commit = None

            # Update repository status
            repo.chunk_count = total_chunks
            repo.indexed_at = datetime.now()
            repo.status = RepositoryStatus.READY
            self._save_repositories()

            logger.info("MCP: Indexing complete", repo_id=repo.id, chunks=total_chunks)

            return {
                "success": True,
                "repo_id": repo.id,
                "name": repo.full_name,
                "branch": repo.branch,
                "files_processed": file_count,
                "chunks_indexed": total_chunks,
            }

        except ValidationError as e:
            logger.error("MCP: Validation error", error=str(e))
            return {"success": False, "error": f"Validation error: {str(e)}"}
        except Exception as e:
            logger.error("MCP: Indexing failed", error=str(e), exc_info=True)
            if "repo" in locals():
                repo.status = RepositoryStatus.ERROR
                repo.error_message = str(e)
                self._save_repositories()
            return {"success": False, "error": str(e)}

    async def query_code(
        self,
        repo_id: str,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Ask a question about a repository."""
        repo = self._find_repository(repo_id)

        if not repo:
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": f"Repository not found: {repo_id}",
            }

        if repo.status != RepositoryStatus.READY:
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": f"Repository not ready: status is {repo.status.value}",
            }

        try:
            # Lazy load generator
            if self.generator is None:
                self.generator = ResponseGenerator()

            query = Query(
                question=question.strip(),
                repo_id=repo.id,
                top_k=int(top_k),
            )

            response = self.generator.generate(query)

            return {
                "answer": response.answer,
                "citations": response.citations,
                "evidence": [
                    {
                        "file": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content": chunk.content[:500],  # Truncate for MCP response
                        "relevance": round(chunk.relevance_score or 0, 3),
                    }
                    for chunk in response.retrieved_chunks
                ],
                "grounded": response.grounded,
            }

        except Exception as e:
            logger.error("MCP: Query failed", error=str(e))
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": str(e),
            }

    async def list_repositories(self) -> dict[str, Any]:
        """List all indexed repositories."""
        self._reload_repositories()

        repos = []
        for repo in self.repositories.values():
            repos.append({
                "id": repo.id,
                "name": repo.full_name,
                "branch": repo.branch,
                "status": repo.status.value,
                "chunk_count": repo.chunk_count,
                "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            })

        return {
            "repositories": repos,
            "count": len(repos),
        }

    async def get_repository_info(self, repo_id: str) -> dict[str, Any]:
        """Get detailed repository information."""
        repo = self._find_repository(repo_id)

        if not repo:
            return {"error": f"Repository not found: {repo_id}"}

        # Get indexed files from vectorstore
        indexed_files: list[str] = []
        try:
            files = self.vectorstore.get_indexed_files(repo.id)
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
        repo = self._find_repository(repo_id)

        if not repo:
            return {"success": False, "error": f"Repository not found: {repo_id}"}

        try:
            # Get chunk count before deletion
            chunk_count = self.vectorstore.get_repo_chunk_count(repo.id)

            # Delete from vector store
            self.vectorstore.delete_repo_chunks(repo.id)

            # Delete cached repo
            try:
                self.loader.delete_cache(
                    type("RepoInfo", (), {"owner": repo.owner, "name": repo.name})()
                )
            except Exception:
                pass

            # Remove from records
            del self.repositories[repo.id]
            self._save_repositories()

            logger.info("MCP: Repository deleted", repo_id=repo.id)

            return {
                "success": True,
                "repo_id": repo.id,
                "name": repo.full_name,
                "chunks_deleted": chunk_count,
            }

        except Exception as e:
            logger.error("MCP: Delete failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def update_repository(self, repo_id: str) -> dict[str, Any]:
        """Incremental update of a repository."""
        repo = self._find_repository(repo_id)

        if not repo:
            return {"success": False, "error": f"Repository not found: {repo_id}"}

        if not repo.last_commit:
            return {
                "success": False,
                "error": "No previous indexing found. Please re-index the full repository.",
            }

        if not repo.clone_path or not Path(repo.clone_path).exists():
            return {"success": False, "error": "Repository cache not found. Please re-index."}

        try:
            repo_path = Path(repo.clone_path)

            # Update local repository
            logger.info("MCP: Updating local repository", repo_id=repo.id)
            self.loader._update_repository(repo_path, repo.branch, None)

            current_commit = self._get_current_commit(repo_path)

            if current_commit == repo.last_commit:
                return {
                    "success": True,
                    "message": "Repository is already up to date",
                    "files_changed": 0,
                    "chunks_added": 0,
                    "chunks_deleted": 0,
                }

            added, modified, deleted = self._get_changed_files(
                repo_path, repo.last_commit, current_commit
            )

            logger.info(
                "MCP: Changes detected",
                added=len(added),
                modified=len(modified),
                deleted=len(deleted),
            )

            # Delete chunks for deleted/modified files
            chunks_deleted = 0
            for file_path in deleted | modified:
                count = self.vectorstore.delete_file_chunks(repo.id, file_path)
                chunks_deleted += count if count else 0

            # Index new/modified files
            files_to_index = []
            file_filter = FileFilter()
            for file_path in added | modified:
                full_path = repo_path / file_path
                if full_path.exists() and file_filter.should_include(full_path, repo_path):
                    files_to_index.append(full_path)

            new_chunks = 0
            if files_to_index:
                batch_size = self.settings.ingestion.batch_size
                batch: list[Chunk] = []

                for file_path in files_to_index:
                    try:
                        doc = Document.from_file(file_path, repo_path, repo.id)
                        for chunk in self.chunker.chunk_document(doc):
                            chunk.repo_id = repo.id
                            batch.append(chunk)

                            if len(batch) >= batch_size:
                                new_chunks += self._process_batch(batch)
                                batch = []
                    except Exception as e:
                        logger.warning("Failed to process file", path=str(file_path), error=str(e))

                if batch:
                    new_chunks += self._process_batch(batch)

            # Update metadata
            repo.last_commit = current_commit
            repo.indexed_at = datetime.now()
            repo.chunk_count = self.vectorstore.get_repo_chunk_count(repo.id)
            self._save_repositories()

            return {
                "success": True,
                "files_changed": len(added | modified | deleted),
                "files_added": len(added),
                "files_modified": len(modified),
                "files_deleted": len(deleted),
                "chunks_added": new_chunks,
                "chunks_deleted": chunks_deleted,
                "total_chunks": repo.chunk_count,
            }

        except Exception as e:
            logger.error("MCP: Incremental update failed", error=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_code(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        file_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Semantic code search without LLM generation."""
        repo = self._find_repository(repo_id)

        if not repo:
            return {"results": [], "error": f"Repository not found: {repo_id}"}

        if repo.status != RepositoryStatus.READY:
            return {"results": [], "error": f"Repository not ready: status is {repo.status.value}"}

        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query, is_query=True)

            # Search vectorstore (query returns list of (Chunk, score) tuples)
            results = self.vectorstore.query(
                query_embedding=query_embedding,
                repo_id=repo.id,
                top_k=top_k,
            )

            # Filter by file pattern if provided
            if file_filter:
                import fnmatch

                results = [(chunk, score) for chunk, score in results if fnmatch.fnmatch(chunk.file_path, file_filter)]

            # Filter by chunk type if provided
            if chunk_type:
                results = [(chunk, score) for chunk, score in results if chunk.chunk_type == chunk_type]

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
                    for chunk, score in results[:top_k]
                ],
                "count": len(results),
            }

        except Exception as e:
            logger.error("MCP: Search failed", error=str(e))
            return {"results": [], "error": str(e)}


# Singleton pattern
_mcp_handlers: Optional[MCPHandlers] = None


def get_mcp_handlers() -> MCPHandlers:
    """Get the singleton MCPHandlers instance."""
    global _mcp_handlers
    if _mcp_handlers is None:
        _mcp_handlers = MCPHandlers()
    return _mcp_handlers
