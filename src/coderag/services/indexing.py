"""Repository indexing orchestration service."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from coderag.config import get_settings
from coderag.graph.extractor import CodeGraphExtractor
from coderag.logging import get_logger
from coderag.models.chunk import Chunk
from coderag.models.document import Document
from coderag.models.graph import FileGraph
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.registry import RepositoryRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexingOptions:
    """Options for a full repository indexing run."""

    branch: str | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


@dataclass(frozen=True)
class IndexingResult:
    """Result of a full repository indexing run."""

    repo: Repository
    files_processed: int
    chunks_indexed: int
    last_commit: str | None


@dataclass(frozen=True)
class IncrementalIndexingResult:
    """Result of an incremental repository update."""

    repo: Repository
    files_changed: int
    files_added: int
    files_modified: int
    files_deleted: int
    chunks_added: int
    chunks_deleted: int
    total_chunks: int
    already_up_to_date: bool = False


@dataclass(frozen=True)
class RepositoryDeleteResult:
    """Result of deleting a repository from metadata, cache, and vector index."""

    repo: Repository
    chunks_deleted: int


class IndexingService:
    """Shared indexing service for API, MCP, CLI, and UI adapters."""

    def __init__(
        self,
        registry: RepositoryRegistry | None = None,
        validator: Any | None = None,
        loader: Any | None = None,
        file_filter_factory: Callable[..., Any] | None = None,
        chunker: Any | None = None,
        embedder: Any | None = None,
        vectorstore: Any | None = None,
        graph_extractor: CodeGraphExtractor | None = None,
        commit_resolver: Callable[[Path], str | None] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.registry = registry or RepositoryRegistry()
        self.validator = validator
        self.loader = loader
        self.file_filter_factory = file_filter_factory
        self.chunker = chunker
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.commit_resolver = commit_resolver or self._get_current_commit
        self.graph_extractor = graph_extractor or CodeGraphExtractor()

    async def index_repository(self, url: str, options: IndexingOptions | None = None) -> IndexingResult:
        """Async-compatible full indexing entrypoint."""
        return self.index_repository_sync(url, options)

    def create_repository_record(
        self,
        url: str,
        options: IndexingOptions | None = None,
        status: RepositoryStatus = RepositoryStatus.PENDING,
    ) -> Repository:
        """Create and persist a repository record before background indexing."""
        options = options or IndexingOptions()
        repo_info = self._get_validator().parse_url(url)
        branch = (options.branch or repo_info.branch or "main").strip() or "main"
        repo = Repository(url=repo_info.url, branch=branch, status=status)
        self.registry.add(repo)
        return repo

    def index_repository_record(
        self,
        repo_id: str,
        url: str,
        options: IndexingOptions | None = None,
    ) -> IndexingResult:
        """Index an adapter-created repository record by id."""
        options = options or IndexingOptions()
        repo_info = self._get_validator().parse_url(url)
        branch = (options.branch or repo_info.branch or "main").strip() or "main"
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"Repository not found: {repo_id}")
        return self.index_existing_repository(repo, repo_info, branch, options)

    def index_repository_sync(self, url: str, options: IndexingOptions | None = None) -> IndexingResult:
        """Clone, chunk, embed, and persist a repository index."""
        options = options or IndexingOptions()
        repo_info = self._get_validator().parse_url(url)
        branch = (options.branch or repo_info.branch or "main").strip() or "main"
        repo = Repository(url=repo_info.url, branch=branch, status=RepositoryStatus.CLONING)
        self.registry.add(repo)
        return self.index_existing_repository(repo, repo_info, branch, options)

    def index_existing_repository(
        self,
        repo: Repository,
        repo_info: Any,
        branch: str,
        options: IndexingOptions | None = None,
    ) -> IndexingResult:
        """Index into an existing repository record created by an adapter."""
        options = options or IndexingOptions()
        job = self.registry.begin_job(repo.id, "index")
        try:
            logger.info("Indexing service: cloning repository", url=repo.url, branch=branch)
            repo.status = RepositoryStatus.CLONING
            self.registry.update(repo)
            repo_path = self._get_loader().clone_repository(repo_info, branch)
            repo.clone_path = repo_path
            repo.branch = branch
            repo.status = RepositoryStatus.INDEXING
            self.registry.update(repo)

            file_filter = self._get_file_filter_factory()(
                include_patterns=options.include_patterns,
                exclude_patterns=options.exclude_patterns,
            )
            files = list(file_filter.filter_files(repo_path))

            self._get_vectorstore().delete_repo_chunks(repo.id)
            self.registry.clear_file_metadata(repo.id)
            self.registry.clear_chunk_metadata(repo.id)
            self.registry.clear_graph_metadata(repo.id)
            for file_path in files:
                relative_path = str(file_path.relative_to(repo_path))
                self.registry.record_file_metadata(repo.id, relative_path, size_bytes=file_path.stat().st_size)
            chunks_indexed = self.index_files(files, repo_path, repo.id)
            last_commit = self._resolve_commit(repo_path)

            repo.chunk_count = chunks_indexed
            repo.indexed_at = datetime.now()
            repo.status = RepositoryStatus.READY
            repo.last_commit = last_commit
            self.registry.update(repo)
            self.registry.finish_job(job.id, "succeeded", files_processed=len(files), chunks_indexed=chunks_indexed)
            return IndexingResult(repo=repo, files_processed=len(files), chunks_indexed=chunks_indexed, last_commit=last_commit)
        except Exception as exc:
            repo.status = RepositoryStatus.ERROR
            repo.error_message = str(exc)
            self.registry.update(repo)
            self.registry.finish_job(job.id, "failed", error=str(exc))
            logger.error("Indexing service: indexing failed", repo_id=repo.id, error=str(exc))
            raise
        finally:
            unload = getattr(self.embedder, "unload", None)
            if callable(unload):
                unload()

    def index_files(self, files: Iterable[Path], repo_path: Path, repo_id: str) -> int:
        """Chunk/embed/store a collection of files for one repository."""
        total_chunks = 0
        batch: list[Chunk] = []
        batch_size = self.settings.ingestion.batch_size

        for file_path in files:
            try:
                document = Document.from_file(file_path, repo_path, repo_id)
                graph = self.graph_extractor.extract(document)
                self.registry.replace_graph_file(repo_id, document.file_path, graph)
                for chunk in self._get_chunker().chunk_document(document):
                    chunk.repo_id = repo_id
                    batch.append(chunk)
                    if len(batch) >= batch_size:
                        total_chunks += self._process_batch(batch)
                        batch = []
            except Exception as exc:
                logger.warning("Failed to process file", path=str(file_path), error=str(exc))

        if batch:
            total_chunks += self._process_batch(batch)

        return total_chunks

    def delete_repository(self, repo_id: str) -> RepositoryDeleteResult | None:
        """Delete repository chunks, cache, and metadata."""
        repo = self.registry.get(repo_id)
        if repo is None:
            return None

        job = self.registry.begin_job(repo.id, "delete")
        try:
            vectorstore = self._get_vectorstore()
            chunks_deleted = vectorstore.get_repo_chunk_count(repo.id)
            vectorstore.delete_repo_chunks(repo.id)

            try:
                self._get_loader().delete_cache(type("RepoInfo", (), {"owner": repo.owner, "name": repo.name})())
            except Exception:
                logger.debug("Repository cache delete skipped", repo_id=repo.id)

            removed = self.registry.remove(repo.id)
            if removed is None:
                self.registry.finish_job(job.id, "failed", error="Repository not found")
                return None
            self.registry.finish_job(job.id, "succeeded", chunks_indexed=chunks_deleted)
            return RepositoryDeleteResult(repo=removed, chunks_deleted=chunks_deleted)
        except Exception as exc:
            self.registry.finish_job(job.id, "failed", error=str(exc))
            raise

    def update_repository(self, repo_id: str) -> IncrementalIndexingResult:
        """Incrementally update an indexed repository from its cached clone."""
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"Repository not found: {repo_id}")
        job = self.registry.begin_job(repo.id, "update")
        try:
            if not repo.last_commit:
                raise ValueError("No previous indexing found. Please re-index the full repository.")
            if not repo.clone_path or not Path(repo.clone_path).exists():
                raise ValueError("Repository cache not found. Please re-index.")

            repo_path = Path(repo.clone_path)
            self._get_loader()._update_repository(repo_path, repo.branch, None)
            current_commit = self._resolve_commit(repo_path)
            if current_commit == repo.last_commit:
                self.registry.finish_job(job.id, "succeeded", files_processed=0, chunks_indexed=0)
                return IncrementalIndexingResult(
                    repo=repo,
                    files_changed=0,
                    files_added=0,
                    files_modified=0,
                    files_deleted=0,
                    chunks_added=0,
                    chunks_deleted=0,
                    total_chunks=repo.chunk_count,
                    already_up_to_date=True,
                )

            added, modified, deleted = self._get_changed_files(repo_path, repo.last_commit, current_commit or repo.last_commit)
            chunks_deleted = 0
            for file_path in deleted | modified:
                count = self._get_vectorstore().delete_file_chunks(repo.id, file_path)
                chunks_deleted += count or 0
                self.registry.replace_graph_file(repo.id, file_path, FileGraph(repo_id=repo.id, file_path=file_path, language=None))
                self.registry.remove_file_metadata(repo.id, file_path)

            file_filter = self._get_file_filter_factory()()
            files_to_index = []
            for file_path in added | modified:
                full_path = repo_path / file_path
                if full_path.exists() and file_filter.should_include(full_path, repo_path):
                    files_to_index.append(full_path)
            for indexed_path in files_to_index:
                relative_path = str(indexed_path.relative_to(repo_path))
                self.registry.record_file_metadata(repo.id, relative_path, size_bytes=indexed_path.stat().st_size)

            chunks_added = self.index_files(files_to_index, repo_path, repo.id) if files_to_index else 0
            repo.last_commit = current_commit
            repo.indexed_at = datetime.now()
            repo.chunk_count = self._get_vectorstore().get_repo_chunk_count(repo.id)
            self.registry.update(repo)
            self.registry.finish_job(
                job.id,
                "succeeded",
                files_processed=len(added | modified | deleted),
                chunks_indexed=chunks_added,
            )

            return IncrementalIndexingResult(
                repo=repo,
                files_changed=len(added | modified | deleted),
                files_added=len(added),
                files_modified=len(modified),
                files_deleted=len(deleted),
                chunks_added=chunks_added,
                chunks_deleted=chunks_deleted,
                total_chunks=repo.chunk_count,
            )
        except Exception as exc:
            self.registry.finish_job(job.id, "failed", error=str(exc))
            raise

    def get_indexed_files(self, repo_id: str) -> set[str]:
        return cast(set[str], self._get_vectorstore().get_indexed_files(repo_id))

    def get_chunk_count(self, repo_id: str) -> int:
        return int(self._get_vectorstore().get_repo_chunk_count(repo_id))

    def _process_batch(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        try:
            embedded = self._get_embedder().embed_chunks(chunks, show_progress=False)
            self._get_vectorstore().add_chunks(embedded)
            for chunk in embedded:
                self.registry.record_chunk_metadata(chunk)
            return len(chunks)
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _resolve_commit(self, repo_path: Path) -> str | None:
        try:
            return self.commit_resolver(repo_path)
        except Exception:
            return None

    def _get_current_commit(self, repo_path: Path) -> str:
        from git import Repo

        git_repo = Repo(repo_path)
        return cast(str, git_repo.head.commit.hexsha)

    def _get_changed_files(
        self,
        repo_path: Path,
        last_commit: str,
        current_commit: str,
    ) -> tuple[set[str], set[str], set[str]]:
        from git import Repo

        git_repo = Repo(repo_path)
        diff = git_repo.commit(last_commit).diff(current_commit)
        added: set[str] = set()
        modified: set[str] = set()
        deleted: set[str] = set()
        for item in diff:
            if item.new_file:
                added.add(item.b_path)
            elif item.deleted_file:
                deleted.add(item.a_path)
            elif item.renamed:
                deleted.add(item.a_path)
                added.add(item.b_path)
            else:
                modified.add(item.b_path or item.a_path)
        return added, modified, deleted

    def _get_validator(self) -> Any:
        if self.validator is None:
            from coderag.ingestion.validator import GitHubURLValidator

            self.validator = GitHubURLValidator()
        return self.validator

    def _get_loader(self) -> Any:
        if self.loader is None:
            from coderag.ingestion.loader import RepositoryLoader

            self.loader = RepositoryLoader()
        return self.loader

    def _get_file_filter_factory(self) -> Callable[..., Any]:
        if self.file_filter_factory is None:
            from coderag.ingestion.filter import FileFilter

            self.file_filter_factory = FileFilter
        return self.file_filter_factory

    def _get_chunker(self) -> Any:
        if self.chunker is None:
            from coderag.ingestion.chunker import CodeChunker

            self.chunker = CodeChunker()
        return self.chunker

    def _get_embedder(self) -> Any:
        if self.embedder is None:
            from coderag.indexing.embeddings import EmbeddingGenerator

            self.embedder = EmbeddingGenerator()
        return self.embedder

    def _get_vectorstore(self) -> Any:
        if self.vectorstore is None:
            from coderag.indexing.vectorstore import VectorStore

            self.vectorstore = VectorStore()
        return self.vectorstore
