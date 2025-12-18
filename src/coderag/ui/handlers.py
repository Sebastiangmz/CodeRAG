"""UI event handlers for Gradio interface."""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

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


class UIHandlers:
    """Handlers for Gradio UI events."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = GitHubURLValidator()
        self.loader = RepositoryLoader()
        self.filter = FileFilter()
        self.chunker = CodeChunker()
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()
        self.generator: Optional[ResponseGenerator] = None

        # Repository metadata storage
        self.repos_file = self.settings.data_dir / "repositories.json"
        self.repositories: dict[str, Repository] = self._load_repositories()

    def _load_repositories(self) -> dict[str, Repository]:
        if self.repos_file.exists():
            try:
                data = json.loads(self.repos_file.read_text())
                return {r["id"]: Repository.from_dict(r) for r in data}
            except Exception as e:
                logger.error("Failed to load repositories", error=str(e))
        return {}

    def _save_repositories(self) -> None:
        self.repos_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.repositories.values()]
        self.repos_file.write_text(json.dumps(data, indent=2))

    # =========================================================================
    # Streaming Methods (Nivel 1)
    # =========================================================================

    def _document_generator(
        self,
        files: list[Path],
        repo_path: Path,
        repo_id: str,
    ) -> Iterator[Document]:
        """Generate documents one by one without accumulating in memory."""
        for file_path in files:
            try:
                yield Document.from_file(file_path, repo_path, repo_id)
            except Exception as e:
                logger.warning("Failed to load file", path=str(file_path), error=str(e))

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

    def _stream_index_repository(
        self,
        documents: Iterator[Document],
        repo_id: str,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """Index using streaming with batches."""
        total_chunks = 0
        batch: list[Chunk] = []
        doc_count = 0

        for doc in documents:
            doc_count += 1
            for chunk in self.chunker.chunk_document(doc):
                chunk.repo_id = repo_id
                batch.append(chunk)

                if len(batch) >= batch_size:
                    total_chunks += self._process_batch(batch)
                    logger.info("Batch processed", total_so_far=total_chunks, docs_processed=doc_count)
                    if progress_callback:
                        progress_callback(total_chunks, doc_count)
                    batch = []

        # Process final batch
        if batch:
            total_chunks += self._process_batch(batch)
            if progress_callback:
                progress_callback(total_chunks, doc_count)

        return total_chunks

    # =========================================================================
    # Validation Methods (Nivel 2)
    # =========================================================================

    def _estimate_repo_size(self, files: list[Path]) -> dict:
        """Estimate repository size before indexing."""
        total_size_kb = 0
        estimated_chunks = 0
        chunk_size = self.settings.ingestion.chunk_size

        for file_path in files:
            try:
                size_kb = file_path.stat().st_size / 1024
                total_size_kb += size_kb
                # Rough estimate: 1 chunk per chunk_size characters
                estimated_chunks += max(1, int(size_kb * 1024 / chunk_size))
            except Exception:
                continue

        return {
            "file_count": len(files),
            "total_size_kb": total_size_kb,
            "estimated_chunks": estimated_chunks,
            "exceeds_file_limit": len(files) > self.settings.ingestion.max_files_per_repo,
            "exceeds_chunk_limit": estimated_chunks > self.settings.ingestion.max_total_chunks,
            "warn_large_repo": len(files) > self.settings.ingestion.warn_files_threshold,
        }

    def _validate_repo_size(self, estimate: dict) -> tuple[bool, str]:
        """Validate if the repository can be indexed."""
        if estimate["exceeds_file_limit"]:
            return False, f"Repository exceeds file limit ({estimate['file_count']} > {self.settings.ingestion.max_files_per_repo})"
        if estimate["exceeds_chunk_limit"]:
            return False, f"Repository exceeds chunk limit (~{estimate['estimated_chunks']} > {self.settings.ingestion.max_total_chunks})"

        warning = ""
        if estimate["warn_large_repo"]:
            warning = f"Large repository ({estimate['file_count']} files, ~{estimate['estimated_chunks']} chunks). Processing may take several minutes."

        return True, warning

    # =========================================================================
    # Incremental Indexing Methods (Nivel 3)
    # =========================================================================

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

    def index_repository_incremental(self, repo_id: str) -> str:
        """Update only modified files since last indexing (incremental update)."""
        # Find repository by full or partial ID
        found_repo = None
        for rid, repo in self.repositories.items():
            if rid == repo_id or rid.startswith(repo_id):
                found_repo = repo
                break

        if not found_repo:
            return "Repository not found"

        repo = found_repo

        if not repo.last_commit:
            return "No previous indexing found. Please re-index the full repository."

        if not repo.clone_path or not Path(repo.clone_path).exists():
            return "Repository cache not found. Please re-index."

        try:
            repo_path = Path(repo.clone_path)

            # Update local repository
            logger.info("Updating local repository", repo_id=repo.id)
            self.loader._update_repository(repo_path, repo.branch, None)

            current_commit = self._get_current_commit(repo_path)

            if current_commit == repo.last_commit:
                return "Repository is already up to date."

            added, modified, deleted = self._get_changed_files(
                repo_path, repo.last_commit, current_commit
            )

            logger.info(
                "Changes detected",
                added=len(added),
                modified=len(modified),
                deleted=len(deleted),
            )

            # Delete chunks for deleted/modified files
            for file_path in deleted | modified:
                self.vectorstore.delete_file_chunks(repo.id, file_path)

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
                doc_generator = self._document_generator(files_to_index, repo_path, repo.id)
                new_chunks = self._stream_index_repository(doc_generator, repo.id, batch_size)

            # Update metadata
            repo.last_commit = current_commit
            repo.indexed_at = datetime.now()
            repo.chunk_count = self.vectorstore.get_repo_chunk_count(repo.id)
            self._save_repositories()

            return (
                f"Incremental update complete:\n"
                f"- Added/Modified: {len(added | modified)} files\n"
                f"- Deleted: {len(deleted)} files\n"
                f"- New chunks: {new_chunks}\n"
                f"- Total chunks: {repo.chunk_count}"
            )

        except Exception as e:
            logger.error("Incremental indexing failed", error=str(e), exc_info=True)
            return f"Error: {str(e)}"

    def index_repository(
        self,
        url: str,
        branch: str = "",
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> Iterator[str]:
        """Index a GitHub repository with progress updates."""
        try:
            # Validate URL (sync version, skip accessibility check for UI)
            yield "Validating repository URL..."
            logger.info("Starting indexing", url=url, branch=branch)
            repo_info = self.validator.parse_url(url)
            branch = branch.strip() or repo_info.branch or "main"

            # Create repository record
            repo = Repository(
                url=repo_info.url,
                branch=branch,
                status=RepositoryStatus.CLONING,
            )
            self.repositories[repo.id] = repo

            # Clone repository
            yield f"Cloning {repo_info.full_name} (branch: {branch})..."
            logger.info("Cloning repository", url=url, branch=branch)
            repo_path = self.loader.clone_repository(repo_info, branch)
            repo.clone_path = repo_path
            repo.status = RepositoryStatus.INDEXING

            # Setup filter with custom patterns
            include = [p.strip() for p in include_patterns.split(",") if p.strip()] or None
            exclude = [p.strip() for p in exclude_patterns.split(",") if p.strip()] or None
            file_filter = FileFilter(include_patterns=include, exclude_patterns=exclude)

            # Process files
            yield "Scanning files..."
            logger.info("Filtering files", repo_path=str(repo_path))
            files = list(file_filter.filter_files(repo_path))
            file_count = len(files)
            logger.info("Files to process", count=file_count)

            # Validate repository size (Nivel 2)
            estimate = self._estimate_repo_size(files)
            can_proceed, message = self._validate_repo_size(estimate)

            if not can_proceed:
                repo.status = RepositoryStatus.ERROR
                repo.error_message = message
                self._save_repositories()
                yield f"Error: {message}"
                return

            if message:
                logger.warning(message)
                yield f"Warning: {message}"

            yield f"Found {file_count} files to index (~{estimate['estimated_chunks']} chunks)"

            # Delete existing chunks for this repo (re-indexing)
            logger.info("Deleting previous chunks for repo", repo_id=repo.id)
            self.vectorstore.delete_repo_chunks(repo.id)

            # Stream indexing with batches and progress updates
            yield f"Indexing... (0/{file_count} files, 0 chunks)"
            logger.info("Starting streaming indexing", file_count=file_count)
            batch_size = self.settings.ingestion.batch_size
            doc_generator = self._document_generator(files, repo_path, repo.id)

            # Process with progress updates
            total_chunks = 0
            batch: list[Chunk] = []
            doc_count = 0

            for doc in doc_generator:
                doc_count += 1
                for chunk in self.chunker.chunk_document(doc):
                    chunk.repo_id = repo.id
                    batch.append(chunk)

                    if len(batch) >= batch_size:
                        total_chunks += self._process_batch(batch)
                        batch = []
                        # Yield progress update
                        yield f"Indexing... ({doc_count}/{file_count} files, {total_chunks} chunks)"

            # Process final batch
            if batch:
                total_chunks += self._process_batch(batch)

            logger.info("Streaming indexing complete", chunk_count=total_chunks)

            # Save current commit for incremental updates (Nivel 3)
            try:
                repo.last_commit = self._get_current_commit(repo_path)
            except Exception:
                repo.last_commit = None

            # Update repository status
            repo.chunk_count = total_chunks
            repo.indexed_at = datetime.now()
            repo.status = RepositoryStatus.READY
            self._save_repositories()

            result = f"Successfully indexed {repo_info.full_name}\n{file_count} files processed\n{total_chunks} chunks indexed"
            logger.info("Indexing complete", result=result)
            yield result

        except ValidationError as e:
            logger.error("Validation error", error=str(e))
            yield f"Validation error: {str(e)}"
        except Exception as e:
            logger.error("Indexing failed", error=str(e), exc_info=True)
            if "repo" in locals():
                repo.status = RepositoryStatus.ERROR
                repo.error_message = str(e)
                self._save_repositories()
            yield f"Error: {str(e)}"

    def ask_question(
        self,
        repo_id: str,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, str, str]:
        """Ask a question about a repository."""
        if not repo_id:
            return "", "", "Please select a repository"

        if not question.strip():
            return "", "", "Please enter a question"

        try:
            # Lazy load generator
            if self.generator is None:
                self.generator = ResponseGenerator()

            query = Query(
                question=question.strip(),
                repo_id=repo_id,
                top_k=int(top_k),
            )

            response = self.generator.generate(query)

            # Format answer
            answer_md = f"## Answer\n\n{response.answer}"
            if response.citations:
                answer_md += "\n\n### Citations\n"
                for citation in response.citations:
                    answer_md += f"- `{citation}`\n"

            # Format evidence
            evidence_md = response.format_evidence()

            status = "Grounded" if response.grounded else "Not grounded (no citations)"

            return answer_md, evidence_md, status

        except Exception as e:
            logger.error("Question failed", error=str(e))
            return "", "", f"Error: {str(e)}"

    def get_repositories(self):
        """Get list of repositories for dropdown."""
        import gradio as gr
        choices = []
        for repo in self.repositories.values():
            if repo.status == RepositoryStatus.READY:
                label = f"{repo.full_name} ({repo.chunk_count} chunks)"
                choices.append((label, repo.id))
        return gr.update(choices=choices)

    def get_repositories_table(self) -> list[list]:
        """Get repositories as table data."""
        rows = []
        for repo in self.repositories.values():
            rows.append([
                repo.id[:8],
                repo.full_name,
                repo.branch,
                repo.chunk_count,
                repo.status.value,
                repo.indexed_at.strftime("%Y-%m-%d %H:%M") if repo.indexed_at else "-",
            ])
        return rows

    def delete_repository(self, repo_id: str) -> tuple[str, list[list]]:
        """Delete a repository."""
        repo_id = repo_id.strip()

        # Find by full or partial ID
        found_repo = None
        for rid, repo in self.repositories.items():
            if rid == repo_id or rid.startswith(repo_id):
                found_repo = repo
                break

        if not found_repo:
            return "Repository not found", self.get_repositories_table()

        try:
            # Delete from vector store
            self.vectorstore.delete_repo_chunks(found_repo.id)

            # Delete cached repo
            self.loader.delete_cache(
                type("RepoInfo", (), {"owner": found_repo.owner, "name": found_repo.name})()
            )

            # Remove from records
            del self.repositories[found_repo.id]
            self._save_repositories()

            return f"Deleted {found_repo.full_name}", self.get_repositories_table()

        except Exception as e:
            logger.error("Delete failed", error=str(e))
            return f"Error: {str(e)}", self.get_repositories_table()
