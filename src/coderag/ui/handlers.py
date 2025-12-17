"""UI event handlers for Gradio interface."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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

    def index_repository(
        self,
        url: str,
        branch: str = "",
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> str:
        """Index a GitHub repository."""
        try:
            # Validate URL (sync version, skip accessibility check for UI)
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
            logger.info("Cloning repository", url=url, branch=branch)
            repo_path = self.loader.clone_repository(repo_info, branch)
            repo.clone_path = repo_path
            repo.status = RepositoryStatus.INDEXING

            # Setup filter with custom patterns
            include = [p.strip() for p in include_patterns.split(",") if p.strip()] or None
            exclude = [p.strip() for p in exclude_patterns.split(",") if p.strip()] or None
            file_filter = FileFilter(include_patterns=include, exclude_patterns=exclude)

            # Process files
            logger.info("Filtering files", repo_path=str(repo_path))
            files = list(file_filter.filter_files(repo_path))
            logger.info("Files to process", count=len(files))

            documents = []
            for file_path in files:
                try:
                    doc = Document.from_file(file_path, repo_path, repo.id)
                    documents.append(doc)
                except Exception as e:
                    logger.warning("Failed to load file", path=str(file_path), error=str(e))

            logger.info("Documents loaded", count=len(documents))

            # Chunk documents
            logger.info("Starting chunking")
            chunks = []
            for doc in documents:
                for chunk in self.chunker.chunk_document(doc):
                    chunks.append(chunk)

            logger.info("Chunking complete", chunk_count=len(chunks))

            # Generate embeddings and store
            if chunks:
                # Delete existing chunks for this repo (re-indexing)
                logger.info("Deleting previous chunks for repo", repo_id=repo.id)
                self.vectorstore.delete_repo_chunks(repo.id)

                # Embed in batches
                logger.info("Starting embedding generation", chunk_count=len(chunks))
                embedded_chunks = self.embedder.embed_chunks(chunks)
                logger.info("Embeddings complete", embedded_count=len(embedded_chunks))

                logger.info("Storing chunks in vector store")
                self.vectorstore.add_chunks(embedded_chunks)
                logger.info("Chunks stored successfully")

            # Update repository status
            repo.chunk_count = len(chunks)
            repo.indexed_at = datetime.now()
            repo.status = RepositoryStatus.READY
            self._save_repositories()

            result = f"Successfully indexed {repo_info.full_name}\n{len(files)} files processed\n{len(chunks)} chunks indexed"
            logger.info("Indexing complete", result=result)
            return result

        except ValidationError as e:
            logger.error("Validation error", error=str(e))
            return f"Validation error: {str(e)}"
        except Exception as e:
            logger.error("Indexing failed", error=str(e), exc_info=True)
            if "repo" in locals():
                repo.status = RepositoryStatus.ERROR
                repo.error_message = str(e)
                self._save_repositories()
            return f"Error: {str(e)}"

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
