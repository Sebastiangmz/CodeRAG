"""Shared retrieval and query orchestration service."""

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.chunk import Chunk
from coderag.models.context import ContextPack, ContextSnippet
from coderag.models.query import Query
from coderag.models.repository import Repository, RepositoryStatus
from coderag.models.response import Response, RetrievedChunk
from coderag.retrieval.hybrid import HybridRetriever, RetrievalBudget
from coderag.services.providers import ProviderConfigService
from coderag.services.registry import RepositoryRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryServiceResult:
    """Service-level query result with typed adapter-friendly errors."""

    response: Response | None = None
    repo: Repository | None = None
    error: str | None = None


class RetrievalService:
    """Shared retrieval/query service for API, MCP, CLI, and UI adapters."""

    def __init__(
        self,
        registry: RepositoryRegistry | None = None,
        generator_factory: Callable[[], Any] | None = None,
        embedder: Any | None = None,
        vectorstore: Any | None = None,
        provider_config: ProviderConfigService | None = None,
    ) -> None:
        self.registry = registry or RepositoryRegistry()
        self.generator_factory = generator_factory or self._default_generator_factory
        self.embedder = embedder
        self.vectorstore = vectorstore
        self._generator: Any | None = None
        self.provider_config = provider_config or ProviderConfigService()
        self.retrieval_settings = get_settings().retrieval

    def require_ready_repository(self, repo_id: str) -> tuple[Repository | None, str | None]:
        """Return a ready repository or a public adapter-safe error message."""
        repo = self.registry.get(repo_id)
        if repo is None:
            return None, f"Repository not found: {repo_id}"
        if repo.status != RepositoryStatus.READY:
            return repo, f"Repository not ready: status is {repo.status.value}"
        return repo, None

    def query_code(self, repo_id: str, question: str, top_k: int = 5) -> QueryServiceResult:
        """Generate an answer for a ready repository."""
        repo, error = self.require_ready_repository(repo_id)
        if error is not None:
            return QueryServiceResult(repo=repo, error=error)
        assert repo is not None
        block_reason = self.provider_config.generation_block_reason()
        if block_reason is not None:
            return QueryServiceResult(repo=repo, error=block_reason)
        top_k = self._clamp_top_k(top_k)

        try:
            query = Query(question=question.strip(), repo_id=repo.id, top_k=int(top_k))
            response = self._get_generator().generate(query)
            return QueryServiceResult(response=response, repo=repo)
        except Exception as exc:
            logger.error("Retrieval service: query failed", repo_id=repo.id if repo else repo_id, error=str(exc))
            return QueryServiceResult(repo=repo, error=str(exc))

    def search_code(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        file_filter: str | None = None,
        chunk_type: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic code search without LLM generation."""
        repo, error = self.require_ready_repository(repo_id)
        if error is not None:
            raise ValueError(error)
        assert repo is not None

        top_k = self._clamp_top_k(top_k)
        embedder = self._get_embedder()
        vectorstore = self._get_vectorstore()
        query_embedding = embedder.generate_embedding(query, is_query=True)
        results = vectorstore.query(
            query_embedding=query_embedding,
            repo_id=repo.id,
            top_k=top_k,
            similarity_threshold=self.retrieval_settings.similarity_threshold,
        )

        if file_filter:
            results = [(chunk, score) for chunk, score in results if fnmatch.fnmatch(chunk.file_path, file_filter)]
        if chunk_type:
            results = [(chunk, score) for chunk, score in results if chunk.chunk_type.value == chunk_type or chunk.chunk_type == chunk_type]

        return cast(list[tuple[Chunk, float]], results[:top_k])

    def search_hybrid(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        max_chunks_per_file: int = 3,
    ) -> list[RetrievedChunk]:
        """Hybrid code search without LLM generation."""
        repo, error = self.require_ready_repository(repo_id)
        if error is not None:
            raise ValueError(error)
        assert repo is not None

        top_k = self._clamp_top_k(top_k)
        budget = RetrievalBudget(
            max_chunks=top_k,
            max_tokens=max(int(max_tokens), 1),
            max_chunks_per_file=max(int(max_chunks_per_file), 1),
        )
        return self._get_hybrid_retriever().retrieve(query=query, repo_id=repo.id, top_k=top_k, budget=budget)

    def get_context_pack(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        max_chunks_per_file: int = 3,
    ) -> ContextPack:
        """Build a retrieval-only context pack for agents and API clients."""
        chunks = self.search_hybrid(
            repo_id=repo_id,
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            max_chunks_per_file=max_chunks_per_file,
        )
        budget = {
            "max_chunks": self._clamp_top_k(top_k),
            "max_tokens": max(int(max_tokens), 1),
            "max_chunks_per_file": max(int(max_chunks_per_file), 1),
        }
        snippets = [ContextSnippet.from_retrieved_chunk(chunk) for chunk in chunks]
        return ContextPack(
            repo_id=repo_id,
            query=query,
            snippets=snippets,
            token_estimate=sum(snippet.token_estimate for snippet in snippets),
            budget=budget,
            capabilities={
                "generation_required": False,
                "retrieval": "hybrid",
                "semantic": True,
                "lexical": True,
            },
        )

    def _clamp_top_k(self, top_k: int) -> int:
        return min(max(int(top_k), 1), self.retrieval_settings.max_top_k)

    def _default_generator_factory(self) -> Any:
        from coderag.generation.generator import ResponseGenerator

        return ResponseGenerator()

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

    def _get_generator(self) -> Any:
        if self._generator is None:
            self._generator = self.generator_factory()
        return self._generator

    def _get_hybrid_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            vectorstore=self._get_vectorstore(),
            embedder=self._get_embedder(),
            similarity_threshold=self.retrieval_settings.similarity_threshold,
        )
