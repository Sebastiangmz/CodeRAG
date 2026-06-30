"""Retrieval module for semantic search."""

from typing import Any

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.response import RetrievedChunk
from coderag.retrieval.hybrid import HybridRetriever, RetrievalBudget

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant chunks for a query."""

    def __init__(
        self,
        vectorstore: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.vectorstore = vectorstore or self._default_vectorstore()
        self.embedder = embedder or self._default_embedder()
        self.default_top_k = settings.retrieval.default_top_k
        self.max_top_k = settings.retrieval.max_top_k
        self.similarity_threshold = settings.retrieval.similarity_threshold

    def retrieve(
        self,
        query: str,
        repo_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        top_k = min(top_k or self.default_top_k, self.max_top_k)
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        logger.info("Retrieving chunks", query=query[:100], repo_id=repo_id, top_k=top_k)

        retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            embedder=self.embedder,
            similarity_threshold=threshold,
        )
        retrieved_chunks = retriever.retrieve(
            query=query,
            repo_id=repo_id,
            top_k=top_k,
            budget=RetrievalBudget(max_chunks=top_k, max_tokens=4000, max_chunks_per_file=3),
        )
        logger.info("Chunks retrieved", count=len(retrieved_chunks))
        return retrieved_chunks

    def _default_vectorstore(self) -> Any:
        from coderag.indexing.vectorstore import VectorStore

        return VectorStore()

    def _default_embedder(self) -> Any:
        from coderag.indexing.embeddings import EmbeddingGenerator

        return EmbeddingGenerator()

    def retrieve_with_context(
        self,
        query: str,
        repo_id: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        chunks = self.retrieve(query, repo_id, top_k)

        # Build context string for LLM
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {chunk.citation}\n"
                f"Type: {chunk.chunk_type}"
                f"{f' | Name: {chunk.name}' if chunk.name else ''}\n"
                f"```\n{chunk.content}\n```\n"
            )

        context = "\n".join(context_parts) if context_parts else "No relevant code found."

        return chunks, context
