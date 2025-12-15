"""Retrieval module for semantic search."""

from typing import Optional

from coderag.config import get_settings
from coderag.indexing.embeddings import EmbeddingGenerator
from coderag.indexing.vectorstore import VectorStore
from coderag.logging import get_logger
from coderag.models.chunk import Chunk
from coderag.models.response import RetrievedChunk

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant chunks for a query."""

    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
    ) -> None:
        settings = get_settings()
        self.vectorstore = vectorstore or VectorStore()
        self.embedder = embedder or EmbeddingGenerator()
        self.default_top_k = settings.retrieval.default_top_k
        self.max_top_k = settings.retrieval.max_top_k
        self.similarity_threshold = settings.retrieval.similarity_threshold

    def retrieve(
        self,
        query: str,
        repo_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        top_k = min(top_k or self.default_top_k, self.max_top_k)
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        logger.info("Retrieving chunks", query=query[:100], repo_id=repo_id, top_k=top_k)

        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query, is_query=True)

        # Search vector store
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            repo_id=repo_id,
            top_k=top_k,
            similarity_threshold=threshold,
        )

        # Convert to RetrievedChunk
        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk.id,
                content=chunk.content,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                relevance_score=score,
                chunk_type=chunk.chunk_type.value,
                name=chunk.name,
            )
            retrieved_chunks.append(retrieved_chunk)

        logger.info("Chunks retrieved", count=len(retrieved_chunks))
        return retrieved_chunks

    def retrieve_with_context(
        self,
        query: str,
        repo_id: str,
        top_k: Optional[int] = None,
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
