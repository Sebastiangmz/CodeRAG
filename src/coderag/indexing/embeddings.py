"""Embedding generation using nomic-embed-text."""

from typing import Iterator, Optional

import torch
from sentence_transformers import SentenceTransformer

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.chunk import Chunk

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using nomic-embed-text v1.5."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.models.embedding_name
        self.device = self._resolve_device(device or settings.models.embedding_device)
        self.batch_size = batch_size or settings.models.embedding_batch_size
        self._model: Optional[SentenceTransformer] = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device, falling back to CPU if CUDA unavailable."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for embeddings")
            return "cpu"
        return device

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        logger.info("Loading embedding model", model=self.model_name, device=self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        logger.info("Embedding model loaded")

    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        # nomic-embed uses task prefixes
        if is_query:
            text = f"search_query: {text}"
        else:
            text = f"search_document: {text}"

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def generate_embeddings(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> list[list[float]]:
        # Add prefixes
        if is_query:
            texts = [f"search_query: {t}" for t in texts]
        else:
            texts = [f"search_document: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[Chunk]:
        if not chunks:
            return []

        logger.info("Generating embeddings", num_chunks=len(chunks))

        texts = [self._chunk_to_text(chunk) for chunk in chunks]
        embeddings = self.generate_embeddings(texts, is_query=False, show_progress=show_progress)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        logger.info("Embeddings generated", num_chunks=len(chunks))
        return chunks

    def embed_chunks_iter(
        self,
        chunks: Iterator[Chunk],
        batch_size: Optional[int] = None,
    ) -> Iterator[Chunk]:
        batch_size = batch_size or self.batch_size
        batch: list[Chunk] = []

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield from self._embed_batch(batch)
                batch = []

        if batch:
            yield from self._embed_batch(batch)

    def _embed_batch(self, batch: list[Chunk]) -> Iterator[Chunk]:
        texts = [self._chunk_to_text(chunk) for chunk in batch]
        embeddings = self.generate_embeddings(texts, is_query=False, show_progress=False)

        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding
            yield chunk

    def _chunk_to_text(self, chunk: Chunk) -> str:
        parts = []
        if chunk.name:
            parts.append(f"{chunk.chunk_type.value}: {chunk.name}")
        if chunk.metadata.signature:
            parts.append(f"Signature: {chunk.metadata.signature}")
        if chunk.metadata.docstring:
            parts.append(f"Docstring: {chunk.metadata.docstring[:200]}")
        parts.append(f"File: {chunk.file_path}")
        parts.append(chunk.content)
        return "\n".join(parts)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")
