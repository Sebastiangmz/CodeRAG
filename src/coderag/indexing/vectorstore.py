"""ChromaDB vector store operations."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.chunk import Chunk

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB vector store for chunk storage and retrieval."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.persist_directory = persist_directory or settings.vectorstore.persist_directory
        self.collection_name = collection_name or settings.vectorstore.collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._init_client()
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._init_collection()
        return self._collection

    def _init_client(self) -> None:
        logger.info("Initializing ChromaDB", path=str(self.persist_directory))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

    def _init_collection(self) -> None:
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection initialized", name=self.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]

        # Remove embedding and filter None values (ChromaDB doesn't accept None)
        cleaned_metadatas = []
        for m in metadatas:
            m.pop("embedding", None)
            m.pop("content", None)  # Already stored in documents
            # Filter out None values - ChromaDB only accepts str, int, float, bool
            cleaned = {k: v for k, v in m.items() if v is not None}
            cleaned_metadatas.append(cleaned)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas,
        )

        logger.info("Chunks added to vector store", count=len(chunks))
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        repo_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"repo_id": repo_id},
            include=["documents", "metadatas", "distances"],
        )

        chunks_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity for cosine
                distance = results["distances"][0][i]
                similarity = 1 - distance

                if similarity >= similarity_threshold:
                    metadata = results["metadatas"][0][i]
                    metadata["id"] = chunk_id
                    metadata["content"] = results["documents"][0][i]
                    chunk = Chunk.from_dict(metadata)
                    chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete_repo_chunks(self, repo_id: str) -> int:
        # Get all chunks for this repo
        results = self.collection.get(where={"repo_id": repo_id}, include=[])

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            logger.info("Deleted repo chunks", repo_id=repo_id, count=count)
            return count
        return 0

    def delete_file_chunks(self, repo_id: str, file_path: str) -> int:
        """Delete chunks for a specific file in a repository (for incremental updates)."""
        results = self.collection.get(
            where={"$and": [{"repo_id": repo_id}, {"file_path": file_path}]},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            logger.info("Deleted file chunks", repo_id=repo_id, file_path=file_path, count=count)
            return count
        return 0

    def get_indexed_files(self, repo_id: str) -> set[str]:
        """Get set of file paths indexed for a repository."""
        results = self.collection.get(
            where={"repo_id": repo_id},
            include=["metadatas"],
        )

        files = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if "file_path" in metadata:
                    files.add(metadata["file_path"])
        return files

    def get_repo_chunk_count(self, repo_id: str) -> int:
        results = self.collection.get(where={"repo_id": repo_id}, include=[])
        return len(results["ids"]) if results["ids"] else 0

    def get_all_repo_ids(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        repo_ids = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if "repo_id" in metadata:
                    repo_ids.add(metadata["repo_id"])
        return list(repo_ids)

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("Collection cleared", name=self.collection_name)
