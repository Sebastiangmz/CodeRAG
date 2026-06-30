"""Models module: Core dataclasses and entities."""

from coderag.models.chunk import Chunk, ChunkMetadata, ChunkType
from coderag.models.context import ContextPack, ContextSnippet
from coderag.models.document import Document, DocumentMetadata
from coderag.models.repository import Repository, RepositoryStatus
from coderag.models.response import Citation, Query, Response, RetrievedChunk

__all__ = [
    "ContextPack",
    "ContextSnippet",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkMetadata",
    "ChunkType",
    "Query",
    "Response",
    "Citation",
    "RetrievedChunk",
    "Repository",
    "RepositoryStatus",
]
