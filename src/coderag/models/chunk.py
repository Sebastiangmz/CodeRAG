"""Chunk entity model for semantic code units."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4


class ChunkType(str, Enum):
    """Type of code chunk."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    TEXT = "text"
    DOCSTRING = "docstring"
    COMMENT = "comment"


@dataclass
class ChunkMetadata:
    """Metadata for a code chunk."""

    file_path: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    language: Optional[str] = None
    name: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_name: Optional[str] = None

    @property
    def line_range(self) -> str:
        """Get line range as string."""
        return f"{self.start_line}-{self.end_line}"

    @property
    def citation(self) -> str:
        """Get citation format."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"


@dataclass
class Chunk:
    """A semantic unit of code or documentation."""

    content: str
    metadata: ChunkMetadata
    repo_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: Optional[list[float]] = None

    @property
    def file_path(self) -> str:
        """Convenience accessor for file path."""
        return self.metadata.file_path

    @property
    def start_line(self) -> int:
        """Convenience accessor for start line."""
        return self.metadata.start_line

    @property
    def end_line(self) -> int:
        """Convenience accessor for end line."""
        return self.metadata.end_line

    @property
    def chunk_type(self) -> ChunkType:
        """Convenience accessor for chunk type."""
        return self.metadata.chunk_type

    @property
    def name(self) -> Optional[str]:
        """Convenience accessor for name."""
        return self.metadata.name

    @property
    def citation(self) -> str:
        """Get citation format."""
        return self.metadata.citation

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "repo_id": self.repo_id,
            "file_path": self.metadata.file_path,
            "start_line": self.metadata.start_line,
            "end_line": self.metadata.end_line,
            "chunk_type": self.metadata.chunk_type.value,
            "language": self.metadata.language,
            "name": self.metadata.name,
            "signature": self.metadata.signature,
            "docstring": self.metadata.docstring,
            "parent_name": self.metadata.parent_name,
        }

    @classmethod
    def from_dict(cls, data: dict, embedding: Optional[list[float]] = None) -> "Chunk":
        """Create from dictionary."""
        metadata = ChunkMetadata(
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            chunk_type=ChunkType(data["chunk_type"]),
            language=data.get("language"),
            name=data.get("name"),
            signature=data.get("signature"),
            docstring=data.get("docstring"),
            parent_name=data.get("parent_name"),
        )
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=metadata,
            repo_id=data["repo_id"],
            embedding=embedding,
        )
