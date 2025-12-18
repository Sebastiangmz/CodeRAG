"""Repository entity model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4


class RepositoryStatus(str, Enum):
    """Repository indexing status."""

    PENDING = "pending"
    CLONING = "cloning"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


@dataclass
class Repository:
    """Represents an indexed GitHub repository."""

    url: str
    branch: str = "main"
    id: str = field(default_factory=lambda: str(uuid4()))
    clone_path: Optional[Path] = None
    indexed_at: Optional[datetime] = None
    chunk_count: int = 0
    status: RepositoryStatus = RepositoryStatus.PENDING
    error_message: Optional[str] = None
    last_commit: Optional[str] = None  # SHA of last indexed commit (for incremental updates)

    @property
    def name(self) -> str:
        """Extract repository name from URL."""
        return self.url.rstrip("/").split("/")[-1].replace(".git", "")

    @property
    def owner(self) -> str:
        """Extract repository owner from URL."""
        parts = self.url.rstrip("/").split("/")
        return parts[-2] if len(parts) >= 2 else ""

    @property
    def full_name(self) -> str:
        """Get owner/repo format."""
        return f"{self.owner}/{self.name}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "url": self.url,
            "branch": self.branch,
            "clone_path": str(self.clone_path) if self.clone_path else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "chunk_count": self.chunk_count,
            "status": self.status.value,
            "error_message": self.error_message,
            "last_commit": self.last_commit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Repository":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            url=data["url"],
            branch=data.get("branch", "main"),
            clone_path=Path(data["clone_path"]) if data.get("clone_path") else None,
            indexed_at=datetime.fromisoformat(data["indexed_at"])
            if data.get("indexed_at")
            else None,
            chunk_count=data.get("chunk_count", 0),
            status=RepositoryStatus(data.get("status", "pending")),
            error_message=data.get("error_message"),
            last_commit=data.get("last_commit"),
        )
