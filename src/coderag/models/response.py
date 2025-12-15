"""Response entity models for Q&A results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class Citation:
    """A reference to source code location."""

    file_path: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        """Format as citation string."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"

    @classmethod
    def parse(cls, citation_str: str) -> Optional["Citation"]:
        """Parse citation from string format [file:start-end]."""
        try:
            citation_str = citation_str.strip("[]")
            if ":" not in citation_str:
                return None
            file_path, line_range = citation_str.rsplit(":", 1)
            if "-" in line_range:
                start, end = line_range.split("-")
                return cls(
                    file_path=file_path,
                    start_line=int(start),
                    end_line=int(end),
                )
            else:
                line = int(line_range)
                return cls(file_path=file_path, start_line=line, end_line=line)
        except (ValueError, IndexError):
            return None


@dataclass
class RetrievedChunk:
    """A chunk retrieved for answering a query."""

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    chunk_type: str
    name: Optional[str] = None

    @property
    def citation(self) -> str:
        """Get citation format."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"


@dataclass
class Query:
    """A user's question about a repository."""

    question: str
    repo_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    top_k: int = 5


@dataclass
class Response:
    """The system's answer to a query."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    grounded: bool
    query_id: str = ""
    confidence_score: float = 0.0

    @property
    def has_evidence(self) -> bool:
        """Check if response has supporting evidence."""
        return len(self.retrieved_chunks) > 0

    @property
    def citation_count(self) -> int:
        """Count of citations in response."""
        return len(self.citations)

    def format_evidence(self) -> str:
        """Format evidence section for display."""
        if not self.retrieved_chunks:
            return "No evidence retrieved."

        lines = ["## Evidence\n"]
        for i, chunk in enumerate(self.retrieved_chunks, 1):
            lines.append(f"### {i}. {chunk.citation} (Score: {chunk.relevance_score:.3f})")
            if chunk.name:
                lines.append(f"**{chunk.chunk_type}**: `{chunk.name}`\n")
            lines.append("```")
            lines.append(chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""))
            lines.append("```\n")

        return "\n".join(lines)
