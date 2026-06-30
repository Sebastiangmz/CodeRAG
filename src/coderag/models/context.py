"""Context pack models for retrieval-only agent handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coderag.models.response import RetrievedChunk


@dataclass
class ContextSnippet:
    """A retrieved code snippet included in a context pack."""

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str | None = None
    relevance_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    retrieval_sources: list[str] = field(default_factory=list)
    token_estimate: int = 0
    ranking_reason: str | None = None

    @property
    def citation(self) -> str:
        """Return stable source citation."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"

    @classmethod
    def from_retrieved_chunk(cls, chunk: RetrievedChunk) -> ContextSnippet:
        """Create a context snippet from a retrieved chunk."""
        return cls(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            relevance_score=chunk.relevance_score,
            score_breakdown=dict(chunk.score_breakdown),
            retrieval_sources=list(chunk.retrieval_sources),
            token_estimate=chunk.token_estimate,
            ranking_reason=chunk.ranking_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "citation": self.citation,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "relevance_score": self.relevance_score,
            "score_breakdown": self.score_breakdown,
            "retrieval_sources": self.retrieval_sources,
            "token_estimate": self.token_estimate,
            "ranking_reason": self.ranking_reason,
        }


@dataclass
class ContextPack:
    """Retrieval-only context bundle for agents and adapters."""

    repo_id: str
    query: str
    snippets: list[ContextSnippet]
    token_estimate: int
    budget: dict[str, int]
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "query": self.query,
            "snippets": [snippet.to_dict() for snippet in self.snippets],
            "token_estimate": self.token_estimate,
            "budget": self.budget,
            "capabilities": self.capabilities,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Context Pack: {self.repo_id}",
            "",
            f"Query: {self.query}",
            f"Token estimate: {self.token_estimate}",
            f"Generation required: {self.capabilities.get('generation_required', False)}",
            "",
            "## Snippets",
        ]
        if not self.snippets:
            lines.append("No snippets retrieved.")
            return "\n".join(lines)

        for index, snippet in enumerate(self.snippets, 1):
            title = f"{index}. {snippet.citation}"
            if snippet.name:
                title += f" — {snippet.name}"
            lines.extend(
                [
                    "",
                    f"### {title}",
                    f"Sources: {', '.join(snippet.retrieval_sources) or 'none'}",
                    f"Ranking: {snippet.ranking_reason or 'n/a'}",
                    "```",
                    snippet.content,
                    "```",
                ]
            )
        return "\n".join(lines)
