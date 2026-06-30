"""Code intelligence graph models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeSymbol:
    repo_id: str
    file_path: str
    name: str
    kind: str
    language: str | None
    start_line: int
    end_line: int
    container: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "container": self.container,
        }


@dataclass(frozen=True)
class CodeReference:
    repo_id: str
    file_path: str
    symbol_name: str
    reference_kind: str
    start_line: int
    end_line: int
    source_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "reference_kind": self.reference_kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class CodeEdge:
    repo_id: str
    file_path: str
    edge_type: str
    target_name: str
    start_line: int
    end_line: int
    source_name: str | None = None

    def to_reference(self) -> CodeReference:
        return CodeReference(
            repo_id=self.repo_id,
            file_path=self.file_path,
            symbol_name=self.target_name,
            reference_kind=self.edge_type,
            start_line=self.start_line,
            end_line=self.end_line,
            source_name=self.source_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "edge_type": self.edge_type,
            "target_name": self.target_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class FileGraph:
    repo_id: str
    file_path: str
    language: str | None
    symbols: list[CodeSymbol] = field(default_factory=list)
    edges: list[CodeEdge] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlastRadius:
    repo_id: str
    symbol: str
    impacted_files: list[str]
    impacted_symbols: list[str]
    associated_tests: list[str]
    edges: list[dict[str, Any]]
    reasons: list[str]
    capabilities: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "symbol": self.symbol,
            "impacted_files": self.impacted_files,
            "impacted_symbols": self.impacted_symbols,
            "associated_tests": self.associated_tests,
            "edges": self.edges,
            "reasons": self.reasons,
            "capabilities": self.capabilities,
        }
