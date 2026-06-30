"""Service seam for code intelligence graph queries."""

from __future__ import annotations

from coderag.models.graph import BlastRadius, CodeReference, CodeSymbol
from coderag.models.repository import RepositoryStatus
from coderag.services.registry import RepositoryRegistry


class CodeGraphService:
    """Query persisted symbol/reference graph facts."""

    def __init__(self, registry: RepositoryRegistry | None = None) -> None:
        self.registry = registry or RepositoryRegistry()

    def find_symbol(self, repo_id: str, symbol_name: str) -> list[CodeSymbol]:
        self._require_ready(repo_id)
        return self.registry.find_graph_symbols(repo_id, symbol_name)

    def find_references(self, repo_id: str, symbol_name: str) -> list[CodeReference]:
        self._require_ready(repo_id)
        return [edge.to_reference() for edge in self.registry.find_graph_edges(repo_id, symbol_name)]

    def get_blast_radius(self, repo_id: str, symbol_name: str) -> BlastRadius:
        self._require_ready(repo_id)
        symbols = self.find_symbol(repo_id, symbol_name)
        edges = self.registry.find_graph_edges(repo_id, symbol_name)
        impacted_files = sorted({edge.file_path for edge in edges})
        impacted_symbols = sorted({edge.source_name for edge in edges if edge.source_name})
        associated_tests = sorted(path for path in impacted_files if _is_test_file(path))
        reasons = [f"{edge.file_path}:{edge.start_line} {edge.edge_type}s {symbol_name}" for edge in edges]
        return BlastRadius(
            repo_id=repo_id,
            symbol=symbol_name,
            impacted_files=impacted_files,
            impacted_symbols=impacted_symbols,
            associated_tests=associated_tests,
            edges=[edge.to_dict() for edge in edges],
            reasons=reasons,
            capabilities={
                "graph_supported": bool(symbols or edges),
                "fallback": "text" if not symbols and not edges else None,
            },
        )

    def _require_ready(self, repo_id: str) -> None:
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"Repository not found: {repo_id}")
        if repo.status is not RepositoryStatus.READY:
            raise ValueError(f"Repository not ready: status is {repo.status.value}")


def _is_test_file(path: str) -> bool:
    lowered = path.lower()
    return "/test" in lowered or lowered.startswith("test") or lowered.endswith("_test.py") or ".test." in lowered
