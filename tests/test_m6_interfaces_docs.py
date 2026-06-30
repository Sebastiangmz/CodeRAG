"""Milestone 6 interface and documentation contract tests."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,union-attr,arg-type"

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from coderag.models.repository import Repository, RepositoryStatus


def test_mcp_tool_surface_includes_production_context_tools():
    import coderag.mcp.tools as tools

    expected = {
        "index_repository",
        "update_repository",
        "list_repositories",
        "query_code",
        "search_code",
        "search_hybrid",
        "get_context_pack",
        "find_symbol",
        "find_references",
        "get_blast_radius",
    }

    assert expected <= {name for name in expected if callable(getattr(tools, name, None))}


@pytest.mark.asyncio
async def test_rest_hybrid_search_contract(monkeypatch):
    from coderag.api import routes
    from coderag.api.schemas import HybridSearchRequest
    from coderag.models.response import RetrievedChunk

    repo = Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY)

    class FakeRegistry:
        def get_unique(self, repo_id: str):
            return repo if repo.id == repo_id else None

    class FakeRetrievalService:
        def search_hybrid(self, repo_id: str, query: str, top_k: int, max_tokens: int, max_chunks_per_file: int):
            assert (repo_id, query, top_k, max_tokens, max_chunks_per_file) == (
                "repo-ready",
                "auth flow",
                2,
                512,
                1,
            )
            return [
                RetrievedChunk(
                    chunk_id="chunk-1",
                    content="def login(): pass",
                    file_path="src/auth.py",
                    start_line=10,
                    end_line=12,
                    relevance_score=0.91,
                    chunk_type="function",
                    name="login",
                    retrieval_sources=["semantic", "lexical"],
                    score_breakdown={"combined": 0.91, "lexical": 0.8},
                    token_estimate=4,
                    ranking_reason="semantic and lexical match",
                )
            ]

    monkeypatch.setattr(routes, "registry", FakeRegistry())
    monkeypatch.setattr(routes, "retrieval_service", FakeRetrievalService())

    response = await routes.search_hybrid(
        HybridSearchRequest(repo_id="repo-ready", query="auth flow", top_k=2, max_tokens=512, max_chunks_per_file=1)
    )

    assert response.count == 1
    assert response.results[0].citation == "[src/auth.py:10-12]"
    assert response.results[0].retrieval_sources == ["semantic", "lexical"]
    assert response.results[0].score_breakdown["combined"] == 0.91


def test_cli_hybrid_search_outputs_structured_json(monkeypatch):
    from coderag.cli import cli

    async def search_hybrid(**kwargs):
        assert kwargs == {
            "repo_id": "repo-ready",
            "query": "auth flow",
            "top_k": 2,
            "max_tokens": 512,
            "max_chunks_per_file": 1,
        }
        return {
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "file_path": "src/auth.py",
                    "start_line": 10,
                    "end_line": 12,
                    "citation": "[src/auth.py:10-12]",
                    "retrieval_sources": ["semantic", "lexical"],
                    "score_breakdown": {"combined": 0.91},
                    "content": "def login(): pass",
                }
            ],
            "count": 1,
        }

    monkeypatch.setitem(
        __import__("sys").modules,
        "coderag.mcp.handlers",
        SimpleNamespace(get_mcp_handlers=lambda: SimpleNamespace(search_hybrid=search_hybrid)),
    )

    result = CliRunner().invoke(
        cli,
        [
            "search-hybrid",
            "repo-ready",
            "auth flow",
            "--top-k",
            "2",
            "--max-tokens",
            "512",
            "--max-chunks-per-file",
            "1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["results"][0]["citation"] == "[src/auth.py:10-12]"


def test_readme_positions_verified_context_engine_without_production_overclaim():
    readme = __import__("pathlib").Path("README.md").read_text()

    assert "verified codebase context engine" in readme
    assert "not an autonomous coding agent" in readme
    assert "Production readiness" in readme
    assert "Unsupported graph languages degrade explicitly" in readme
    assert "production-ready" not in readme.lower().replace("not production-ready", "")
