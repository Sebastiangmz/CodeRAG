"""Milestone 4 hybrid retrieval and context pack tests."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call"

from coderag.models.chunk import Chunk, ChunkMetadata, ChunkType
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.registry import RepositoryRegistry


def chunk(
    *,
    content: str,
    file_path: str,
    start_line: int,
    end_line: int,
    name: str | None = None,
    chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or f"{file_path}:{start_line}-{end_line}",
        repo_id="repo-1",
        content=content,
        metadata=ChunkMetadata(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            name=name,
        ),
    )


class FakeVectorStore:
    def __init__(self, vector_results: list[tuple[Chunk, float]], repo_chunks: list[Chunk]) -> None:
        self.vector_results = vector_results
        self.repo_chunks = repo_chunks

    def query(self, query_embedding, repo_id: str, top_k: int, similarity_threshold: float = 0.0):
        _ = (query_embedding, repo_id, top_k, similarity_threshold)
        return self.vector_results[:top_k]

    def get_repo_chunks(self, repo_id: str) -> list[Chunk]:
        _ = repo_id
        return self.repo_chunks


class FakeEmbedder:
    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        _ = (text, is_query)
        return [0.0]


def test_hybrid_retrieval_surfaces_lexical_exact_match_ahead_of_unrelated_vector_hit():
    from coderag.retrieval.hybrid import HybridRetriever, RetrievalBudget

    auth = chunk(
        content="def validate_token(token):\n    return token.startswith('auth_')",
        file_path="src/auth.py",
        start_line=10,
        end_line=11,
        name="validate_token",
    )
    billing = chunk(
        content="def charge_card(card):\n    return gateway.charge(card)",
        file_path="src/billing.py",
        start_line=3,
        end_line=4,
        name="charge_card",
    )
    retriever = HybridRetriever(
        vectorstore=FakeVectorStore(vector_results=[(billing, 0.95)], repo_chunks=[auth, billing]),
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve("auth token validation", "repo-1", top_k=2, budget=RetrievalBudget(max_chunks=2, max_tokens=1000, max_chunks_per_file=2))

    assert [result.file_path for result in results] == ["src/auth.py", "src/billing.py"]
    assert results[0].retrieval_sources == ["lexical"]
    assert results[0].score_breakdown["lexical"] > 0
    assert results[1].score_breakdown["vector"] == 0.95


def test_hybrid_retrieval_dedupes_overlapping_vector_and_lexical_candidates():
    from coderag.retrieval.hybrid import HybridRetriever, RetrievalBudget

    lexical = chunk(
        content="class AuthService:\n    def validate_token(self, token):\n        return True",
        file_path="src/auth.py",
        start_line=1,
        end_line=3,
        name="AuthService",
        chunk_id="auth-lexical",
    )
    vector = chunk(
        content=lexical.content,
        file_path="src/auth.py",
        start_line=2,
        end_line=4,
        name="validate_token",
        chunk_id="auth-vector",
    )
    retriever = HybridRetriever(
        vectorstore=FakeVectorStore(vector_results=[(vector, 0.8)], repo_chunks=[lexical]),
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve("validate token", "repo-1", top_k=5, budget=RetrievalBudget(max_chunks=5, max_tokens=1000, max_chunks_per_file=5))

    assert len(results) == 1
    assert results[0].file_path == "src/auth.py"
    assert results[0].retrieval_sources == ["vector", "lexical"]
    assert results[0].score_breakdown["vector"] == 0.8
    assert results[0].score_breakdown["lexical"] > 0


def test_hybrid_retrieval_applies_deterministic_budget_caps():
    from coderag.retrieval.hybrid import HybridRetriever, RetrievalBudget

    chunks = [
        chunk(content="auth token " * 20, file_path="src/auth.py", start_line=1, end_line=2, name="first"),
        chunk(content="auth token " * 20, file_path="src/auth.py", start_line=5, end_line=6, name="second"),
        chunk(content="auth token " * 20, file_path="src/users.py", start_line=1, end_line=2, name="third"),
    ]
    retriever = HybridRetriever(vectorstore=FakeVectorStore(vector_results=[], repo_chunks=chunks), embedder=FakeEmbedder())

    results = retriever.retrieve("auth token", "repo-1", top_k=10, budget=RetrievalBudget(max_chunks=10, max_tokens=80, max_chunks_per_file=1))

    assert len(results) == 2
    assert [result.file_path for result in results] == ["src/auth.py", "src/users.py"]
    assert sum(result.token_estimate for result in results) <= 80


def test_retrieval_service_context_pack_uses_hybrid_results_without_generation(tmp_path):
    from coderag.models.context import ContextPack
    from coderag.services.retrieval import RetrievalService

    repo = Repository(
        id="repo-1",
        url="https://github.com/example/repo",
        branch="main",
        status=RepositoryStatus.READY,
    )
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3")
    registry.add(repo)
    auth = chunk(
        content="def validate_token(token):\n    return token.startswith('auth_')",
        file_path="src/auth.py",
        start_line=10,
        end_line=11,
        name="validate_token",
    )
    service = RetrievalService(
        registry=registry,
        vectorstore=FakeVectorStore(vector_results=[], repo_chunks=[auth]),
        embedder=FakeEmbedder(),
    )

    pack = service.get_context_pack("repo-1", "auth token validation", top_k=5, max_tokens=1000)

    assert isinstance(pack, ContextPack)
    assert pack.repo_id == "repo-1"
    assert pack.query == "auth token validation"
    assert pack.snippets[0].file_path == "src/auth.py"
    assert pack.snippets[0].citation == "[src/auth.py:10-11]"
    assert pack.capabilities["generation_required"] is False
    assert "auth token validation" in pack.to_markdown()


def test_legacy_retriever_uses_hybrid_ranking_when_dependencies_are_injected():
    from coderag.retrieval.retriever import Retriever

    auth = chunk(
        content="def validate_token(token):\n    return token.startswith('auth_')",
        file_path="src/auth.py",
        start_line=10,
        end_line=11,
        name="validate_token",
    )
    billing = chunk(
        content="def charge_card(card):\n    return gateway.charge(card)",
        file_path="src/billing.py",
        start_line=3,
        end_line=4,
        name="charge_card",
    )
    retriever = Retriever(
        vectorstore=FakeVectorStore(vector_results=[(billing, 0.95)], repo_chunks=[auth, billing]),
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve("auth token validation", "repo-1", top_k=2)

    assert [result.file_path for result in results] == ["src/auth.py", "src/billing.py"]
    assert results[0].retrieval_sources == ["lexical"]

def test_cli_context_pack_renders_markdown(monkeypatch):
    import sys
    from types import SimpleNamespace

    from click.testing import CliRunner

    from coderag.cli import cli

    async def get_context_pack(**kwargs):
        return {
            "repo_id": kwargs["repo_id"],
            "query": kwargs["query"],
            "snippets": [
                {
                    "citation": "[src/auth.py:10-11]",
                    "content": "def validate_token(token): pass",
                    "retrieval_sources": ["lexical"],
                    "ranking_reason": "lexical=1.000",
                }
            ],
            "token_estimate": 4,
            "capabilities": {"generation_required": False},
        }

    monkeypatch.setitem(
        sys.modules,
        "coderag.mcp.handlers",
        SimpleNamespace(get_mcp_handlers=lambda: SimpleNamespace(get_context_pack=get_context_pack)),
    )

    result = CliRunner().invoke(cli, ["context-pack", "repo-1", "auth token", "--format", "markdown"])

    assert result.exit_code == 0
    assert "Context Pack: repo-1" in result.output
    assert "[src/auth.py:10-11]" in result.output
