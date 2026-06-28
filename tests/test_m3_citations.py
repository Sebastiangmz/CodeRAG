"""Milestone 3 verified citation engine tests."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,method-assign,attr-defined"
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from click.testing import CliRunner

from coderag.generation.citations import CitationParser, CitationVerifier
from coderag.models.query import Query
from coderag.models.response import Citation, RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATIONS_PATH = REPO_ROOT / "src" / "coderag" / "generation" / "citations.py"


def _citation_verifier_class():
    spec = importlib.util.spec_from_file_location("coderag_generation_citations_under_test", CITATIONS_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CitationVerifier


def chunk(
    file_path: str = "src/auth.py",
    start_line: int = 10,
    end_line: int = 20,
    chunk_id: str = "chunk-1",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content="def authenticate():\n    return True",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        relevance_score=0.95,
        chunk_type="function",
        name="authenticate",
    )


def test_citation_verifier_marks_overlapping_retrieved_chunk_verified():
    verifier = _citation_verifier_class()()

    result = verifier.verify(
        [Citation(file_path="src/auth.py", start_line=12, end_line=15)],
        [chunk()],
    )

    assert result.grounded is True
    assert result.verifications[0].verified is True
    assert result.verifications[0].reason == "verified"
    assert result.verifications[0].chunk_id == "chunk-1"


def test_citation_verifier_rejects_hallucinated_path():
    verifier = _citation_verifier_class()()

    result = verifier.verify(
        [Citation(file_path="src/missing.py", start_line=12, end_line=15)],
        [chunk()],
    )

    assert result.grounded is False
    assert result.verifications[0].verified is False
    assert result.verifications[0].reason == "path_not_retrieved"


def test_citation_verifier_rejects_non_overlapping_lines():
    verifier = _citation_verifier_class()()

    result = verifier.verify(
        [Citation(file_path="src/auth.py", start_line=30, end_line=40)],
        [chunk()],
    )

    assert result.grounded is False
    assert result.verifications[0].verified is False
    assert result.verifications[0].reason == "line_range_not_retrieved"


def test_citation_verifier_rejects_invalid_line_range():
    verifier = _citation_verifier_class()()

    result = verifier.verify(
        [Citation(file_path="src/auth.py", start_line=20, end_line=10)],
        [chunk()],
    )

    assert result.grounded is False
    assert result.verifications[0].verified is False
    assert result.verifications[0].reason == "invalid_line_range"


def test_citation_verifier_requires_all_citations_to_be_verified_for_grounding():
    verifier = _citation_verifier_class()()

    result = verifier.verify(
        [
            Citation(file_path="src/auth.py", start_line=12, end_line=15),
            Citation(file_path="src/missing.py", start_line=1, end_line=2),
        ],
        [chunk()],
    )

    assert result.grounded is False
    assert [verification.verified for verification in result.verifications] == [True, False]


def test_citation_verifier_returns_not_grounded_for_no_citations():
    verifier = _citation_verifier_class()()

    result = verifier.verify([], [chunk()])

    assert result.grounded is False
    assert result.verifications == []


class FakeRetriever:
    def __init__(self, retrieved_chunks: list[RetrievedChunk]) -> None:
        self.retrieved_chunks = retrieved_chunks

    def retrieve_with_context(self, question: str, repo_id: str, top_k: int):
        _ = (question, repo_id, top_k)
        return self.retrieved_chunks, "retrieved context"


def _response_generator_class(monkeypatch):
    retriever_module = ModuleType("coderag.retrieval.retriever")
    retriever_module.Retriever = object
    monkeypatch.setitem(sys.modules, "coderag.retrieval.retriever", retriever_module)

    generator_path = REPO_ROOT / "src" / "coderag" / "generation" / "generator.py"
    spec = importlib.util.spec_from_file_location("coderag_generation_generator_under_test", generator_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ResponseGenerator


def test_response_generator_grounds_only_when_all_citations_verify(monkeypatch):
    ResponseGenerator = _response_generator_class(monkeypatch)
    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator.retriever = FakeRetriever([chunk()])
    generator.citation_parser = CitationParser()
    generator.citation_verifier = CitationVerifier()
    generator.provider = "stub"
    generator._generate_api = lambda _prompt: "Uses auth [src/auth.py:12-15] and missing [src/missing.py:1-2]."

    result = generator.generate(Query(question="How?", repo_id="repo-1"))

    assert result.grounded is False
    assert [verification.verified for verification in result.citation_verifications] == [True, False]


def test_cli_query_surfaces_citation_verification_status():
    cli_source = (REPO_ROOT / "src" / "coderag" / "cli.py").read_text()

    assert "citation_verifications" in cli_source
    assert "verified" in cli_source
    assert "unverified" in cli_source



class FakeCliHandlers:
    async def query_code(self, repo_id: str, question: str, top_k: int = 5):
        _ = (repo_id, question, top_k)
        return {
            "answer": "Answer",
            "citations": [{"file_path": "src/app.py", "start_line": 1, "end_line": 2}],
            "citation_verifications": [
                {"file_path": "src/app.py", "start_line": 1, "end_line": 2, "verified": True, "reason": "verified"},
                {
                    "file_path": "src/missing.py",
                    "start_line": 1,
                    "end_line": 2,
                    "verified": False,
                    "reason": "path_not_retrieved",
                },
            ],
            "evidence": [],
            "grounded": False,
        }


def test_cli_query_json_output_includes_serialized_citation_verifications(monkeypatch):
    import coderag.mcp.handlers as mcp_handlers
    from coderag.cli import cli

    monkeypatch.setattr(mcp_handlers, "get_mcp_handlers", lambda: FakeCliHandlers())

    result = CliRunner().invoke(cli, ["query", "repo-1", "How?", "--format", "json"])

    assert result.exit_code == 0
    payload_text = result.output.split("\n", 2)[2]
    payload = json.loads(payload_text)
    assert payload["citation_verifications"][1]["reason"] == "path_not_retrieved"


def test_cli_query_text_output_marks_verified_and_unverified_citations(monkeypatch):
    import coderag.mcp.handlers as mcp_handlers
    from coderag.cli import cli

    monkeypatch.setattr(mcp_handlers, "get_mcp_handlers", lambda: FakeCliHandlers())

    result = CliRunner().invoke(cli, ["query", "repo-1", "How?"])

    assert result.exit_code == 0
    assert "src/app.py:1-2 [verified: verified]" in result.output
    assert "src/missing.py:1-2 [unverified: path_not_retrieved]" in result.output
