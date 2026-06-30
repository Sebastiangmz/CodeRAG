"""Milestone 5 code intelligence graph tests."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,union-attr"

import json
from types import SimpleNamespace
from typing import Any, cast

from click.testing import CliRunner

from coderag.models.document import Document, DocumentMetadata
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.registry import RepositoryRegistry


def document(content: str, file_path: str, language: str, repo_id: str = "repo-1") -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(file_path=file_path, language=language, line_count=len(content.splitlines())),
        repo_id=repo_id,
    )


def test_graph_extractor_extracts_python_and_typescript_symbols_edges_and_capabilities():
    from coderag.graph.extractor import CodeGraphExtractor

    extractor = CodeGraphExtractor()
    python_graph = extractor.extract(
        document(
            "import os\nfrom auth.tokens import validate_token\n\nclass AuthService:\n    def login(self, token):\n        return validate_token(token)\n\ndef helper():\n    return AuthService()\n",
            "src/auth.py",
            "python",
        )
    )
    ts_graph = extractor.extract(
        document(
            "import { validateToken } from './auth'\nexport class LoginController {\n  login(token: string) {\n    return validateToken(token)\n  }\n}\nexport function helper() { return new LoginController() }\n",
            "src/login.ts",
            "typescript",
        )
    )
    unsupported = extractor.extract(document("body { color: red }", "src/style.css", "css"))

    assert {symbol.name for symbol in python_graph.symbols} >= {"AuthService", "login", "helper"}
    assert {symbol.name for symbol in ts_graph.symbols} >= {"LoginController", "login", "helper"}
    assert any(edge.edge_type == "import" and edge.target_name == "validate_token" for edge in python_graph.edges)
    assert any(edge.edge_type == "call" and edge.target_name == "validateToken" for edge in ts_graph.edges)
    assert unsupported.capabilities["graph_supported"] is False
    assert unsupported.capabilities["fallback"] == "text"


def test_registry_persists_and_replaces_graph_file(tmp_path):
    from coderag.graph.extractor import CodeGraphExtractor

    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3")
    registry.add(Repository(id="repo-1", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    extractor = CodeGraphExtractor()
    first = extractor.extract(document("def login():\n    return True\n", "src/auth.py", "python"))
    second = extractor.extract(document("def logout():\n    return True\n", "src/auth.py", "python"))

    registry.replace_graph_file("repo-1", "src/auth.py", first)
    assert [symbol.name for symbol in registry.find_graph_symbols("repo-1", "login")] == ["login"]

    registry.replace_graph_file("repo-1", "src/auth.py", second)
    assert registry.find_graph_symbols("repo-1", "login") == []
    assert [symbol.name for symbol in registry.find_graph_symbols("repo-1", "logout")] == ["logout"]


def test_graph_service_finds_symbols_references_and_blast_radius(tmp_path):
    from coderag.graph.extractor import CodeGraphExtractor
    from coderag.services.graph import CodeGraphService

    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3")
    registry.add(Repository(id="repo-1", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    extractor = CodeGraphExtractor()
    registry.replace_graph_file(
        "repo-1",
        "src/auth.py",
        extractor.extract(document("def validate_token(token):\n    return bool(token)\n", "src/auth.py", "python")),
    )
    registry.replace_graph_file(
        "repo-1",
        "src/api.py",
        extractor.extract(
            document(
                "from auth import validate_token\n\ndef login(token):\n    return validate_token(token)\n",
                "src/api.py",
                "python",
            )
        ),
    )
    registry.replace_graph_file(
        "repo-1",
        "tests/test_auth.py",
        extractor.extract(
            document(
                "from src.auth import validate_token\n\ndef test_validate_token():\n    assert validate_token('x')\n",
                "tests/test_auth.py",
                "python",
            )
        ),
    )
    service = CodeGraphService(registry=registry)

    symbols = service.find_symbol("repo-1", "validate_token")
    references = service.find_references("repo-1", "validate_token")
    blast = service.get_blast_radius("repo-1", "validate_token")

    assert symbols[0].file_path == "src/auth.py"
    assert {reference.file_path for reference in references} >= {"src/api.py", "tests/test_auth.py"}
    assert blast.symbol == "validate_token"
    assert "src/api.py" in blast.impacted_files
    assert "tests/test_auth.py" in blast.associated_tests
    assert any("call" in reason or "import" in reason for reason in blast.reasons)


def test_graph_adapters_return_json_shapes(tmp_path, monkeypatch):
    from coderag.api import routes
    from coderag.api.schemas import GraphQueryRequest
    from coderag.cli import cli
    from coderag.mcp.handlers import MCPHandlers
    from coderag.models.graph import BlastRadius, CodeReference, CodeSymbol

    symbol = CodeSymbol(repo_id="repo-1", file_path="src/auth.py", name="validate_token", kind="function", language="python", start_line=1, end_line=2)
    reference = CodeReference(repo_id="repo-1", file_path="src/api.py", symbol_name="validate_token", reference_kind="call", start_line=4, end_line=4)
    blast = BlastRadius(
        repo_id="repo-1",
        symbol="validate_token",
        impacted_files=["src/api.py"],
        impacted_symbols=["login"],
        associated_tests=["tests/test_auth.py"],
        edges=[{"edge_type": "call", "source": "login", "target": "validate_token"}],
        reasons=["src/api.py calls validate_token"],
        capabilities={"graph_supported": True},
    )

    class FakeGraphService:
        def find_symbol(self, _repo_id: str, _symbol_name: str):
            return [symbol]

        def find_references(self, _repo_id: str, _symbol_name: str):
            return [reference]

        def get_blast_radius(self, _repo_id: str, _symbol_name: str):
            return blast

    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3")
    registry.add(Repository(id="repo-1", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    fake_graph_service = FakeGraphService()
    monkeypatch.setattr(routes, "registry", registry)
    monkeypatch.setattr(routes, "graph_service", fake_graph_service)

    api_response = routes.find_symbol(GraphQueryRequest(repo_id="repo-1", symbol="validate_token"))
    mcp = MCPHandlers(registry=registry, graph_service=cast("Any", fake_graph_service))
    mcp_result = mcp.find_symbol("repo-1", "validate_token")

    assert api_response.symbols[0].name == "validate_token"
    assert mcp_result["symbols"][0]["file_path"] == "src/auth.py"
    assert json.dumps(mcp.get_blast_radius("repo-1", "validate_token"))

    async def get_blast_radius(**_kwargs):
        return blast.to_dict()

    monkeypatch.setitem(
        __import__("sys").modules,
        "coderag.mcp.handlers",
        SimpleNamespace(get_mcp_handlers=lambda: SimpleNamespace(get_blast_radius=get_blast_radius)),
    )
    result = CliRunner().invoke(cli, ["blast-radius", "repo-1", "validate_token", "--format", "json"])

    assert result.exit_code == 0
    assert "src/api.py" in result.output
