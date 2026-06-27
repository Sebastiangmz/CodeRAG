"""UI privacy/profile guard tests."""

import ast
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLERS_PATH = REPO_ROOT / "src" / "coderag" / "ui" / "handlers.py"
METADATA_PATH = REPO_ROOT / "src" / "coderag" / "ui" / "repository_metadata.py"


def _load_metadata_module():
    spec = importlib.util.spec_from_file_location("coderag_ui_repository_metadata_under_test", METADATA_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_ui_ask_question_checks_provider_profile_before_generation_setup():
    module = ast.parse(HANDLERS_PATH.read_text())
    ask_question = _function(module, "ask_question")

    call_names = [name for node in ast.walk(ask_question) if (name := _call_name(node)) is not None]

    assert "generation_block_reason" in call_names
    assert "ResponseGenerator" in call_names
    assert "generate" in call_names
    assert call_names.index("generation_block_reason") < call_names.index("ResponseGenerator")
    assert call_names.index("generation_block_reason") < call_names.index("generate")


def test_ui_repository_loader_fails_closed_on_corrupt_metadata(tmp_path):
    metadata = _load_metadata_module()
    repos_file = tmp_path / "repositories.json"
    repos_file.write_text("{not-json")

    try:
        metadata.load_repositories(repos_file)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("corrupt repositories.json must not load as an empty registry")

    assert repos_file.read_text() == "{not-json"
