"""UI privacy/profile guard tests."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call"

import ast
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLERS_PATH = REPO_ROOT / "src" / "coderag" / "ui" / "handlers.py"
METADATA_PATH = REPO_ROOT / "src" / "coderag" / "ui" / "repository_metadata.py"
GROUNDING_PATH = REPO_ROOT / "src" / "coderag" / "ui" / "grounding.py"


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


def test_ui_grounding_status_uses_verified_citation_state():
    spec = importlib.util.spec_from_file_location("coderag_ui_grounding_under_test", GROUNDING_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    format_grounding_status = module.format_grounding_status

    assert format_grounding_status(grounded=True, has_citations=True, has_citation_verifications=True) == "Grounded (verified citations)"
    assert (
        format_grounding_status(grounded=False, has_citations=True, has_citation_verifications=True)
        == "Not grounded (unverified citations)"
    )
    assert (
        format_grounding_status(grounded=False, has_citations=False, has_citation_verifications=False)
        == "Not grounded (no verified citations)"
    )

def test_ui_repository_state_uses_shared_registry_not_json_helpers():
    module = ast.parse(HANDLERS_PATH.read_text())
    imported_names = {
        alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    init_method = _function(module, "__init__")

    init_call_names = [name for node in ast.walk(init_method) if (name := _call_name(node)) is not None]

    assert "RepositoryRegistry" in imported_names
    assert "load_repositories" not in imported_names
    assert "save_repositories" not in imported_names
    assert "RepositoryRegistry" in init_call_names


def test_ui_repository_operations_record_durable_jobs():
    module = ast.parse(HANDLERS_PATH.read_text())
    index_method = _function(module, "index_repository")
    update_method = _function(module, "index_repository_incremental")
    delete_method = _function(module, "delete_repository")

    index_calls = [name for node in ast.walk(index_method) if (name := _call_name(node)) is not None]
    update_calls = [name for node in ast.walk(update_method) if (name := _call_name(node)) is not None]
    delete_calls = [name for node in ast.walk(delete_method) if (name := _call_name(node)) is not None]

    assert "begin_job" in index_calls
    assert "finish_job" in index_calls
    assert "record_file_metadata" in index_calls
    assert "begin_job" in update_calls
    assert "finish_job" in update_calls
    update_source = ast.get_source_segment(HANDLERS_PATH.read_text(), update_method)
    assert update_source is not None
    assert update_source.index("begin_job") < update_source.index("last_commit")
    assert update_source.index("begin_job") < update_source.index("clone_path")
    assert "remove_file_metadata" in update_calls
    assert "record_file_metadata" in update_calls
    assert "begin_job" in delete_calls
    assert "finish_job" in delete_calls


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
