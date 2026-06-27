"""Milestone 0 baseline tests.

These tests pin the governance-critical defaults before feature work continues.
"""

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

import coderag
import coderag.cli as cli_module
from coderag.config import ModelSettings, ServerSettings, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_default_config_versions_match_pyproject() -> None:
    """User-visible version surfaces must agree with package metadata."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    expected_version = pyproject["project"]["version"]
    default_yaml: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "configs" / "default.yaml").read_text()
    )

    assert coderag.__version__ == expected_version
    assert Settings().app_version == expected_version
    assert default_yaml["app"]["version"] == expected_version


def test_env_example_does_not_reintroduce_stale_version_or_broad_bind() -> None:
    """Copying .env.example must preserve the safe Milestone 0 defaults."""
    entries = {}
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key] = value

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert entries["APP_VERSION"] == pyproject["project"]["version"]
    assert entries["SERVER_HOST"] == "127.0.0.1"


def test_readme_configuration_documents_loopback_server_default() -> None:
    """README quick configuration must not tell local users to bind broadly by default."""
    readme = (REPO_ROOT / "README.md").read_text()

    assert "SERVER_HOST=127.0.0.1" in readme


def test_docker_compose_exposes_service_on_host_loopback() -> None:
    """The compose profile can bind inside the container while exposing only host loopback."""
    compose = (REPO_ROOT / "docker-compose.yaml").read_text()

    assert '"127.0.0.1:8000:8000"' in compose


def test_ci_runs_committed_evalfly_smoke_runner() -> None:
    """CI must execute a semantic smoke check for EvalFly cases, not only JSON syntax."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "scripts/evalfly_smoke.py" in workflow
    assert (REPO_ROOT / "scripts" / "evalfly_smoke.py").exists()


def test_default_model_loading_does_not_trust_remote_code() -> None:
    """Remote model code execution must be an explicit opt-in, never a default path."""
    assert ModelSettings().allow_remote_code is False

    embedding_source = (REPO_ROOT / "src" / "coderag" / "indexing" / "embeddings.py").read_text()
    generator_source = (REPO_ROOT / "src" / "coderag" / "generation" / "generator.py").read_text()
    models_yaml = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text())

    assert "trust_remote_code=True" not in embedding_source
    assert "trust_remote_code=True" not in generator_source
    assert models_yaml["llm"]["trust_remote_code"] is False

def test_server_defaults_bind_to_loopback_and_disallow_wildcard_credentialed_cors() -> None:
    """Fresh server defaults must not expose a credentialed API to arbitrary origins."""
    settings = ServerSettings()

    assert settings.host == "127.0.0.1"
    assert "*" not in settings.cors_allowed_origins
    assert not (settings.cors_allow_credentials and settings.cors_allowed_origins == ["*"])


def test_default_yaml_mirrors_safe_server_defaults() -> None:
    """The shipped YAML config must mirror the safe runtime defaults."""
    default_yaml: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "configs" / "default.yaml").read_text()
    )
    server = default_yaml["server"]

    assert server["host"] == "127.0.0.1"
    assert "*" not in server["cors_allowed_origins"]
    assert not (server["cors_allow_credentials"] and server["cors_allowed_origins"] == ["*"])


def test_cli_serve_default_host_is_loopback() -> None:
    """The CLI serve command must inherit the safe loopback default."""
    serve_command = cli_module.cli.commands["serve"]
    host_param = next(param for param in serve_command.params if param.name == "host")

    assert host_param.default == "127.0.0.1"


def test_doctor_does_not_print_configured_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor may report that a key exists, but must never print the secret value."""
    secret = "sk-test-secret-value-that-must-not-leak"
    config_dir = tmp_path / "config"
    config_file = config_dir / "config.json"
    config_dir.mkdir()
    config_file.write_text(json.dumps({"llm_provider": "groq", "llm_api_key": secret}))

    monkeypatch.setattr(cli_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", config_file)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(tmp_path / "data"))

    result = CliRunner().invoke(cli_module.cli, ["doctor"])

    assert result.exit_code == 0
    assert secret not in result.output
    assert "API key configured" in result.output
