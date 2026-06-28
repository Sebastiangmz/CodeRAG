"""Run the committed deterministic subset of the EvalFly smoke suite.

This CI helper is intentionally narrow: it validates the repository-local EvalFly
configuration enough to execute deterministic repository-local assertions in GitHub
Actions, while the full OMP EvalFly CLI remains the authoritative local tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "evals" / "config.json"


class SmokeFailure(Exception):
    """Raised when the committed smoke suite fails."""


def _load_config() -> dict[str, Any]:
    try:
        config = cast(dict[str, Any], json.loads(CONFIG_PATH.read_text()))
    except FileNotFoundError as exc:
        raise SmokeFailure(f"missing EvalFly config: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"invalid EvalFly config JSON: {exc}") from exc

    if config.get("schema_version") != "evalfly.config.v1":
        raise SmokeFailure("evals/config.json must use schema_version evalfly.config.v1")
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise SmokeFailure("evals/config.json must define a non-empty name")
    if not isinstance(config.get("cases"), list) or not config["cases"]:
        raise SmokeFailure("evals/config.json must define at least one case")
    return config


def _run_case(case: dict[str, Any]) -> None:
    case_id = case.get("case_id", "<unknown>")
    if case.get("schema_version") != "evalfly.case.v1":
        raise SmokeFailure(f"{case_id}: expected schema_version evalfly.case.v1")
    if case.get("suite") != "smoke":
        raise SmokeFailure(f"{case_id}: committed runner only supports smoke cases")

    judge = case.get("judge")
    if not isinstance(judge, dict) or judge.get("type") != "deterministic":
        raise SmokeFailure(f"{case_id}: smoke cases must use deterministic judges")

    assertions = judge.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        raise SmokeFailure(f"{case_id}: deterministic judge must include assertions")

    for assertion in assertions:
        if not isinstance(assertion, dict):
            raise SmokeFailure(f"{case_id}: assertion must be an object")
        assertion_type = assertion.get("type")
        raw_path = assertion.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise SmokeFailure(f"{case_id}: assertion requires a path")
        path = REPO_ROOT / raw_path

        if assertion_type == "file_exists":
            if not path.is_file():
                raise SmokeFailure(f"{case_id}: expected file does not exist: {raw_path}")
            continue

        if assertion_type == "file_contains":
            expected_text = assertion.get("text")
            if not isinstance(expected_text, str) or not expected_text:
                raise SmokeFailure(f"{case_id}: file_contains assertion requires text")
            if not path.is_file():
                raise SmokeFailure(f"{case_id}: expected file does not exist: {raw_path}")
            if expected_text not in path.read_text():
                raise SmokeFailure(f"{case_id}: expected {raw_path} to contain {expected_text!r}")
            continue

        raise SmokeFailure(f"{case_id}: unsupported assertion type {assertion_type!r}")


def run_smoke() -> int:
    """Run all committed deterministic smoke cases."""
    config = _load_config()
    cases = config["cases"]
    for case in cases:
        if not isinstance(case, dict):
            raise SmokeFailure("each EvalFly case must be an object")
        _run_case(case)
    print(f"evalfly committed smoke passed: {len(cases)} cases")
    return len(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="smoke", choices=["smoke"])
    parser.parse_args(argv)

    try:
        run_smoke()
    except SmokeFailure as exc:
        print(f"evalfly committed smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
