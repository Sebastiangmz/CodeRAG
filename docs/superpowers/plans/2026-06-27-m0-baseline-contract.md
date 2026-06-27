# CodeRAG Milestone 0 Baseline Contract

Slice: `CRAG-20260627-M0-BASELINE`
Parent plan: `docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`

## Scope

Milestone 0 establishes the delivery harness and fixes the highest-risk baseline contradictions before feature work begins. It does not implement hybrid retrieval, verified citations, graph extraction, SQLite registry, or autonomous agent behavior.

## Frozen acceptance criteria for this slice

### M0-AC1: Governance

Pass requires:
- SpecSafe slice `CRAG-20260627-M0-BASELINE` is open before edits.
- This contract exists before implementation edits.
- EvalFly project shape exists and validates.

Fail if:
- Product code changes before the slice/contract/eval baseline exist.
- Missing evals are relabeled as not applicable.

Required evidence:
- SpecSafe status.
- `evalfly validate` output.

### M0-AC2: Version baseline

Pass requires:
- Package runtime version, default settings version, and default YAML version agree with `pyproject.toml`.

Fail if:
- Any user-visible surface still reports `0.1.0` while package metadata reports `0.1.2`.

Required evidence:
- Targeted pytest for version synchronization.

### M0-AC3: Safe default server exposure

Pass requires:
- Default server bind host is `127.0.0.1`.
- CLI `serve` default host is `127.0.0.1`.
- CORS defaults do not combine wildcard origins with credentials.
- Default YAML mirrors safe server defaults.

Fail if:
- A fresh default serve path binds to `0.0.0.0`.
- Default CORS allows `*` with credentials.

Required evidence:
- Targeted pytest for settings, CLI defaults, and YAML defaults.

### M0-AC4: Secret redaction smoke

Pass requires:
- `coderag doctor` does not print configured API key values.

Fail if:
- Any API key value appears in doctor output.

Required evidence:
- Targeted pytest using isolated CLI config.

### M0-AC5: CI baseline

Pass requires:
- A GitHub Actions validation workflow exists separately from publish.
- It includes test, lint, typecheck/baseline, package build, and eval configuration validation steps.

Fail if:
- Publishing remains the only workflow.

Required evidence:
- Workflow file exists.
- EvalFly smoke case checks the workflow file exists.

## Evidence plan

Run after implementation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_m0_baseline.py
bun run ~/.omp/agent/skills/evalfly/bin/evalfly.ts validate
bun run ~/.omp/agent/skills/evalfly/bin/evalfly.ts run --suite smoke --commit-range main..HEAD
bun run ~/.omp/agent/skills/evalfly/bin/evalfly.ts check --suite smoke --commit-range main..HEAD
```

Optional if dependency installation completes:

```bash
.venv/bin/python -m build
```
