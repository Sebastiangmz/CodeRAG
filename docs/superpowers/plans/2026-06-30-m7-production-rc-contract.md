# Milestone 7 Contract: Production Release Candidate

## Scope

Produce release-candidate evidence for CodeRAG after M0-M6. This milestone does not add product features unless a release blocker is found; it records and gates the evidence needed to call the branch a release candidate.

## Acceptance Criteria

- M7-AC1: A clean install path is exercised from an isolated virtual environment using the built wheel.
- M7-AC2: CI-equivalent local gates pass for M0-M6 tests, lint, typecheck, EvalFly smoke, py_compile, and build.
- M7-AC3: A run record documents CLI, REST, MCP, retrieval/context, citation, graph, and interface evidence; unsupported UI/E2E scope is explicitly named if not exercised.
- M7-AC4: EvalFly run and report artifacts are committed for the release-candidate pass.
- M7-AC5: README release-candidate wording is limited to verified evidence and does not claim unrestricted production readiness.
- M7-AC6: CI includes a release-candidate artifact gate so future changes cannot drop the M7 evidence files silently.

## Non-goals

- No new retrieval, graph, generation, or UI features unless required to fix a release blocker.
- No claim of hosted/SaaS hardening.
- No autonomous code-editing behavior.
