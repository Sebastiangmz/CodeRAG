# CodeRAG Milestone 2 SQLite Registry and Durable Job State Contract

Slice: `CRAG-20260627-M2-SQLITE-REGISTRY`
Parent plan: `docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`
Branch: `feat/m2-sqlite-registry-state`

## Scope

Milestone 2 replaces fragile JSON repository metadata with a SQLite-backed registry and durable job records while keeping the Milestone 1 service seam. This milestone is a persistence/state milestone: it must not introduce hybrid retrieval, verified citation semantics, graph extraction, context packs, or autonomous editing behavior.

## Source of truth

1. Roadmap AC-3 and Milestone 2 (`docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`).
2. Current `main` after PR #2 merge.
3. Existing public API/MCP/CLI/UI repository state behavior unless this contract explicitly narrows it.
4. Pre-1.0 alpha users may either receive a deterministic `repositories.json` migration or an explicit reset-path behavior with tests; this milestone chooses migration because the existing Repository model already serializes the current alpha metadata shape.

## Frozen acceptance criteria

### M2-AC1: SQLite registry schema and migration

Pass requires:
- `RepositoryRegistry` uses SQLite as the canonical store for repository metadata.
- Schema creation is explicit and versioned enough for current migrations, not implicit ad-hoc table creation at each call site.
- The registry tracks repo id, URL/path, branch, head SHA, indexed_at, status, error, file count, chunk count, and index/provider metadata fields.
- Existing `repositories.json` entries are migrated into SQLite on first registry load when a SQLite DB does not already contain repositories.
- Corrupt `repositories.json` still fails closed and is not overwritten or silently ignored during migration.

Fail if:
- Repository metadata remains canonical in a global dict or `repositories.json` after M2.
- A valid existing `repositories.json` user loses repositories without an explicit tested reset path.
- Migration hides corrupt JSON by creating an empty SQLite registry.

Required evidence:
- Unit tests using a temp SQLite DB for schema creation, add/update/list/get/remove, partial-id behavior, and reopen/restart persistence.
- Migration test from a sample `repositories.json` into SQLite.
- Corrupt JSON migration test proving fail-closed behavior.

### M2-AC2: Durable indexing job state

Pass requires:
- Registry has durable job records for index/update/delete operations.
- Job records include id, repo id, operation, status, error, started_at, finished_at, and counters for files/chunks when available.
- `IndexingService` records job start/success/failure for full indexing and deletion.
- Failed indexing records both repository error status and durable job error.
- Restarting a process can read the last job state from SQLite.

Fail if:
- Job state only exists in memory or only in logs.
- Indexing failure persists repository error but no job error.
- Delete removes all evidence of the delete operation before callers can inspect its job state.

Required evidence:
- Unit tests proving successful index job, failed index job, and delete job state are persisted in temp SQLite.
- Restart simulation test reopening the registry and reading the recorded jobs.

### M2-AC3: Shared registry across API/MCP/CLI/UI

Pass requires:
- API, MCP, indexing/retrieval services, and UI repository listing/status paths use the same `RepositoryRegistry` abstraction.
- Public response shapes remain compatible unless covered by a migration note and regression test.
- REST destructive operations continue to reject ambiguous partial ids before delete/query/get service calls.
- MCP first-match prefix compatibility from M1 remains unless a test documents an intentional compatibility change.

Fail if:
- UI still owns independent `repositories.json` load/save state for repository metadata.
- API and MCP can report conflicting repository state after a process restart.
- M1 partial-id ambiguity safety regresses.

Required evidence:
- API/MCP shared-state integration test using the same temp SQLite DB.
- UI repository table/status test using the shared registry path.
- Regression test for ambiguous REST delete/query/get behavior or shared helper coverage plus destructive delete test.

### M2-AC4: Governance and evidence

Pass requires:
- SpecSafe slice `CRAG-20260627-M2-SQLITE-REGISTRY` is open before implementation edits and closed only after verification.
- EvalFly smoke/config is extended with M2 deterministic registry/job-state checks, or an explicit committed reason documents why deeper state evals wait for a later fixture layer.
- CI targeted tests include M2 registry/job-state tests.
- Work is committed on `feat/m2-sqlite-registry-state`, pushed after every commit, and opened as a PR after local verification.

Fail if:
- Tests/evals are weakened, deleted, bypassed, or relabeled to pass.
- M2 is declared production-ready or claims hybrid/citation/graph/context-pack capabilities.
- The SQLite schema is only exercised through mocks that bypass the real registry persistence layer.

Required evidence:
- SpecSafe status showing the active slice.
- Targeted pytest output for M0/M1/M2 tests and affected API/MCP/UI tests.
- Ruff and mypy output for touched modules and tests.
- `scripts/evalfly_smoke.py --suite smoke` output.
- Package build output.
- Post-implementation review against this contract.

## Implementation boundaries

Allowed:
- Replace `RepositoryRegistry` internals with SQLite while preserving its public methods where practical.
- Add registry/job dataclasses or enums inside `src/coderag/services/registry.py` or a focused sibling module.
- Add SQLite schema/migration helpers.
- Add tests under `tests/test_m2_registry.py` and update M1/API/MCP/UI tests to use temp SQLite DBs.
- Route UI repository metadata listing/status through `RepositoryRegistry`.
- Extend EvalFly smoke with deterministic static checks for SQLite registry/job-state artifacts.

Not allowed in M2:
- Hybrid/BM25 retrieval.
- Verified citation semantics or `grounded=True` behavior changes.
- Code graph extraction.
- Context pack APIs.
- Autonomous editing-agent behavior.

## Evidence commands

Run before PR:

```bash
bun run /Users/sebastian/.omp/omp-pantheon/skills/specsafe/bin/specsafe.ts status
PYTHONPATH=src .venv/bin/python -m pytest tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py
.venv/bin/python -m ruff check src/coderag/services src/coderag/config.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py scripts/evalfly_smoke.py
PYTHONPATH=src .venv/bin/python -m mypy src/coderag/services src/coderag/config.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_api_routes.py tests/test_ui_handlers.py scripts/evalfly_smoke.py
.venv/bin/python scripts/evalfly_smoke.py --suite smoke
.venv/bin/python -m py_compile src/coderag/services/*.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_api_routes.py tests/test_ui_handlers.py
.venv/bin/python -m build
```
