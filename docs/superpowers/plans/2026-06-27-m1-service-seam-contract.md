# CodeRAG Milestone 1 Service Seam Contract

Slice: `CRAG-20260627-M1-SERVICE-SEAM`
Parent plan: `docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`
Branch: `feat/m1-service-seam`

## Scope

Milestone 1 extracts shared core services that preserve current behavior while removing duplicated repository metadata, indexing, retrieval/query, and deletion orchestration from API/MCP/CLI-facing handlers. This milestone is a seam extraction, not a retrieval-quality milestone: it must not introduce hybrid retrieval, SQLite registry, verified citation semantics, graph extraction, or autonomous editing behavior.

## Source of truth

1. Roadmap Milestone 1 (`docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`).
2. Current implementation on `main` after PR #1 merge.
3. Existing public behavior of API routes, MCP tools, and CLI commands.
4. Existing JSON repository metadata shape in `settings.data_dir / "repositories.json"` until Milestone 2 replaces it with SQLite.

## Frozen acceptance criteria

### M1-AC1: Shared JSON-backed repository registry seam

Pass requires:
- A `RepositoryRegistry` service owns loading, saving, listing, partial-id resolution, get, add/update, remove, and reloading of repository metadata.
- API routes and MCP handlers no longer each own separate `repositories` dictionaries and separate `repositories.json` load/save logic.
- The registry preserves the current JSON file format (`list[Repository.to_dict()]`) so existing alpha users are not forced into a migration in M1.

Fail if:
- API and MCP can still report conflicting repository metadata because each maintains independent JSON state code.
- Repository ID partial matching semantics disappear.
- JSON format changes without a migration/reset decision.

Required evidence:
- Unit tests for registry load/save/list/partial-id/remove behavior using a temp data directory.
- Regression tests proving API and MCP can share the same registry-backed metadata file.

### M1-AC2: Shared indexing service seam

Pass requires:
- An `IndexingService` owns repository validation/parsing, clone, file filtering, batching, embedding, vector-store writes, status transitions, error recording, commit SHA recording, and repo deletion cache cleanup where applicable.
- MCP handlers delegate full index, update, and delete operations to the service.
- API background indexing delegates to the same service path for full indexing.
- Existing response shapes for API and MCP are preserved.

Fail if:
- Full indexing logic remains duplicated in both API routes and MCP handlers.
- Error status is not persisted on failed indexing.
- Embedding resources are not unloaded/released on the full-index path.

Required evidence:
- Unit tests with fake dependencies proving full indexing status transitions and batch writes.
- Unit tests proving failed indexing persists `error` status and message.
- Unit tests or existing MCP regression tests proving handler response compatibility.

### M1-AC3: Shared retrieval/query service seam

Pass requires:
- A `RetrievalService` or `QueryService` owns readiness checks and response generation through `ResponseGenerator`.
- API query and MCP query delegate to the shared service instead of duplicating ready/not-found checks and generator orchestration.
- A search-only service path owns semantic search formatting for MCP `search_code` without invoking the LLM.
- Existing API `QueryResponse` and MCP dict shapes are preserved.

Fail if:
- API and MCP still independently implement repository readiness checks and generator construction.
- Missing repo and not-ready errors become uncaught exceptions in MCP/CLI flows.

Required evidence:
- Unit tests for query success, missing repo, and not-ready repo using fake generator/retriever dependencies.
- Existing MCP handler tests pass unchanged or with strictly seam-related fixture updates.

### M1-AC4: Provider/profile config foundation

Pass requires:
- A `ProviderConfigService` exposes the selected profile: `cloud`, `hybrid`, `private`, or `retrieval-only`.
- Invalid profiles fail configuration validation.
- Retrieval-only profile can be detected without requiring an LLM API key.
- Existing default remains low-friction cloud-oriented behavior and does not require a local generative model.

Fail if:
- Default profile implies private/local-only hardware requirements.
- Retrieval-only mode still attempts to initialize/generate with an LLM in service-level validation paths or the existing UI query path.

Required evidence:
- Config/service tests for valid profiles, invalid profile rejection, documented `.env` wiring, retrieval-only capability flags, and a static UI guard regression.

### M1-AC5: Governance and evidence

Pass requires:
- SpecSafe slice is open before implementation edits and closed only after verification.
- EvalFly smoke config is extended with M1 deterministic file/service checks, or a committed note explains why a deeper M1 EvalFly suite waits for fixture repos.
- CI remains green for the M0 gate and includes M1 unit tests in its targeted pytest step.
- Work is committed on `feat/m1-service-seam`, pushed, and opened as a PR after local verification.

Fail if:
- Tests/evals are weakened, deleted, bypassed, or relabeled to pass.
- M1 is declared production-ready or claims SQLite/hybrid/citation/graph capabilities.

Required evidence:
- Targeted pytest output for M0 + M1 tests.
- Ruff and mypy output for touched tests/scripts/service modules.
- `scripts/evalfly_smoke.py --suite smoke` output.
- Package build output.
- Post-implementation review against this contract.

## Implementation boundaries

Allowed:
- Add `src/coderag/services/` modules.
- Refactor API and MCP handlers to delegate to services.
- Update CLI only where it consumes MCP handler behavior.
- Keep JSON registry as the M1 backing store.
- Keep ChromaDB and current vector-only retrieval behind the new service seam.

Not allowed in M1:
- SQLite registry schema or migrations.
- Hybrid/BM25 retrieval.
- Verified citation semantics or `grounded=True` behavior changes.
- Code graph extraction.
- New autonomous agent behavior.

## Evidence commands

Run before PR:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_m0_baseline.py tests/test_m1_services.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py
.venv/bin/python -m ruff check src/coderag/services src/coderag/config.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/ui/repository_metadata.py tests/test_m1_services.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py scripts/evalfly_smoke.py
PYTHONPATH=src .venv/bin/python -m mypy src/coderag/services src/coderag/config.py src/coderag/ui/repository_metadata.py tests/test_m1_services.py tests/test_api_routes.py scripts/evalfly_smoke.py
.venv/bin/python scripts/evalfly_smoke.py --suite smoke
.venv/bin/python -m py_compile src/coderag/services/*.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m1_services.py tests/test_api_routes.py tests/test_ui_handlers.py
.venv/bin/python -m build
```
