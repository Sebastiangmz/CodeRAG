# CodeRAG M4-M7 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete CodeRAG Milestones 4 through 7 without changing the product into an autonomous coding agent.

**Architecture:** Keep API, MCP, CLI, and UI as adapters over shared services. Add hybrid retrieval and context packs first, then graph metadata, then stabilize public interfaces/docs, then prove release readiness with deterministic evidence.

**Tech Stack:** Python 3.11+, FastAPI, Click, MCP/FastMCP, SQLite registry, ChromaDB behind the existing vector-store boundary, Tree-sitter/Python AST where available, pytest, ruff, mypy, EvalFly, GitHub Actions.

## Global Constraints

- CodeRAG remains a RAG/code-intelligence product; it must not edit files, create commits for users, open PRs for users, or run autonomous issue-solving loops.
- Default quickstart must not require a local generative LLM.
- Retrieval-only flows must not require cloud or local LLM credentials.
- Any cloud generation path must remain profile-gated and documented; no blanket local-first privacy claim.
- No production-ready claim is allowed without fresh tests, EvalFly evidence, branch E2E/run records where feasible, GitHub CI, and reviewer validation.
- Existing public query behavior remains compatible unless a migration is documented and tested.
- Each milestone uses its own branch, SpecSafe slice, contract file, TDD implementation, multiple atomic commits, guarded push per commit, PR, CI watch, review, merge, and local main sync.

---

## Milestone 4: Hybrid Retrieval and Context Packs

**Branch:** `feat/m4-hybrid-retrieval-context-packs`

**Slice:** `CRAG-20260630-M4-HYBRID-CONTEXT-PACKS`

**Goal:** Replace vector-only retrieval with explainable hybrid retrieval and add no-generation context packs.

**Files:**
- Create: `src/coderag/retrieval/hybrid.py`
- Create: `src/coderag/models/context.py`
- Modify: `src/coderag/retrieval/retriever.py`
- Modify: `src/coderag/services/retrieval.py`
- Modify: `src/coderag/indexing/vectorstore.py`
- Modify: `src/coderag/api/schemas.py`
- Modify: `src/coderag/api/routes.py`
- Modify: `src/coderag/mcp/handlers.py`
- Modify: `src/coderag/mcp/tools.py`
- Modify: `src/coderag/cli.py`
- Test: `tests/test_m4_hybrid_retrieval.py`
- Test: `tests/test_api_routes.py`
- Test: `tests/test_mcp_handlers.py`
- Test: `tests/test_mcp_tools.py`
- Test: `tests/test_m3_citations.py` only if query result shapes need CLI coverage reuse.
- Modify: `evals/config.json`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- `VectorStore.get_repo_chunks(repo_id: str) -> list[Chunk]`: returns indexed chunks with content from Chroma documents/metadatas.
- `HybridRetriever.retrieve(query: str, repo_id: str, top_k: int, max_tokens: int | None = None) -> list[RetrievedChunk]`: returns deduped, scored, explained chunks.
- `RetrievalService.get_context_pack(repo_id: str, query: str, top_k: int = 8, max_tokens: int = 4000) -> ContextPack`.
- MCP tool: `get_context_pack(repo_id, query, top_k=8, max_tokens=4000)`.
- CLI command: `coderag context-pack <repo_id> <query> --format json|markdown`.

**Acceptance:**
- Hybrid retrieval combines vector candidates and local lexical candidates.
- Each retrieved chunk exposes score components and `retrieval_sources` through metadata-compatible fields.
- Dedupe merges overlapping same-file ranges.
- Context pack generation invokes no LLM.
- Context pack includes query, repository id, files/snippets, citation candidates, ranking reasons, token estimate, budget, and profile/capability metadata.
- EvalFly includes retrieval/context-pack smoke assertions.

---

## Milestone 5: Code Intelligence Graph MVP

**Branch:** `feat/m5-code-intelligence-graph`

**Slice:** `CRAG-20260630-M5-CODE-GRAPH`

**Goal:** Add deterministic Python and TS/JS symbol graph with blast-radius APIs.

**Files:**
- Create: `src/coderag/graph/models.py`
- Create: `src/coderag/graph/extractor.py`
- Create: `src/coderag/services/graph.py`
- Modify: `src/coderag/services/registry.py` for graph tables or cached graph records.
- Modify: `src/coderag/api/schemas.py`
- Modify: `src/coderag/api/routes.py`
- Modify: `src/coderag/mcp/handlers.py`
- Modify: `src/coderag/mcp/tools.py`
- Modify: `src/coderag/cli.py`
- Test: `tests/test_m5_graph.py`
- Test: API/MCP/CLI tests for `find_symbol`, `find_references`, `get_blast_radius`.
- Modify: `evals/config.json`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- `GraphService.build_graph(repo_id: str) -> CodeGraph`.
- `GraphService.find_symbol(repo_id: str, name: str) -> list[SymbolRecord]`.
- `GraphService.find_references(repo_id: str, name: str) -> list[ReferenceRecord]`.
- `GraphService.get_blast_radius(repo_id: str, path: str | None = None, symbol: str | None = None) -> BlastRadius`.

**Acceptance:**
- Python fixture extracts modules, classes, functions, methods, imports, and test associations.
- TS/JS fixture extracts imports, exported functions/classes, and references with deterministic regex/Tree-sitter-safe fallback.
- Unsupported languages return explicit capability fallback, not fake graph confidence.
- Incremental cache uses file content hash.
- CLI/API/MCP expose graph operations with structured JSON.

---

## Milestone 6: Production Interfaces and Docs

**Branch:** `feat/m6-production-interfaces-docs`

**Slice:** `CRAG-20260630-M6-PRODUCTION-INTERFACES`

**Goal:** Stabilize user-facing contracts and align docs with tested behavior.

**Files:**
- Modify: `src/coderag/api/schemas.py`
- Modify: `src/coderag/api/routes.py`
- Modify: `src/coderag/mcp/tools.py`
- Modify: `src/coderag/mcp/handlers.py`
- Modify: `src/coderag/cli.py`
- Modify: `README.md`
- Create/modify docs under existing docs path only if needed: privacy/profile matrix, migration notes, troubleshooting.
- Test: contract snapshot tests for CLI/API/MCP representative outputs.
- Modify: `evals/config.json`
- Modify: `.github/workflows/ci.yml`

**Acceptance:**
- MCP tools include index/update/list/query/search_hybrid/get_context_pack/find_symbol/find_references/get_blast_radius/verify_citations where implemented.
- CLI exposes equivalent useful commands.
- REST schemas cover outputs and error shapes.
- README positions CodeRAG as verified codebase context engine, not autonomous agent.
- Docs do not claim untested production readiness or blanket privacy.

---

## Milestone 7: Production Release Candidate

**Branch:** `feat/m7-production-release-candidate`

**Slice:** `CRAG-20260630-M7-PRODUCTION-RC`

**Goal:** Prove the product works end to end and only then mark release-candidate readiness.

**Files:**
- Modify: `evals/config.json`
- Create: fresh `evals/runs/<run-id>.json` and `evals/reports/<run-id>.md` if the existing EvalFly harness writes them.
- Create: `docs/superpowers/plans/2026-06-30-m7-release-run-record.md`.
- Modify: `README.md` only for evidence badges/claims that are actually supported.
- Modify CI if a release gate is missing.

**Acceptance:**
- Clean install path is exercised from fresh venv or equivalent isolated environment.
- CI-equivalent local transcript passes.
- EvalFly smoke, retrieval, citations, graph, interfaces/security smoke cases pass or unsupported cases are explicitly excluded with evidence and user-visible wording.
- Branch E2E/run record covers CLI/API/MCP flows; UI flow is included only if Gradio dependencies are installable in the environment.
- Final reviewer validates AC-0 through AC-10.
- Release checklist in roadmap is all PASS or explicitly out of release scope.

---

## Execution Rules

- Use TDD for behavior changes: red test, expected failure, green implementation, full affected gate.
- Do not weaken existing M0-M3 tests or CI gates.
- Do not add autonomous editing behavior.
- Commit messages use English semantic style observed in repo history.
- Every commit includes `Spec-Slice: <active slice>` trailer and is pushed immediately with the guarded push wrapper.
- Remote PR/merge operations use the guarded GitHub wrapper with preview before `--i-approve`.
