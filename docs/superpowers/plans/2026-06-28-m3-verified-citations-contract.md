# Milestone 3 Verified Citation Engine Contract

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or inline TDD execution. Do not implement runtime code before observing failing M3 citation tests.

**Slice:** `CRAG-20260628-M3-VERIFIED-CITATIONS`  
**Branch:** `feat/m3-verified-citations`  
**Source of truth:** `docs/superpowers/plans/2026-06-27-coderag-2-production-roadmap.md`, Milestone 3 lines 505-529, plus current `main` after M2 merge.

## Goal

Make CodeRAG's `grounded` claim depend on deterministic citation verification instead of citation string presence.

## Scope

M3 may add or modify:

- citation verification models/helpers under `src/coderag/generation/` or `src/coderag/services/`;
- response model fields needed to expose verification status while preserving existing `answer`, `citations`, `retrieved_chunks`, `grounded`, and `query_id` fields;
- `ResponseGenerator` citation handling;
- `RetrievalService`, API, MCP, CLI, and UI adapters only as needed to expose verification status;
- registry/file/chunk metadata reads only as needed for citation verification;
- EvalFly deterministic smoke coverage and citation-focused tests.

M3 must not add:

- hybrid/BM25 retrieval;
- context pack APIs;
- code graph extraction;
- autonomous editing behavior;
- broad provider or model changes;
- production-ready claims.

## Definitions

A citation is **verified** only if all are true:

1. the parsed citation path matches a retrieved chunk path;
2. the citation line range is valid (`start_line >= 1`, `end_line >= start_line`);
3. the citation line range overlaps at least one retrieved chunk from the same file;
4. the cited chunk belongs to the current query's retrieved evidence set.

A citation is **unverified** if parsing succeeds but any verification condition fails.

A response is `grounded=True` only when:

- at least one citation is present;
- at least one citation is verified;
- no parsed citation is unverified.

No-citation responses remain `grounded=False`.

## Acceptance Criteria

### M3-AC1: Citation verifier

- Add a deterministic verifier that accepts parsed `Citation` objects and current `RetrievedChunk` evidence.
- It returns per-citation verification status and reason.
- It detects hallucinated paths, non-overlapping line ranges, invalid ranges, and stale/non-retrieved citations.

### M3-AC2: Grounded semantics

- `ResponseGenerator.generate()` must compute `grounded` from verifier output, not from `len(citations) > 0 and len(chunks) > 0`.
- A response with any unverified parsed citation is not grounded.
- A response with verified citations is grounded.
- A response with no citations remains not grounded.

### M3-AC3: Adapter exposure

- REST `QueryResponse` exposes citation verification status without removing existing fields.
- MCP query result exposes citation verification status without removing existing keys.
- CLI/UI may display a basic verified/unverified status, but must not invent new behavior beyond M3.

### M3-AC4: EvalFly and tests

- Add citation verifier unit tests with negative cases for:
  - hallucinated path;
  - out-of-range/non-overlapping lines;
  - invalid line range;
  - mixed verified and unverified citations.
- Add generator/service tests proving `grounded` follows verifier results.
- Extend committed EvalFly smoke with deterministic citation-engine checks.

### M3-AC5: Governance

- Keep all commits on `feat/m3-verified-citations` with `Spec-Slice: CRAG-20260628-M3-VERIFIED-CITATIONS` trailers.
- Push after every commit using the guarded push wrapper.
- Open PR with guarded GitHub wrapper only after local verification and review.

## Required Evidence

Before PR:

```bash
bun run /Users/sebastian/.omp/omp-pantheon/skills/specsafe/bin/specsafe.ts status
PYTHONPATH=src .venv/bin/python -m pytest tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_ui_handlers.py
.venv/bin/python -m ruff check src/coderag/generation src/coderag/models src/coderag/services src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/ui/handlers.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_ui_handlers.py scripts/evalfly_smoke.py
PYTHONPATH=src .venv/bin/python -m mypy src/coderag/generation src/coderag/models src/coderag/services src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/ui/handlers.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_ui_handlers.py scripts/evalfly_smoke.py
.venv/bin/python scripts/evalfly_smoke.py --suite smoke
.venv/bin/python -m py_compile src/coderag/generation/*.py src/coderag/models/*.py src/coderag/services/*.py src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/ui/handlers.py tests/test_citations.py tests/test_m3_citations.py
.venv/bin/python -m build
```

## Non-goals

- Do not require reading the live filesystem during query verification. M3 verifies citations against retrieved evidence, not against mutable working tree state.
- Do not claim stale working-tree detection unless file snapshot hashes are actually checked in the query path.
- Do not change retrieval ranking or chunking behavior.
