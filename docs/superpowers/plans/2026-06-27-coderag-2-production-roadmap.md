# CodeRAG 2.0 Production Roadmap and Acceptance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Do not implement any task until the acceptance criteria and evidence requirements in this document are copied into the active SpecSafe slice and EvalFly contract.

**Goal:** Evolve CodeRAG from a repository Q&A RAG demo into a production-ready, provider-flexible codebase context engine with hybrid retrieval, verified citations, code intelligence, MCP/API/CLI interfaces, reproducible evals, and safe deployment defaults.

**Architecture:** CodeRAG remains a RAG/code-intelligence tool, not an autonomous coding agent. The core product indexes repositories, builds searchable code context, verifies file/line evidence, and exposes structured context to humans, IDEs, CI, and optional external coding agents. Cloud inference is the low-friction default; local/private execution is an optional profile, not a hard requirement.

**Tech Stack:** Python 3.11+, FastAPI, Click CLI, MCP/FastMCP, SQLite registry, ChromaDB initially behind a vector-store interface, Tree-sitter, SentenceTransformers or provider embeddings behind adapters, pytest, ruff, mypy, EvalFly, GitHub Actions.

## Global Constraints

- CodeRAG must remain a RAG/code-intelligence product; it must not become an autonomous editing agent in this roadmap.
- Default user path must work on ordinary laptops without requiring a local generative LLM.
- Cloud/model providers must be configurable; local/private profiles must be optional and documented.
- Repository indexing, metadata, citation validation, graph metadata, and retrieval orchestration must run locally by default.
- Any code snippet sent to a cloud provider must be explicit in the selected profile and documented in the privacy model.
- No production-ready claim is allowed without fresh tests, EvalFly evidence, and reviewer validation against this contract.
- Existing public interfaces must either remain compatible or have migration notes and regression tests.
- Security defaults must not expose a credentialed local API to arbitrary origins.
- Evidence must be deterministic first; LLM-as-judge is advisory only unless explicitly approved later.

---

## Source of Truth

1. User request in this session: complete and refine the adjusted roadmap as a disciplined engineering/PM plan, with acceptance criteria and evidence defined before implementation.
2. Current repository state observed in `/Users/sebastian/Documents/CodeRAG`.
3. Prior analysis of current CodeRAG architecture and gaps.
4. Current-market reference points gathered from public repositories/docs: ChunkHound, Claude Context, code-review-graph, Mantic.sh, Aider, mini-SWE-agent, OpenHands Agent Canvas, Repomix, and MCP docs.

## Current Repository Baseline

Observed facts from the current clone:

- Python package `code-rag-me` is version `0.1.2` in `pyproject.toml`; `src/coderag/__init__.py` and `configs/default.yaml` still report `0.1.0`.
- Stack: FastAPI REST API, Gradio UI, Click CLI, MCP server, ChromaDB, SentenceTransformers, GitPython, Tree-sitter Python.
- Structural chunking exists only for Python; all other included languages fall back to text chunks.
- Retrieval is vector-only: query embedding -> Chroma top-k -> prompt.
- Citations are parsed from LLM output but not verified against retrieved chunks, file snapshots, or line content.
- API and MCP share repository metadata through global in-memory state plus `repositories.json`.
- Server config allows `0.0.0.0`; FastAPI CORS is `allow_origins=["*"]` with `allow_credentials=True`.
- CI only builds/publishes package on release/workflow dispatch; it does not run tests, lint, typecheck, or evals.
- Verification observed: `python3 -m compileall src tests scripts` passed; `python3 -m pytest` did not run because global Python lacks pytest.
- Current repo became dirty from generated `__pycache__/` artifacts during compile verification; implementation tasks must clean generated artifacts before commits.

## Product Positioning

### Corrected Positioning

**CodeRAG 2.0: verified codebase context engine.**

One-liner:

> CodeRAG retrieves minimal, source-cited, verifiable context from codebases for humans, IDEs, CI, and AI coding tools.

### Non-Goals

- CodeRAG will not edit files, create commits, open PRs, run autonomous issue-solving loops, or compete directly with Claude Code/Codex/Cursor/OpenHands/Aider.
- CodeRAG will not require users to run a local generative LLM.
- CodeRAG will not claim full privacy when cloud providers are selected.
- CodeRAG will not mark citation-formatted answers as grounded unless citations are verified.

### Primary Personas

1. **Developer asking questions locally:** wants trustworthy file/line-cited answers and context.
2. **AI coding tool user:** wants Claude Code/Codex/Cursor/etc. to read the right files before acting.
3. **Reviewer/maintainer:** wants changed-file context, blast radius, and suggested tests.
4. **PM/engineering lead:** wants demonstrable quality through eval reports, CI, and acceptance gates.

---

## Frozen Acceptance Criteria

### AC-0: Delivery Governance

**Pass requires**

- A SpecSafe slice is open before implementation starts and references this plan.
- Acceptance criteria and required evidence are frozen before each implementation milestone.
- EvalFly is initialized before behavior-changing implementation begins.
- Every milestone has deterministic tests and EvalFly evidence or an explicit documented reason why EvalFly does not apply.
- No eval/test is weakened, deleted, bypassed, or relabeled to make a failing implementation pass.

**Fail if**

- Implementation starts without a slice or without criteria/evidence captured.
- A milestone is declared complete based only on code review or successful import.
- An eval/test is removed instead of fixing the underlying behavior.

**Required evidence**

- SpecSafe status showing the active slice.
- `evalfly validate` report once EvalFly exists.
- Milestone-specific verification logs.

### AC-1: Production Baseline and Maintainability

**Pass requires**

- Project installs reproducibly with a documented command using `uv` or a committed equivalent lock strategy.
- Version is single-sourced or synchronized across package, runtime, and config surfaces.
- CI runs tests, lint, typecheck or documented baseline, package build, and EvalFly smoke suite.
- Generated artifacts are ignored and absent from commits.
- API, CLI, MCP, and UI use shared services instead of duplicated indexing/query logic.

**Fail if**

- CI only publishes package but does not validate behavior.
- Handlers continue to own duplicated core indexing/retrieval behavior.
- Version mismatch remains visible to users or package consumers.

**Required evidence**

- CI workflow file review.
- Local `uv run pytest`, `uv run ruff check .`, `uv run mypy src` or typed-baseline report, and `python -m build`.
- Unit tests for extracted services.

### AC-2: Safe Defaults and Privacy Profiles

**Pass requires**

- Default server binds to `127.0.0.1` unless explicitly configured otherwise.
- CORS defaults to local trusted origins; wildcard plus credentials is not the default.
- Profiles are documented and enforced: `cloud`, `hybrid`, `private`, and `retrieval-only`.
- Cloud profile is the default for low-friction use and clearly states what code context may leave the machine.
- Private profile uses local embeddings and local generation adapters when configured.
- Retrieval-only mode requires no generative model.
- API keys are not logged, emitted in diagnostics, or stored in repo-local files.

**Fail if**

- A fresh `coderag serve` exposes credentialed API access broadly.
- Local generative model is required for the default quickstart.
- Privacy claims imply no data egress while cloud generation is configured.

**Required evidence**

- Config tests for default host/CORS/profile behavior.
- CLI doctor tests proving secrets are redacted.
- README/privacy model review.
- Negative tests for disallowed CORS defaults.

### AC-3: Repository Registry and Job State

**Pass requires**

- Repository metadata moves from global dict plus `repositories.json` to SQLite-backed registry with explicit schema and migrations.
- Registry tracks repo id, URL/path, branch, head SHA, indexed_at, status, error, files, chunks, and model/index metadata.
- Index jobs have durable status and error reporting.
- API/MCP/UI access the same registry abstraction.
- Existing `repositories.json` users have a migration path or a documented reset path for pre-1.0 alpha.

**Fail if**

- Concurrent API/MCP operations can corrupt repository metadata.
- Index state is lost on process restart except for vector DB contents.
- Different interfaces report conflicting repo state.

**Required evidence**

- Registry unit tests using temp SQLite DB.
- Migration tests from sample `repositories.json` or explicit alpha reset behavior test.
- API/MCP integration tests proving shared state.

### AC-4: Hybrid Retrieval Foundation

**Pass requires**

- Retrieval supports semantic vector search, lexical/BM25 or equivalent local text search, path/filename/symbol boosting, score normalization, dedupe, and deterministic context budgeting.
- Vector store is behind an interface so ChromaDB can remain initial backend without coupling the architecture.
- Query results include source, score components, token estimates, and citation candidates.
- Existing `query_code` behavior remains available or has migration tests.

**Fail if**

- Retrieval remains vector-only after this milestone.
- Search result order cannot be explained or debugged.
- Duplicated chunks consume context budget without dedupe.

**Required evidence**

- Retrieval fixture repo with known expected files/symbols.
- EvalFly retrieval suite comparing vector-only baseline vs hybrid.
- Unit tests for score merging, dedupe, and budget caps.

### AC-5: Verified Citations and Grounded Answers

**Pass requires**

- Citation verifier validates file exists in indexed snapshot, line range exists, range intersects retrieved evidence, and cited content hash matches indexed snapshot.
- `grounded=True` is emitted only when citations are verified.
- Invalid, unsupported, or hallucinated citations are reported separately and degrade answer confidence.
- Answer schema separates answer, retrieved evidence, verified citations, invalid citations, and unsupported claims where implemented.
- Existing citation parser tests are preserved and expanded.

**Fail if**

- Any answer is called grounded solely because the LLM emitted `[file:start-end]`.
- Citations to non-retrieved files are silently accepted.
- Stale line numbers pass verification after file changes.

**Required evidence**

- Deterministic tests for valid citations, nonexistent file, out-of-range line, stale hash, non-retrieved citation, and malformed citation.
- EvalFly citation-validity suite.
- Negative tests with hallucinated citations.

### AC-6: Code Intelligence Graph

**Pass requires**

- CodeRAG extracts symbols and edges for at least Python and TypeScript/JavaScript fixtures.
- Stored graph supports definitions, references or imports, basic call/import edges where feasible, file-to-symbol mapping, and test-file association heuristics.
- CLI/MCP/API expose `find_symbol`, `find_references`, and `get_blast_radius` or equivalent names.
- Graph extraction is incremental by file hash.
- Unsupported languages degrade to text search with explicit capability flags.

**Fail if**

- Tree-sitter remains Python-only while docs claim multi-language code intelligence.
- Blast radius is produced without explaining graph evidence.
- Graph rebuild requires full reindex for every small file change.

**Required evidence**

- Fixture repos for Python and TS/JS with expected symbol and edge snapshots.
- Unit tests for parser outputs.
- EvalFly graph suite for symbol lookup and blast-radius expected sets.
- Negative tests for unsupported language fallback.

### AC-7: Context Packs for Humans and AI Tools

**Pass requires**

- CodeRAG can return structured context packs without invoking a generative LLM.
- Context pack includes task/query, files, symbols, snippets, verified citation candidates, score/ranking reasons, token count estimate, and capability/profile metadata.
- MCP exposes context-pack retrieval for external tools.
- CLI can emit JSON and human-readable markdown.
- Context packs enforce max token/character budgets deterministically.

**Fail if**

- Context pack generation requires cloud or local LLM.
- Output lacks enough provenance for a reviewer to trust why files were selected.
- Budget settings are ignored.

**Required evidence**

- Snapshot tests for JSON schema and markdown rendering.
- MCP tool tests.
- EvalFly context-pack suite measuring expected relevant files and token budget compliance.

### AC-8: MCP/API/CLI Production Interfaces

**Pass requires**

- MCP tools are stable, documented, and covered by tests: index, update, list, query, search_hybrid, get_context_pack, find_symbol, find_references, get_blast_radius, verify_citations.
- CLI exposes equivalent commands where useful.
- REST API has schemas for core operations and error responses.
- All interfaces use shared core services.
- Error responses distinguish validation, missing repo, not indexed, provider failure, and internal failure.

**Fail if**

- MCP output is unstructured text where structured data is required.
- CLI/API/MCP disagree on result shape or state.
- Provider errors expose secrets or raw stack traces to normal users.

**Required evidence**

- MCP handler/tool tests.
- API route tests through FastAPI test client.
- CLI tests through Click runner.
- Contract snapshots for representative outputs.

### AC-9: EvalFly and Quality Gates

**Pass requires**

- `evals/` exists with config, cases, reports/runs directories, and deterministic smoke suite.
- EvalFly validates and runs locally.
- CI invokes EvalFly check at least for smoke suite before release or merge-readiness claims.
- Reports are cited in milestone completion notes.
- Eval cases cover retrieval, citation verification, context pack budget, graph lookup, and MCP contract smoke.

**Fail if**

- Evals exist but are not connected to acceptance criteria.
- Evals require private credentials or unsanitized traces.
- Production-ready claim is made without a passing EvalFly report.

**Required evidence**

- `bun run ~/.omp/agent/skills/evalfly/bin/evalfly.ts validate`.
- `bun run ~/.omp/agent/skills/evalfly/bin/evalfly.ts check --suite smoke --commit-range <range>`.
- Saved `evals/reports/<run-id>.md` and `evals/runs/<run-id>.json`.

### AC-10: Production Release Readiness

**Pass requires**

- Fresh install path works from clean environment.
- Default quickstart indexes a small public repo or local fixture, answers/retrieves context, and verifies citations.
- Cloud profile works without local LLM hardware.
- Retrieval-only mode works without LLM key.
- Private/local profile is documented as optional and passes smoke on supported hardware/runtime where available.
- Docs match actual commands and behavior.
- Security/privacy defaults pass tests.
- Reviewer validates against original contract.

**Fail if**

- Production release requires undocumented credentials, hardware, or manual DB repair.
- README claims features not covered by tests/evals.
- Branch E2E is skipped for user-visible CLI/API/UI behavior.

**Required evidence**

- Clean-environment install log.
- Branch E2E run record for CLI/API/UI/MCP user flows.
- Passing CI.
- Passing EvalFly report.
- Reviewer/validator output.

---

## Required Evidence Plan

### Targeted Tests

- Unit tests for config profiles, CORS/host defaults, registry schema, migration/reset behavior, vector-store interface, hybrid ranking, dedupe, context budgeting, citation verifier, parser graph extraction, and output schemas.
- Integration tests for API, MCP, CLI using fixture repos and temp storage.
- Contract snapshot tests for context pack JSON, MCP outputs, and human-readable CLI output.
- Negative tests for invalid repo URL, unsupported language fallback, provider failure, missing API key, invalid citation, stale citation, and secret redaction.

### EvalFly Suites

Create these deterministic suites:

1. `smoke`
   - validates one fixture repo can index;
   - retrieves expected context for one query;
   - verifies at least one valid citation;
   - checks MCP tool schema for context pack;
   - confirms retrieval-only mode does not require LLM key.

2. `retrieval`
   - known query -> expected relevant files/symbols in top-k;
   - vector-only baseline recorded;
   - hybrid must improve or match recall while staying within budget.

3. `citations`
   - valid file/range/hash passes;
   - nonexistent file fails;
   - out-of-range line fails;
   - non-retrieved citation fails;
   - stale snapshot hash fails.

4. `graph`
   - Python fixture symbol extraction;
   - TS/JS fixture symbol extraction;
   - imports/references/calls where implemented;
   - blast radius includes expected dependent file and suggested test file.

5. `interfaces`
   - CLI output JSON schema;
   - MCP tool output schema;
   - FastAPI response schema;
   - errors are typed and secret-redacted.

6. `security`
   - default host is loopback;
   - wildcard credentialed CORS is not default;
   - `.env` and secret-like files are excluded;
   - diagnostics redact configured API keys.

### Branch E2E Evidence

Use `agentic-branch-e2e` after implementation for user-visible behavior. Required run records:

1. **CLI flow**
   - create clean temp environment;
   - install package;
   - run setup with safe test profile or retrieval-only profile;
   - index fixture repo;
   - run `coderag context` and `coderag query` where applicable;
   - verify JSON/markdown output and citation validity;
   - negative case: missing provider key in cloud generation path fails closed with actionable error.

2. **API flow**
   - start real FastAPI server with local test config;
   - call health;
   - index fixture repo;
   - poll job;
   - retrieve context;
   - verify citations;
   - negative case: hostile origin/CORS behavior and invalid repo id.

3. **MCP flow**
   - run MCP server through stdio or supported test client;
   - call list/index/search/context tools;
   - validate structured output;
   - negative case: unknown repo and invalid citation verification request.

4. **UI flow** if Gradio remains supported in release scope
   - start real app;
   - drive browser-visible index/query path;
   - capture screenshot/logs/network;
   - negative case: query before repo ready.

### Logs, Reports, and Reviewer Evidence

- Startup logs for API/MCP/CLI E2E runs.
- EvalFly report paths and run ids.
- CI run URL or local CI-equivalent transcript.
- Validator/reviewer output against this plan, not just diff.
- Sanitized traces only if trace evidence is used; raw traces must remain under `.pi/evalfly/raw/` and not be committed.

---

## Milestone Roadmap

### Milestone 0: Governance and Baseline Stabilization

**Purpose:** Make the project safe to change.

**Scope**

- Open SpecSafe slice for implementation.
- Initialize EvalFly with deterministic smoke skeleton.
- Adopt reproducible dev commands using `uv` or equivalent.
- Add CI for tests/lint/typecheck/build/eval smoke.
- Remove generated `__pycache__/` artifacts and ensure they stay ignored.
- Synchronize version fields.
- Document current alpha migration/reset policy.

**PM go/no-go gate**

- Go only if a new contributor can clone, install, run tests, and see smoke evidence without manual package guessing.

**Acceptance criteria covered**

- AC-0, AC-1, AC-9 foundation.

**Required verification**

- `uv run pytest`.
- `uv run ruff check .`.
- `uv run mypy src` or a committed typed-baseline decision with issue list.
- `python -m build`.
- `evalfly validate` and smoke placeholder validation.

### Milestone 1: Core Service Extraction and Safe Config

**Purpose:** Eliminate duplicated logic and fix unsafe defaults before new features.

**Scope**

- Create shared services: `RepositoryRegistry`, `IndexingService`, `RetrievalService`, `CitationService`, `ProviderConfigService`.
- API/MCP/CLI/UI call services instead of owning pipelines.
- Default host loopback and safe CORS.
- Add privacy/profile model: `cloud`, `hybrid`, `private`, `retrieval-only`.
- Ensure default quickstart does not require local LLM.

**PM go/no-go gate**

- Go only if old user flows still work through the refactored services and unsafe defaults are corrected.

**Acceptance criteria covered**

- AC-1, AC-2, AC-8 partial.

**Required verification**

- Unit tests for profiles and safe config.
- API/MCP/CLI regression tests.
- Branch E2E smoke for CLI/API if behavior changes are visible.

### Milestone 2: SQLite Registry and Durable Job State

**Purpose:** Replace fragile global state and JSON metadata.

**Scope**

- SQLite registry schema for repos, files, chunks metadata, jobs, provider/index metadata.
- Migration or alpha reset from `repositories.json`.
- Job state for indexing/update/delete.
- Shared registry across API/MCP/CLI/UI.

**PM go/no-go gate**

- Go only if process restart preserves accurate repo/job state and interfaces agree.

**Acceptance criteria covered**

- AC-3.

**Required verification**

- Registry unit/integration tests.
- Restart simulation test.
- API/MCP state consistency test.

### Milestone 3: Verified Citation Engine

**Purpose:** Make “verifiable citations” true.

**Scope**

- Store file snapshot hashes and line metadata.
- Implement citation verifier.
- Change grounded semantics to require verified citations.
- Expose citation verification through CLI/API/MCP.
- Expand citation tests and EvalFly citation suite.

**PM go/no-go gate**

- Go only if hallucinated or stale citations fail deterministically.

**Acceptance criteria covered**

- AC-5.

**Required verification**

- Citation unit tests.
- EvalFly `citations` suite.
- Negative cases for hallucinated/stale/non-retrieved citations.

### Milestone 4: Hybrid Retrieval and Context Packs

**Purpose:** Modernize beyond vector-only RAG and enable no-generation workflows.

**Scope**

- Add lexical retrieval.
- Add path/name/symbol metadata boosting where available.
- Add vector-store abstraction.
- Add result merge/dedupe/ranking explanation.
- Add context budgeter.
- Add `ContextPack` schema and CLI/API/MCP outputs.
- Support retrieval-only mode.

**PM go/no-go gate**

- Go only if context packs are useful without any generative LLM and beat or match vector-only baseline on eval fixtures.

**Acceptance criteria covered**

- AC-4, AC-7.

**Required verification**

- Hybrid retrieval unit tests.
- Context pack snapshot tests.
- EvalFly `retrieval` and `context-pack` cases.
- Branch E2E CLI/API context-pack flow.

### Milestone 5: Code Intelligence Graph MVP

**Purpose:** Add code-structure awareness.

**Scope**

- Multi-language parser MVP: Python and TypeScript/JavaScript.
- Symbol table and file-to-symbol mapping.
- Import/reference/call edges where feasible.
- Test-file association heuristics.
- Incremental graph update by file hash.
- CLI/API/MCP: `find_symbol`, `find_references`, `get_blast_radius`.

**PM go/no-go gate**

- Go only if graph outputs are deterministic on fixtures and unsupported languages explicitly degrade.

**Acceptance criteria covered**

- AC-6.

**Required verification**

- Parser fixture tests.
- Graph snapshot tests.
- EvalFly `graph` suite.
- Negative fallback tests.

### Milestone 6: Production Interfaces and Docs

**Purpose:** Make the product usable and supportable.

**Scope**

- Stabilize CLI commands and MCP tool contracts.
- Add OpenAPI schemas and examples.
- Rewrite README around corrected positioning.
- Add privacy/profile docs.
- Add migration docs.
- Add troubleshooting docs for provider keys, local/private profiles, and retrieval-only mode.

**PM go/no-go gate**

- Go only if docs match commands proven by E2E and do not claim unsupported features.

**Acceptance criteria covered**

- AC-8, AC-10 docs portion.

**Required verification**

- CLI/API/MCP contract tests.
- README command smoke tests where feasible.
- Reviewer docs audit.

### Milestone 7: Production Release Candidate

**Purpose:** Prove the whole product works end-to-end.

**Scope**

- Clean install from source and package build.
- Full targeted tests.
- EvalFly smoke/retrieval/citations/graph/interfaces/security.
- Branch E2E CLI/API/MCP/UI where supported.
- Reviewer validation against this contract.
- Release checklist and rollback notes.

**PM go/no-go gate**

- Go only if every AC is PASS or explicitly removed from release scope by user approval before implementation; no silent scope shrink.

**Acceptance criteria covered**

- AC-0 through AC-10.

**Required verification**

- Passing CI-equivalent transcript.
- EvalFly passing report paths.
- Branch E2E run record(s).
- Validator/reviewer output.

---

## PM Gap Register

| ID | Gap | Severity | Risk if ignored | Resolution before development |
|---|---|---:|---|---|
| PM-1 | Release persona priority not locked | High | Features optimize for wrong user | Prioritize developer + external tool context workflows over autonomous agents |
| PM-2 | Privacy wording can overpromise | High | Trust/reputation risk | Use profile-specific privacy matrix; no blanket local-first claim |
| PM-3 | Local model expectation unclear | High | Adoption blocked by hardware assumptions | Cloud profile default; local/private optional |
| PM-4 | Existing PyPI users migration unclear | Medium | Upgrade confusion | Decide alpha reset vs JSON migration in Milestone 2 |
| PM-5 | UI importance unclear | Medium | Gradio absorbs time with low strategic value | Treat UI as secondary unless explicitly required in release scope |
| PM-6 | Supported languages could sprawl | High | Graph MVP never finishes | Commit to Python + TS/JS first; fallback for others |
| PM-7 | Eval success metrics not public | High | CV/project claims look ungrounded | Publish retrieval/citation/token-budget metrics in README after evals pass |
| PM-8 | Provider costs and data egress unpriced | Medium | User surprise | Add provider matrix with cost/privacy tradeoffs |
| PM-9 | MCP client support scope can balloon | Medium | Installation complexity | Support generic MCP stdio first; document Claude/Codex/Cursor examples after tests |
| PM-10 | Production readiness conflated with feature completeness | Critical | Premature release | Use AC-10 gate and reviewer validation before release claim |

## Technical Decision Log Required Before Milestone 1

1. Dev environment manager: use `uv` unless a conflicting project constraint appears.
2. SQLite migration policy: support JSON migration or declare pre-1.0 reset.
3. Vector abstraction: keep ChromaDB backend initially; define interface before adding alternatives.
4. Lexical backend: choose BM25 implementation or SQLite FTS/ripgrep-backed index.
5. Parser package strategy: choose explicit Tree-sitter grammars vs `tree-sitter-language-pack`.
6. Provider abstraction: choose direct adapters vs LiteLLM. Default recommendation: provider adapter boundary now, LiteLLM evaluation before broad provider expansion.
7. Public API stability level: mark 2.0 pre-release contracts as beta until contract tests stabilize.

## Out-of-Scope for First Production Release

- Autonomous code editing.
- Pull request creation.
- Multi-agent orchestration.
- Hosted SaaS deployment.
- Full 30+ language graph support.
- LLM judge as a required gate.
- Enterprise auth/multi-user RBAC.
- Performance guarantees for very large monorepos beyond documented fixture benchmarks.

## Release Readiness Checklist

- [ ] SpecSafe slice references this plan.
- [ ] EvalFly initialized and validated.
- [ ] CI runs tests/lint/typecheck/build/eval smoke.
- [ ] Default quickstart requires no local generative LLM.
- [ ] Retrieval-only flow works without LLM credentials.
- [ ] Cloud profile documents code-context egress.
- [ ] Citation verifier blocks invalid/stale/non-retrieved citations.
- [ ] Hybrid retrieval eval passes against vector-only baseline.
- [ ] Graph MVP works for Python + TS/JS fixtures.
- [ ] MCP/API/CLI output contracts are tested.
- [ ] Branch E2E run records exist for visible flows.
- [ ] Reviewer validates implementation against AC-0 through AC-10.
- [ ] README claims match tested behavior.

## Final Planning Verdict

This plan is ready to use as a PM/engineering contract for implementation planning, but the product itself is not production-ready yet. The correct next action is to begin Milestone 0 in a fresh implementation slice, initialize EvalFly, and implement only against the frozen acceptance criteria above.

---

## Parallel Readiness Review Integration

Three specialist reviews were run against this plan before final delivery:

- `PMReadinessReview`: product/delivery readiness.
- `ArchitectureReadinessReview`: production architecture readiness.
- `EvidencePlanReview`: EvalFly/evidence readiness.

Their findings are incorporated into the gates above. The non-negotiable deltas from those reviews are:

1. **Fase 0 is not optional.** Before feature-heavy work, freeze the product contract: ICP, top workflows, no-goals, North Star metric, and beta scope.
2. **Verified citations are the product trust core.** Citation verification must land before graph-heavy or UI-heavy work.
3. **Service seam precedes feature work.** API, MCP, CLI, and UI must become adapters over shared services before hybrid retrieval/graph work.
4. **SQLite registry must model snapshots, jobs, chunks, and schema versions.** It cannot be a shallow replacement for `repositories.json`.
5. **Source snapshots and stable chunk identity are prerequisites.** Citation verification, hybrid retrieval, graph, and incremental indexing depend on them.
6. **Hybrid retrieval must be measured against vector-only baseline.** Use retrieval traces and EvalFly reports; do not claim improvement from architecture alone.
7. **Graph MVP must be scoped.** Python + TS/JS is the maximum first production scope unless fixtures/evals prove more languages.
8. **Security/cloud readiness is a first-class milestone.** CORS, auth, quotas, clone restrictions, secret redaction, and storage state cannot be left to final cleanup.
9. **MCP-first does not mean agentic autonomy.** CodeRAG exposes context/evidence tools; it does not edit, execute, branch, commit, or open PRs.
10. **Production-ready means evidence-complete.** The final release gate requires CI, EvalFly reports, branch E2E run records, security checks, and reviewer validation.

## PM Approval Packet Required Before Development

A project manager can approve Milestone 0 only after this packet exists:

- ICP: primary user is explicitly chosen.
- Top 3 workflows are ranked.
- Non-goals are signed off: no autonomous editing, no mandatory local LLM, no unsupported privacy claims.
- North Star metric is chosen.
- Beta scope is chosen: recommended default is public/local repos first, private/cloud multi-tenant only after security gates.
- Language support matrix is chosen.
- Production target is chosen: package/CLI, self-hosted API, cloud service, or staged combination.
- Cost/privacy matrix for provider profiles is published.
- Owner is assigned for each go/no-go gate.

Recommended default PM contract:

- **Primary user:** developer or AI-engineering practitioner who needs trustworthy codebase context for human use and external coding tools.
- **Top workflows:** `index repo -> get context pack`, `ask question -> verified citations`, `changed files -> blast radius/test suggestions`.
- **North Star:** percentage of real queries that return useful context with 100% verified citations within the configured token budget.
- **Beta:** CLI/MCP/API over local/public repos first; Gradio UI secondary; cloud/private repos only after security/storage hardening.

## Additional Technical Gates from Architecture Review

Before each milestone can close, the implementation must prove:

- `RepositoryRegistry` is the single source of truth for repo state.
- `SourceSnapshot` can reconstruct cited lines by snapshot id without trusting mutable working-tree `HEAD`.
- Chunk IDs are stable and include file path/range/content hash/chunker version or equivalent deterministic identity.
- Retrieval returns traceable scores per stage: vector, lexical, graph expansion, fusion, rerank if present.
- Provider adapters return usage, latency, provider/model, finish reason, and retryability; Anthropic native is not treated as OpenAI-compatible unless routed through an OpenAI-compatible gateway.
- Cloud/default Docker path does not require NVIDIA GPU.
- API/MCP/CLI parity is tested with one deterministic provider or generation stub.

## Additional EvalFly Gates from Evidence Review

The minimum blocking `smoke` suite must include:

- exact-symbol retrieval fixture;
- behavioral multi-file retrieval fixture;
- no-evidence/abstention fixture;
- valid citation fixture;
- hallucinated citation rejection fixture;
- context-pack schema and budget fixture;
- Python graph extraction fixture;
- stale graph edge removal fixture once graph exists;
- repository ingestion security boundary fixture;
- cloud config/CORS fixture;
- MCP query contract fixture;
- CLI/API/MCP parity fixture.

Global production verdict rules:

- **PASS** requires deterministic smoke + regression suites, 100% valid citations for grounded fixture answers, context-pack schema validity, graph claims backed by fixtures, security gates, MCP E2E, branch E2E, CI, and human readiness review.
- **FAIL** if a hallucinated/stale citation can be marked grounded, security fails open, or CI/EvalFly gates are absent.
- **INCONCLUSIVE** if any claimed capability lacks fixture/golden labels, relies only on mocks that bypass real ingestion/retrieval interfaces, or is judged only by LLM/human review when deterministic evidence is possible.
