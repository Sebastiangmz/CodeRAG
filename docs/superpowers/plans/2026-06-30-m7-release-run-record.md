# M7 Release Candidate Run Record

## Verdict

PASS for release-candidate evidence. This is not a hosted/SaaS production-readiness claim.

## Scope

Branch: `feat/m7-production-release-candidate`

Slice: `CRAG-20260630-M7-PRODUCTION-RC`

Goal: prove the CodeRAG M0-M6 product surfaces work as a release candidate and preserve evidence in committed artifacts.

## Evidence transcript

| Gate | Command / scenario | Result |
| --- | --- | --- |
| Build wheel | `.venv/bin/python -m build` | Built `code_rag_me-0.1.2.tar.gz` and `code_rag_me-0.1.2-py3-none-any.whl`. |
| Clean install | `python3 -m venv /tmp/coderag-m7-venv && /tmp/coderag-m7-venv/bin/python -m pip install dist/code_rag_me-0.1.2-py3-none-any.whl` | Installed wheel and dependencies in isolated venv. |
| CLI smoke | `/tmp/coderag-m7-venv/bin/coderag --help` | Listed `query`, `search-hybrid`, `context-pack`, `find-symbol`, `find-references`, `blast-radius`, and repository commands. |
| Package import | `/tmp/coderag-m7-venv/bin/python -c "import coderag; print(coderag.__version__)"` | Printed `0.1.2`. |
| Targeted tests | `PYTHONPATH=src .venv/bin/python -m pytest tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py tests/test_m4_hybrid_retrieval.py tests/test_m5_code_graph.py tests/test_m6_interfaces_docs.py` | `103 passed`. |
| Lint | `.venv/bin/ruff check <M0-M6 baseline paths>` | OK. |
| Typecheck | `PYTHONPATH=src .venv/bin/mypy <M0-M6 baseline paths>` | OK. |
| EvalFly smoke | `.venv/bin/python scripts/evalfly_smoke.py --suite smoke` | `evalfly committed smoke passed: 33 cases`. |
| Installed API/MCP smoke | `/tmp/coderag-m7-venv/bin/python -c "from fastapi.testclient import TestClient; from coderag.main import create_app; ..."` | `/health` returned 200, OpenAPI contained `/api/v1/search-hybrid` and `/api/v1/graph/blast-radius`, MCP mount was present. |
| Installed UI smoke | `/tmp/coderag-m7-venv/bin/python -c "from fastapi.testclient import TestClient; from coderag.main import create_app; ... client.get('/') ..."` | `/` returned 200 and response text contained Gradio/CodeRAG UI content. |
| Compile | `.venv/bin/python -m py_compile <M0-M6 runtime/test paths>` | OK. |

## Surface coverage

- CLI: clean-install `coderag --help` proves entrypoint availability; M6 tests cover structured `search-hybrid` JSON; M5 tests cover `blast-radius` JSON; M4 tests cover `context-pack` output.
- REST: installed-app OpenAPI smoke proves user-visible routes are mounted; API route tests cover index/query/context-pack/search-hybrid/graph contracts with explicit schema assertions.
- MCP: installed-app smoke proves MCP mount is present; MCP handler/tool tests cover indexing, query, search, context pack, and graph tool availability/shape.
- UI: installed-app TestClient smoke proves the Gradio root route renders an HTTP 200 page containing Gradio/CodeRAG content.
- Retrieval/context: M4 tests cover hybrid ranking, dedupe, budgets, and context-pack snippets.
- Citations: M3 tests cover parsed citation verification, grounded flag semantics, and adapter exposure.
- Graph: M5 tests cover Python and TypeScript/JavaScript extraction, persisted replacement, symbol lookup, references, and blast radius/test association.
- Docs: M6 tests assert README wording: verified codebase context engine, not an autonomous coding agent, explicit production-readiness limits, and unsupported graph language fallback.

## Explicit exclusions

- Hosted deployment hardening is not claimed. Public binding still requires explicit host, CORS, authentication, and network controls.
- The UI smoke is HTTP-level through FastAPI TestClient, not a browser-visible Playwright journey.
- CodeRAG does not autonomously edit code or run target repository tests.

## Release checklist against roadmap AC-0..AC-10

- AC-0 governance: PASS — SpecSafe slice and committed contracts exist.
- Baseline/CI: PASS — CI-equivalent local gates pass and CI workflow includes M0-M6 gates plus EvalFly M7 artifact checks.
- Safe defaults/privacy profiles: PASS — loopback defaults and provider profile guardrails covered by M0/M1 tests and README wording.
- SQLite registry/job state: PASS — M2 registry tests pass.
- Hybrid retrieval/context packs: PASS — M4 tests pass.
- Verified citations: PASS — M3 tests pass.
- Code graph/context intelligence: PASS — M5 tests pass for supported languages with explicit fallback.
- MCP/API/CLI/UI interfaces: PASS — installed-app smoke plus M4-M6 tests cover mounted routes and structured outputs.
- EvalFly gates: PASS — committed smoke suite has 33 passing cases, including M7 artifact and behavior coverage-pointer gates.
- Release evidence: PASS — this run record plus EvalFly run/report artifacts committed.
- Production claim discipline: PASS — docs avoid unrestricted production-ready claims.
