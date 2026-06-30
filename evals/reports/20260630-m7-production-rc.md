# EvalFly Report: 20260630-m7-production-rc

## Verdict

PASS.

## Evidence

- EvalFly smoke: `.venv/bin/python scripts/evalfly_smoke.py --suite smoke` → `evalfly committed smoke passed: 33 cases`.
- CI-equivalent pytest gate: M0-M6 targeted tests → `103 passed`.
- Clean install gate: built `dist/code_rag_me-0.1.2-py3-none-any.whl`, installed it into `/tmp/coderag-m7-venv`, ran `coderag --help`, and imported `coderag.__version__` → `0.1.2`.

## Covered surfaces

- Governance contracts M0-M7.
- Safe defaults, provider profiles, citation verification, hybrid retrieval, context packs, graph extraction/blast radius, REST/MCP/CLI interface contracts, README readiness wording.

## Explicit exclusions

- Hosted/SaaS production hardening is not claimed.
- Browser-visible Gradio E2E is not claimed in this report; local UI launch remains covered only by package/build/import and existing unit-level handlers.
