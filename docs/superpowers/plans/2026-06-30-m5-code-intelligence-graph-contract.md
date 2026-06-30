# CodeRAG Milestone 5: Code Intelligence Graph Contract

Spec-Slice: CRAG-20260630-M5-CODE-INTELLIGENCE-GRAPH

## Scope

Milestone 5 adds a deterministic, local code-intelligence graph that can answer graph questions without LLM generation.

## Acceptance

1. Extract symbol graph data for Python and TypeScript/JavaScript documents.
2. Persist graph facts in the SQLite registry, keyed by repository and file path.
3. Support incremental replacement by file path so changed files do not require a full graph rebuild.
4. Expose service, REST, MCP, and CLI paths for:
   - `find_symbol`
   - `find_references`
   - `get_blast_radius`
5. Return explicit capability flags when a language is unsupported and falls back to text retrieval only.
6. Include graph evidence in blast radius: impacted files, symbols, direct edges, associated tests, and reason strings.

## Required tests

- Parser fixtures for Python and TS/JS symbols/imports/calls/references.
- Registry persistence and per-file replacement tests.
- Service tests for symbol lookup, references, blast radius, and unsupported-language capability fallback.
- REST/MCP/CLI adapter shape tests.
- EvalFly smoke cases for graph model, service, and adapter exposure.

## Verification

Run before closing the slice:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py tests/test_m4_hybrid_retrieval.py tests/test_m5_code_graph.py
.venv/bin/ruff check src/coderag/generation src/coderag/indexing/vectorstore.py src/coderag/models src/coderag/retrieval src/coderag/services src/coderag/config.py src/coderag/api/schemas.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/cli.py src/coderag/ui/grounding.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_ui_handlers.py tests/test_m4_hybrid_retrieval.py tests/test_m5_code_graph.py scripts/evalfly_smoke.py
PYTHONPATH=src .venv/bin/mypy src/coderag/generation src/coderag/indexing/vectorstore.py src/coderag/models src/coderag/retrieval src/coderag/services src/coderag/config.py src/coderag/api/schemas.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/ui/grounding.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_ui_handlers.py tests/test_m4_hybrid_retrieval.py tests/test_m5_code_graph.py scripts/evalfly_smoke.py
python -m py_compile src/coderag/__init__.py src/coderag/config.py src/coderag/main.py src/coderag/cli.py src/coderag/indexing/embeddings.py src/coderag/indexing/vectorstore.py src/coderag/generation/*.py src/coderag/models/*.py src/coderag/retrieval/*.py src/coderag/services/*.py src/coderag/api/schemas.py src/coderag/api/routes.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/ui/grounding.py src/coderag/ui/repository_metadata.py src/coderag/ui/handlers.py tests/test_m0_baseline.py tests/test_m1_services.py tests/test_m2_registry.py tests/test_citations.py tests/test_m3_citations.py tests/test_api_routes.py tests/test_ui_handlers.py tests/test_m4_hybrid_retrieval.py tests/test_m5_code_graph.py scripts/evalfly_smoke.py
python scripts/evalfly_smoke.py --suite smoke
.venv/bin/python -m build
```
