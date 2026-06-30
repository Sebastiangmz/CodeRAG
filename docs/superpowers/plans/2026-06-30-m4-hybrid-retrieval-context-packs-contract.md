# Milestone 4 Contract: Hybrid Retrieval and Context Packs

**Slice:** `CRAG-20260630-M4-HYBRID-CONTEXT-PACKS`

**Branch:** `feat/m4-hybrid-retrieval-context-packs`

**Goal:** Modernize CodeRAG beyond vector-only retrieval by adding explainable hybrid retrieval and structured context packs that work without LLM generation.

## Scope

- Add local lexical retrieval over indexed chunks.
- Preserve existing semantic vector retrieval and query behavior.
- Merge vector + lexical candidates with score components.
- Add path/name metadata boosting.
- Deduplicate same-file overlapping ranges.
- Add deterministic token/context budgeting.
- Add `ContextPack` schema and shared service method.
- Expose context packs via MCP, REST, and CLI.
- Extend deterministic tests, EvalFly smoke checks, and CI M4 gates.

## Non-goals

- No code graph MVP in M4.
- No autonomous editing behavior.
- No new model/provider abstraction beyond using existing embeddings/vector store.
- No production-ready claim.
- No broad README rewrite beyond later M6.

## Required Interfaces

```python
@dataclass
class ScoreBreakdown:
    vector: float | None
    lexical: float | None
    path: float
    name: float
    combined: float

@dataclass
class RetrievalBudget:
    max_chunks: int
    max_tokens: int
    max_chunks_per_file: int

class HybridRetriever:
    def retrieve(
        self,
        query: str,
        repo_id: str,
        top_k: int,
        budget: RetrievalBudget | None = None,
    ) -> list[RetrievedChunk]: ...
```

`RetrievedChunk` may be extended with optional metadata fields for score components, token estimate, and retrieval sources without breaking existing callers.

```python
class RetrievalService:
    def search_hybrid(self, repo_id: str, query: str, top_k: int = 10) -> list[RetrievedChunk]: ...
    def get_context_pack(self, repo_id: str, query: str, top_k: int = 8, max_tokens: int = 4000) -> ContextPack: ...
```

## Acceptance Criteria

### AC-M4.1 Hybrid retrieval

**Pass requires**
- Semantic vector candidates remain included when embeddings/vectorstore are available.
- Lexical candidates are computed locally from indexed chunk text.
- Combined score includes vector, lexical, path, name, and combined components.
- Ranking order is explainable from score components.

**Fail if**
- Retrieval remains vector-only.
- Lexical-only exact matches cannot surface without semantic score.
- Results have no score explanation.

### AC-M4.2 Dedupe and budget

**Pass requires**
- Same-file overlapping line ranges are merged/deduped.
- Budget enforces max chunks, max estimated tokens, and max chunks per file deterministically.
- Top relevant result is preserved when within budget.

**Fail if**
- Duplicate vector+lexical hits consume multiple slots.
- Budget fields are ignored.

### AC-M4.3 Context packs

**Pass requires**
- Context packs are generated without invoking an LLM.
- Pack includes query, repo id, snippets, citation candidates, score/ranking reasons, token estimate, budget, and capability metadata.
- MCP, REST, and CLI expose the pack as structured JSON; CLI also exposes markdown.

**Fail if**
- Context pack generation requires cloud or local generation.
- Pack lacks provenance for why files/snippets were selected.

### AC-M4.4 Compatibility

**Pass requires**
- Existing `query_code` behavior still returns answer/citations/evidence/grounded.
- Existing `search_code` compatibility is preserved or a migration test proves the shape remains usable.
- API/MCP/CLI tests cover new outputs.

**Fail if**
- Existing M0-M3 tests are weakened or removed.
- MCP/CLI/API disagree on context-pack result shape.

## Required Verification

Run before closing the slice:

```bash
bun run /Users/sebastian/.omp/omp-pantheon/skills/specsafe/bin/specsafe.ts status
PYTHONPATH=src .venv/bin/python -m pytest tests/test_m4_hybrid_retrieval.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_m3_citations.py
.venv/bin/python -m ruff check src/coderag/retrieval src/coderag/models src/coderag/services src/coderag/indexing/vectorstore.py src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/cli.py tests/test_m4_hybrid_retrieval.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_m3_citations.py scripts/evalfly_smoke.py
PYTHONPATH=src .venv/bin/mypy src/coderag/retrieval src/coderag/models src/coderag/services src/coderag/indexing/vectorstore.py src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py tests/test_m4_hybrid_retrieval.py tests/test_api_routes.py tests/test_mcp_handlers.py tests/test_mcp_tools.py tests/test_m3_citations.py scripts/evalfly_smoke.py
.venv/bin/python scripts/evalfly_smoke.py --suite smoke
.venv/bin/python -m py_compile src/coderag/retrieval/*.py src/coderag/models/*.py src/coderag/services/*.py src/coderag/indexing/vectorstore.py src/coderag/api/routes.py src/coderag/api/schemas.py src/coderag/mcp/handlers.py src/coderag/mcp/tools.py src/coderag/cli.py tests/test_m4_hybrid_retrieval.py
.venv/bin/python -m build
```

Then run the local CI suite with M0-M4 tests.

## Review Gate

A reviewer must approve that:
- hybrid retrieval is not vector-only;
- context packs do not call generation;
- score/budget/dedupe tests are behavior-level;
- public adapters remain JSON-safe;
- M4 does not implement graph/autonomous behavior.
