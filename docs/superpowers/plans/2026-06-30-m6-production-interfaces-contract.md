# Milestone 6 Contract: Production Interfaces and Docs

## Scope

Stabilize the user-facing CodeRAG contracts across REST, MCP, CLI, and README without claiming production readiness beyond verified evidence.

## Acceptance Criteria

- M6-AC1: REST/MCP/CLI expose the same documented capability set where implementation exists: index, update, list, query, search, hybrid search, context packs, symbol lookup, reference lookup, and blast radius.
- M6-AC2: Representative REST, MCP, and CLI output shapes are covered by deterministic snapshot/contract tests.
- M6-AC3: User-facing docs position CodeRAG as a verified codebase context engine, not an autonomous coding agent.
- M6-AC4: Docs state current privacy and readiness limits: loopback default, explicit deployment hardening required, cloud LLMs require user-configured keys, and unsupported graph languages degrade explicitly.
- M6-AC5: CI and EvalFly smoke gates include the M6 interface/doc contract checks.

## Non-goals

- No autonomous editing behavior.
- No blanket production-ready claim.
- No broad redesign of Gradio UI.
- No expansion of unsupported language graph confidence.
