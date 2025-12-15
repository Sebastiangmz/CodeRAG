# CodeRAG Constitution

<!--
SYNC IMPACT REPORT
==================
Version Change: 0.0.0 → 1.0.0 (MAJOR - Initial constitution)
Added Principles:
  - I. Grounded Responses (NON-NEGOTIABLE)
  - II. Local-First & Open Source
  - III. Citation-Driven Output
  - IV. Modular Pipeline Architecture
  - V. Hardware-Conscious Design
  - VI. Reproducible Deployment
  - VII. Evaluation-Driven Development
  - VIII. Security & Privacy First
Added Sections:
  - Technical Constraints
  - Development Workflow
  - Governance
Templates Status:
  - plan-template.md: ✅ Compatible
  - spec-template.md: ✅ Compatible
  - tasks-template.md: ✅ Compatible
Follow-up TODOs: None
-->

## Core Principles

### I. Grounded Responses (NON-NEGOTIABLE)

The system MUST generate responses that are fundamentally grounded in retrieved evidence. This is the core value proposition of CodeRAG.

- Every factual assertion MUST include a verifiable citation in format `[file:start_line-end_line]`
- The system MUST refuse to answer when insufficient evidence exists in retrieved chunks
- Hallucination is unacceptable: never invent code, functions, files, or behaviors not present in the indexed repository
- When context is ambiguous, the system MUST request clarification rather than guess
- Responses MUST be concise, structured, and directly supported by retrieved content

**Rationale**: RAG systems fail when they hallucinate. Our differentiation is trustworthy, verifiable answers about code.

### II. Local-First & Open Source

All core functionality MUST run 100% locally without external API dependencies or costs.

- LLM generation: Qwen2.5-Coder-7B-Instruct (Apache 2.0) with 4-bit quantization
- Embeddings: nomic-embed-text v1.5 (Apache 2.0)
- Vector storage: ChromaDB (local persistence)
- No API keys required for core functionality
- Models download from Hugging Face Hub on first run
- Respect license obligations: maintain copyright notices, do not redistribute model weights in repo

**Rationale**: Eliminates cost barriers, ensures privacy, enables offline operation, and keeps the project truly open source for portfolio demonstration.

### III. Citation-Driven Output

Every response MUST provide traceable evidence to the source code.

- Citation format: `[file_path:start_line-end_line]` (e.g., `[src/auth/handlers.py:45-78]`)
- Multiple citations per response when evidence spans multiple locations
- Citations MUST be accurate and verifiable against the indexed repository
- Include relevant code snippets only when short and directly supportive
- Provide "Evidence" section listing all chunks used in response generation

**Rationale**: Citations transform the assistant from a black box into a transparent research tool that builds user trust.

### IV. Modular Pipeline Architecture

The system MUST be composed of independent, single-responsibility modules.

- **Ingestion**: Repository loading, file filtering, chunking (separate concerns)
- **Indexing**: Embedding generation, vector storage (separate concerns)
- **Retrieval**: Query processing, similarity search, ranking (separate concerns)
- **Generation**: Prompt construction, LLM inference, citation parsing (separate concerns)
- Each module MUST be independently testable
- Configuration via YAML files, not hardcoded values
- Clear interfaces between modules using dataclasses for type safety
- Avoid premature abstraction: start simple, refactor when patterns emerge

**Rationale**: Modularity enables independent development, testing, and future replacement of components (e.g., swapping vector DB or LLM).

### V. Hardware-Conscious Design

All design decisions MUST respect the target hardware constraints: NVIDIA RTX 4060 with 8GB VRAM.

- LLM MUST use 4-bit quantization (bitsandbytes NF4)
- Total VRAM budget: ~6GB (LLM ~4.5GB + Embeddings ~0.5GB + Overhead ~1GB)
- Batch embedding generation to avoid OOM
- Provide lightweight model alternative (Llama-3.2-3B) for constrained environments
- Monitor and log VRAM usage in development
- Fail gracefully with clear error messages when resources are insufficient

**Rationale**: The project targets consumer hardware to maximize accessibility and demonstrate practical ML engineering.

### VI. Reproducible Deployment

The system MUST be 100% reproducible via Docker Compose with a single command.

- `docker compose up` MUST produce a working system from scratch
- All dependencies pinned to specific versions
- Persistent volumes for: ChromaDB data, cloned repos cache, HuggingFace model cache, LoRA adapters
- No manual configuration steps required beyond `.env` setup
- Health checks for service readiness
- Clear separation: code in Git, artifacts in volumes (never commit models, indexes, or adapters)

**Rationale**: Reproducibility is essential for portfolio demonstration, collaboration, and professional credibility.

### VII. Evaluation-Driven Development

Features MUST be validated against measurable evaluation criteria.

- Maintain evaluation datasets in `eval_datasets/` (JSONL format)
- Two test categories: Closed (answer exists in repo) and Open (answer doesn't exist - test abstention)
- Key metrics:
  - Retrieval: Precision@K, Recall@K, MRR
  - Generation: Faithfulness, Citation Accuracy
  - End-to-end: Correctness, Groundedness, Abstention Accuracy
- Document baseline metrics before optimization
- Compare metrics after changes to validate improvements
- No feature is complete without evaluation demonstrating it works

**Rationale**: Quantitative evaluation prevents regression and provides evidence of system quality for portfolio.

### VIII. Security & Privacy First

The system MUST handle code repositories responsibly and securely.

- Never index files likely containing secrets: `.env`, `credentials.json`, API keys
- Sanitize all user inputs (URLs, queries)
- Rate limiting on API endpoints
- Volumes containing indexed data stay local, never in Git
- Clear warnings when processing potentially sensitive content
- MVP scope: public repositories only (no authentication complexity)

**Rationale**: Code repositories often contain sensitive information. Responsible handling builds trust and demonstrates security awareness.

## Technical Constraints

### Stack Requirements

| Component | Technology | Constraint |
|-----------|------------|------------|
| Language | Python 3.11+ | Type hints required |
| Web Framework | FastAPI | Async endpoints |
| UI Framework | Gradio 4.x | Mounted on FastAPI |
| Vector DB | ChromaDB | Cosine similarity |
| LLM | Qwen2.5-Coder-7B-Instruct | 4-bit quantization |
| Embeddings | nomic-embed-text v1.5 | 768 dimensions |
| Code Parser | Tree-sitter | Python AST extraction |
| Container | Docker + NVIDIA runtime | CUDA 12.1+ |

### Performance Targets

- Indexing: Process 1000 files in under 5 minutes
- Query latency: Response within 10 seconds (including LLM generation)
- Memory: Stay within 8GB VRAM budget
- Storage: Model cache ~15GB, indexes scale with repo size

### Code Quality Standards

- Formatting: Black (line length 100)
- Linting: Ruff
- Type checking: mypy (strict mode)
- Documentation: Docstrings on public functions
- Testing: pytest with fixtures

## Development Workflow

### Branch Strategy

- `main`: Stable, deployable code
- `feature/*`: New features (merge via PR)
- `fix/*`: Bug fixes

### Commit Standards

- Descriptive messages explaining "why" not just "what"
- Reference issue numbers when applicable
- Small, focused commits over large monolithic changes

### Review Checklist

- [ ] Follows grounded response principle
- [ ] Respects hardware constraints
- [ ] Includes evaluation metrics (if applicable)
- [ ] No secrets or sensitive data
- [ ] Docker build succeeds
- [ ] Tests pass

### What to Commit

**YES - Commit to Git:**
- All source code (`src/`)
- Configuration files (`configs/*.yaml`)
- Dockerfiles and compose files
- Evaluation datasets (`eval_datasets/`)
- Training scripts (`scripts/`)
- Documentation
- `.env.example` (not `.env`)

**NO - Never Commit:**
- Model weights (`*.safetensors`, `*.bin`)
- LoRA adapters (`adapters/`)
- Vector indexes (`data/`, `chroma_db/`)
- Cloned repositories (`repos/`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

## Governance

### Constitution Authority

This constitution supersedes all other development practices. When in conflict, constitution principles take precedence.

### Amendment Process

1. Propose amendment with rationale
2. Document impact on existing code
3. Update version number:
   - MAJOR: Principle removal or fundamental redefinition
   - MINOR: New principle or significant expansion
   - PATCH: Clarifications, wording improvements
4. Update dependent templates if affected
5. Communicate changes to all contributors

### Compliance Verification

All pull requests MUST verify:
- No constitution violations
- Complexity justified if exceeding simplicity principle
- Evaluation metrics maintained or improved

### Runtime Guidance

For day-to-day development decisions not covered by this constitution, refer to:
- `RAG_CODEBASE_ARCHITECTURE_UPDATED.md` for detailed technical specifications
- `openspec/project.md` for project conventions
- Component-specific documentation in `docs/`

**Version**: 1.0.0 | **Ratified**: 2025-12-15 | **Last Amended**: 2025-12-15
