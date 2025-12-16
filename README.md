# CodeRAG - Code Q&A with Verifiable Citations

RAG-based Q&A system for code repositories that provides grounded answers with verifiable citations.

## Features

- **Grounded Responses**: Every answer includes citations to source code `[file:start-end]`
- **Local-First**: Runs 100% locally with Qwen2.5-Coder-7B and nomic-embed-text
- **GitHub Integration**: Index any public GitHub repository
- **Semantic Chunking**: Tree-sitter for Python, text fallback for other languages
- **Web Interface**: Gradio UI for easy interaction
- **REST API**: Programmatic access for integration
- **GPU Optimized**: 4-bit quantization fits in 8GB VRAM

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU support)
- 8GB+ GPU VRAM (RTX 4060 or better)
- Docker & Docker Compose (optional)

### Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG

# Start the application
docker compose up
```

Access the UI at http://localhost:8000

#### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG

# Install dependencies
pip install -e .

# Create .env file
cp .env.example .env

# Run the application
python -m coderag.main
```

## Usage

### Web Interface

1. Navigate to http://localhost:8000
2. Go to "Index Repository" tab
3. Enter a GitHub URL (e.g., `https://github.com/owner/repo`)
4. Click "Index Repository" and wait for completion
5. Go to "Ask Questions" tab
6. Select your repository
7. Ask questions about the code

### REST API

#### Index a Repository

```bash
curl -X POST http://localhost:8000/api/v1/repos/index \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/owner/repo",
    "branch": "main"
  }'
```

#### Query a Repository

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where is function X defined?",
    "repo_id": "your-repo-id",
    "top_k": 5
  }'
```

#### List Repositories

```bash
curl http://localhost:8000/api/v1/repos
```

## Configuration

Edit `.env` file or use environment variables:

```bash
# Models
MODEL_LLM_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
MODEL_EMBEDDING_NAME=nomic-ai/nomic-embed-text-v1.5

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Data directories
VECTORSTORE_PERSIST_DIRECTORY=./data/chroma_db
INGESTION_REPOS_CACHE_DIR=./data/repos

# Retrieval
RETRIEVAL_DEFAULT_TOP_K=5
RETRIEVAL_SIMILARITY_THRESHOLD=0.3
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (Gradio + REST API)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     Ingestion Pipeline                        │
│  GitHub Clone → File Filter → Chunker (Tree-sitter/Text)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   Indexing & Storage                          │
│      Embeddings (nomic-embed) → ChromaDB (Cosine)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    Retrieval & Generation                     │
│   Query → Top-K Search → LLM (Qwen2.5-Coder) → Response     │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/coderag/
├── ingestion/      # Repository loading and chunking
├── indexing/       # Embeddings and vector storage
├── retrieval/      # Semantic search
├── generation/     # LLM inference and citations
├── ui/             # Gradio web interface
├── api/            # REST API endpoints
└── models/         # Data models

configs/            # Configuration files
tests/              # Unit tests
eval_datasets/      # Evaluation datasets
```

## Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Performance

- **Indexing**: ~1000 files in < 5 minutes
- **Query**: Response in < 10 seconds
- **VRAM Usage**: ~6GB total (LLM 4.5GB + Embeddings 0.5GB + overhead 1GB)

## Citation Format

All responses include citations in the format:

```
[file_path:start_line-end_line]
```

Example:
```
The authentication logic is implemented in the login() function [src/auth.py:45-78].
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- **Issues**: https://github.com/Sebastiangmz/CodeRAG/issues
- **Documentation**: https://github.com/Sebastiangmz/CodeRAG/wiki

## Acknowledgments

- Qwen2.5-Coder by Alibaba Cloud
- nomic-embed-text by Nomic AI
- ChromaDB for vector storage
- Tree-sitter for code parsing
