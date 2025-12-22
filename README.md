# CodeRAG - Code Q&A with Verifiable Citations

[![PyPI version](https://badge.fury.io/py/coderag.svg)](https://badge.fury.io/py/coderag)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG-based Q&A system for code repositories that provides grounded answers with verifiable citations.

## 🚀 Quick Start (No GPU Required)

```bash
# Install
pip install coderag

# Configure (get free API key from https://console.groq.com/keys)
coderag setup

# Start web interface
coderag serve
```

That's it! Open http://localhost:8000 to use the web interface.

### Claude Desktop Integration (MCP)

```bash
# Auto-configure Claude Desktop
coderag mcp-install

# Restart Claude Desktop
```

Now you can use CodeRAG directly in Claude Desktop!

## ✨ Features

- **Grounded Responses**: Every answer includes citations to source code `[file:start-end]`
- **Cloud or Local LLM**: Use Groq (free), OpenAI, Anthropic, or run locally with GPU
- **GitHub Integration**: Index any public GitHub repository
- **MCP Support**: Integrate directly with Claude Desktop
- **Semantic Chunking**: Tree-sitter for Python, text fallback for other languages
- **Web Interface**: Gradio UI for easy interaction
- **REST API**: Programmatic access for integration
- **CLI**: Full command-line interface

## 📋 CLI Commands

```bash
coderag setup              # Configure LLM provider and API key
coderag serve              # Start web server
coderag mcp-install        # Configure Claude Desktop for MCP
coderag mcp-run            # Run MCP server (used by Claude Desktop)
coderag index <url>        # Index a GitHub repository
coderag query <repo> "?"   # Ask a question about code
coderag repos              # List indexed repositories
coderag doctor             # Diagnose setup issues
```

## 🔧 Installation Options

### Option 1: pip (Recommended)

```bash
pip install coderag
coderag setup
```

### Option 2: From Source

```bash
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG
pip install -e .
coderag setup
```

### Option 3: Docker

```bash
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG
docker compose up
```

## 📖 Usage Examples

### Web Interface

1. Run `coderag serve`
2. Open http://localhost:8000
3. Go to "Index Repository" → Enter GitHub URL → Click "Index"
4. Go to "Ask Questions" → Select repo → Ask questions

### Command Line

```bash
# Index a repository
coderag index https://github.com/owner/repo

# Ask questions
coderag query abc12345 "How does authentication work?"

# List repositories
coderag repos
```

### REST API

```bash
# Index repository
curl -X POST http://localhost:8000/api/v1/repos/index \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/owner/repo"}'

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does X work?", "repo_id": "abc12345"}'
```

### Claude Desktop (MCP)

After running `coderag mcp-install` and restarting Claude Desktop:

```
You: Use coderag to index https://github.com/owner/repo

Claude: I'll index that repository for you...
        ✅ Indexed! 150 files, 1,234 chunks.

You: How does the authentication system work?

Claude: Based on the code, authentication is handled in...
        [src/auth/handler.py:45-78]
```

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider (groq, openai, anthropic, openrouter, together, local)
MODEL_LLM_PROVIDER=groq
MODEL_LLM_API_KEY=your-api-key

# Embeddings (runs locally on CPU by default)
MODEL_EMBEDDING_DEVICE=auto  # auto, cuda, or cpu

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### Config File

Configuration is stored in `~/.config/coderag/config.json` after running `coderag setup`.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│              (Gradio UI / REST API / MCP / CLI)             │
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
│   Query → Top-K Search → LLM (Cloud/Local) → Response       │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
src/coderag/
├── cli.py          # Unified CLI
├── ingestion/      # Repository loading and chunking
├── indexing/       # Embeddings and vector storage
├── retrieval/      # Semantic search
├── generation/     # LLM inference and citations
├── mcp/            # Model Context Protocol server
├── ui/             # Gradio web interface
├── api/            # REST API endpoints
└── models/         # Data models
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## 📊 Performance

- **Indexing**: ~1000 files in < 5 minutes
- **Query**: Response in < 10 seconds
- **Embeddings**: Runs on CPU (~275MB model)
- **LLM**: Cloud (instant) or Local (requires 8GB+ VRAM)

## 📝 Citation Format

All responses include citations:

```
[file_path:start_line-end_line]
```

Example:
```
The authentication logic is in the login() function [src/auth.py:45-78].
```

## 🐛 Troubleshooting

Run diagnostics:
```bash
coderag doctor
```

Common issues:
- **No API key**: Run `coderag setup` to configure
- **CUDA errors**: Set `MODEL_EMBEDDING_DEVICE=cpu` or use cloud LLM
- **Claude Desktop not detecting MCP**: Restart Claude Desktop after `mcp-install`

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🙏 Acknowledgments

- [Groq](https://groq.com) for fast, free LLM inference
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) by Nomic AI
- [ChromaDB](https://www.trychroma.com) for vector storage
- [Tree-sitter](https://tree-sitter.github.io) for code parsing
- [MCP](https://modelcontextprotocol.io) by Anthropic
