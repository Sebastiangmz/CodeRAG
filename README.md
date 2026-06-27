# CodeRAG - Code Q&A with Verifiable Citations

[![PyPI version](https://badge.fury.io/py/code-rag-me.svg)](https://pypi.org/project/code-rag-me/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG-based Q&A system for code repositories that provides grounded answers with verifiable citations.

## 🚀 Quick Start

```bash
# Install
pip install code-rag-me

# Configure (get free API key from https://console.groq.com/keys)
coderag setup

# Start web interface
coderag serve
```

Open http://localhost:8000 to use the web interface.

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

## 🔧 Installation

### Linux

#### Arch Linux / Manjaro

Arch Linux uses PEP 668 to protect system Python. Use one of these methods:

**Option A: pipx (Recommended for CLI tools)**
```bash
sudo pacman -S python-pipx
pipx install code-rag-me
```

**Option B: Virtual environment**
```bash
python -m venv ~/.local/share/coderag-venv
source ~/.local/share/coderag-venv/bin/activate
pip install code-rag-me
```

To always have `coderag` available, add to your `~/.bashrc` or `~/.zshrc`:
```bash
alias coderag="~/.local/share/coderag-venv/bin/coderag"
```

#### Debian / Ubuntu

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Option A: pipx (Recommended)
sudo apt install pipx
pipx install code-rag-me

# Option B: Virtual environment
python3 -m venv ~/.local/share/coderag-venv
source ~/.local/share/coderag-venv/bin/activate
pip install code-rag-me
```

#### Fedora / RHEL / CentOS

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Option A: pipx (Recommended)
sudo dnf install pipx
pipx install code-rag-me

# Option B: Virtual environment
python3 -m venv ~/.local/share/coderag-venv
source ~/.local/share/coderag-venv/bin/activate
pip install code-rag-me
```

#### Other Linux Distributions

```bash
# Create virtual environment
python3 -m venv ~/.local/share/coderag-venv
source ~/.local/share/coderag-venv/bin/activate
pip install code-rag-me
```

### macOS

**Option A: pipx (Recommended)**
```bash
# Install pipx via Homebrew
brew install pipx
pipx ensurepath
pipx install code-rag-me
```

**Option B: Virtual environment**
```bash
python3 -m venv ~/.local/share/coderag-venv
source ~/.local/share/coderag-venv/bin/activate
pip install code-rag-me
```

**Option C: Homebrew Python**
```bash
brew install python@3.11
pip3 install code-rag-me
```

### Windows

#### Option A: pipx (Recommended)

```powershell
# Install pipx
pip install pipx
pipx ensurepath

# Install CodeRAG
pipx install code-rag-me
```

#### Option B: Virtual environment

```powershell
# Create virtual environment
python -m venv %USERPROFILE%\coderag-venv

# Activate (Command Prompt)
%USERPROFILE%\coderag-venv\Scripts\activate.bat

# Activate (PowerShell)
& $env:USERPROFILE\coderag-venv\Scripts\Activate.ps1

# Install
pip install code-rag-me
```

#### Option C: Direct install (not recommended)

```powershell
pip install code-rag-me
```

> **Note**: On Windows, you may need to run PowerShell as Administrator or enable script execution with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### From Source

```bash
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG
pip install -e .
coderag setup
```

### Docker

```bash
git clone https://github.com/Sebastiangmz/CodeRAG.git
cd CodeRAG
docker compose up
```

### Post-Installation

After installing, configure your LLM provider:

```bash
coderag setup
```

This will prompt you to:
1. Choose an LLM provider (Groq recommended - free tier available)
2. Enter your API key (get one at https://console.groq.com/keys)
3. Configure optional settings

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

# Server defaults to loopback for local safety.
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Bind to 0.0.0.0 only as an explicit deployment override
# after configuring trusted CORS origins and network access controls.
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

### Common Issues

#### `externally-managed-environment` error (Linux)

```
error: externally-managed-environment
× This environment is externally managed
```

This happens on modern Linux distributions (Arch, Fedora 38+, Ubuntu 23.04+) that implement PEP 668. Solution: use `pipx` or a virtual environment. See the [Installation](#-installation) section for your distribution.

#### No API key configured

```bash
coderag setup  # Run interactive setup
```

#### CUDA / GPU errors

If you don't have a GPU or encounter CUDA errors:
```bash
export MODEL_EMBEDDING_DEVICE=cpu
coderag serve
```

Or add to your `.env` file:
```
MODEL_EMBEDDING_DEVICE=cpu
```

#### Claude Desktop not detecting MCP

1. Run `coderag mcp-install`
2. Completely quit Claude Desktop (not just close the window)
3. Restart Claude Desktop
4. Check the MCP icon in Claude Desktop settings

#### Permission denied on Linux/macOS

```bash
# If using pipx
pipx ensurepath
source ~/.bashrc  # or ~/.zshrc

# If using venv, make sure it's activated
source ~/.local/share/coderag-venv/bin/activate
```

#### PowerShell execution policy (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

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
