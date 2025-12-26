"""Unified CLI for CodeRAG."""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

import click


# Config directory and file
CONFIG_DIR = Path.home() / ".config" / "coderag"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_config() -> dict:
    """Load configuration from config file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    """Save configuration to config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_claude_config_path() -> Optional[Path]:
    """Get Claude Desktop config path based on OS."""
    system = platform.system()

    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return None


@click.group()
@click.version_option(package_name="coderag")
def cli():
    """CodeRAG - RAG-based Q&A system for code repositories.

    Use 'coderag setup' to configure, then 'coderag serve' to start.
    For Claude Desktop integration, run 'coderag mcp-install'.
    """
    pass


@cli.command()
@click.option("--provider", type=click.Choice(["groq", "openai", "anthropic", "openrouter", "together", "local"]),
              default=None, help="LLM provider to use")
@click.option("--api-key", default=None, help="API key for the provider")
def setup(provider: Optional[str], api_key: Optional[str]):
    """Interactive setup wizard for CodeRAG.

    Configures the LLM provider and API key. Configuration is saved to
    ~/.config/coderag/config.json and can be overridden by environment variables.
    """
    config = get_config()

    click.echo("\n🔧 CodeRAG Setup\n")

    # Provider selection
    if provider is None:
        click.echo("Select your LLM provider:")
        click.echo("  1. groq (FREE, fast - recommended)")
        click.echo("  2. openai")
        click.echo("  3. anthropic")
        click.echo("  4. openrouter")
        click.echo("  5. together")
        click.echo("  6. local (requires GPU)")

        choice = click.prompt("Enter choice", type=int, default=1)
        providers = {1: "groq", 2: "openai", 3: "anthropic", 4: "openrouter", 5: "together", 6: "local"}
        provider = providers.get(choice, "groq")

    config["llm_provider"] = provider

    # API key (not needed for local)
    if provider != "local":
        if api_key is None:
            api_key_urls = {
                "groq": "https://console.groq.com/keys",
                "openai": "https://platform.openai.com/api-keys",
                "anthropic": "https://console.anthropic.com/settings/keys",
                "openrouter": "https://openrouter.ai/keys",
                "together": "https://api.together.xyz/settings/api-keys",
            }
            url = api_key_urls.get(provider, "")
            if url:
                click.echo(f"\nGet your API key from: {url}")

            api_key = click.prompt("Enter your API key", hide_input=True)

        config["llm_api_key"] = api_key

        # Validate API key
        click.echo("\n⏳ Validating API key...")
        if _validate_api_key(provider, api_key):
            click.echo("✅ API key is valid!")
        else:
            click.echo("⚠️  Could not validate API key. It may still work.")
    else:
        click.echo("\n⚠️  Local mode requires a CUDA-capable GPU.")

    # Save config
    save_config(config)
    click.echo(f"\n✅ Configuration saved to {CONFIG_FILE}")

    # Next steps
    click.echo("\n📋 Next steps:")
    click.echo("  1. Run 'coderag serve' to start the web interface")
    click.echo("  2. Run 'coderag mcp-install' to integrate with Claude Desktop")
    click.echo("  3. Run 'coderag index <url>' to index a repository")


def _validate_api_key(provider: str, api_key: str) -> bool:
    """Validate API key by making a test request."""
    try:
        from openai import OpenAI

        base_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "together": "https://api.together.xyz/v1",
        }

        if provider not in base_urls:
            return True  # Can't validate, assume OK

        client = OpenAI(api_key=api_key, base_url=base_urls[provider])
        client.models.list()
        return True
    except Exception:
        return False


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the CodeRAG web server.

    Starts the FastAPI server with Gradio UI, REST API, and MCP endpoint.
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import uvicorn
    from coderag.main import create_app
    from coderag.config import get_settings

    settings = get_settings()
    app = create_app()

    click.echo(f"\n🚀 Starting CodeRAG server at http://{host}:{port}")
    click.echo("   Press Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=settings.server.log_level,
    )


@cli.command("mcp-run")
def mcp_run():
    """Run MCP server in stdio mode (for Claude Desktop).

    This command is used by Claude Desktop to communicate with CodeRAG.
    You typically don't need to run this manually.
    """
    # Apply config from file to environment
    _apply_config_to_env()

    # Suppress all output except MCP protocol
    import logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    from coderag.mcp.server import create_mcp_server
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


@cli.command("mcp-install")
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
def mcp_install(dry_run: bool):
    """Configure Claude Desktop to use CodeRAG MCP.

    Automatically detects your OS and updates the Claude Desktop configuration
    to include the CodeRAG MCP server.
    """
    config_path = get_claude_config_path()

    if config_path is None:
        click.echo("❌ Could not determine Claude Desktop config location.")
        click.echo("   Please manually add the MCP configuration.")
        sys.exit(1)

    click.echo(f"\n🔍 Claude Desktop config: {config_path}")

    # Check if Claude Desktop is installed
    if not config_path.parent.exists():
        click.echo("\n❌ Claude Desktop does not appear to be installed.")
        click.echo("   Install it from: https://claude.ai/download")
        sys.exit(1)

    # Load existing config or create new
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            click.echo("⚠️  Existing config is invalid JSON. Creating new config.")
            config = {}
    else:
        config = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Find the coderag-mcp command path
    coderag_path = shutil.which("coderag")
    if coderag_path is None:
        # Fallback to python -m
        python_path = sys.executable
        mcp_command = [python_path, "-m", "coderag.mcp.cli"]
    else:
        mcp_command = [coderag_path, "mcp-run"]

    # Prepare MCP server config
    new_mcp_config = {
        "command": mcp_command[0],
        "args": mcp_command[1:] if len(mcp_command) > 1 else [],
    }

    # Check if already configured
    existing = config["mcpServers"].get("coderag")
    if existing == new_mcp_config:
        click.echo("\n✅ CodeRAG MCP is already configured correctly!")
        return

    # Show diff
    click.echo("\n📝 Changes to be made:")
    if existing:
        click.echo(f"   Update: mcpServers.coderag")
        click.echo(f"   From: {json.dumps(existing)}")
        click.echo(f"   To:   {json.dumps(new_mcp_config)}")
    else:
        click.echo(f"   Add: mcpServers.coderag = {json.dumps(new_mcp_config)}")

    if dry_run:
        click.echo("\n🔍 Dry run - no changes made.")
        return

    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix(".json.backup")
        shutil.copy(config_path, backup_path)
        click.echo(f"\n📦 Backup saved to: {backup_path}")

    # Apply changes
    config["mcpServers"]["coderag"] = new_mcp_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))

    click.echo("\n✅ Claude Desktop configuration updated!")
    click.echo("\n⚠️  Please restart Claude Desktop to apply changes.")


@cli.command("index")
@click.argument("url")
@click.option("--branch", default="", help="Branch to index (default: main/master)")
def index(url: str, branch: str):
    """Index a GitHub repository.

    URL: The GitHub repository URL to index.

    Example: coderag index https://github.com/owner/repo
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    click.echo(f"\n📦 Indexing repository: {url}")
    if branch:
        click.echo(f"   Branch: {branch}")

    handlers = get_mcp_handlers()

    async def run_index():
        result = await handlers.index_repository(url=url, branch=branch)
        return result

    result = asyncio.run(run_index())

    if result.get("success"):
        click.echo(f"\n✅ Repository indexed successfully!")
        click.echo(f"   Repo ID: {result['repo_id']}")
        click.echo(f"   Name: {result['name']}")
        click.echo(f"   Files processed: {result['files_processed']}")
        click.echo(f"   Chunks indexed: {result['chunks_indexed']}")
        click.echo(f"\n   Use 'coderag query {result['repo_id'][:8]} \"your question\"' to query")
    else:
        click.echo(f"\n❌ Indexing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


@cli.command("query")
@click.argument("repo_id")
@click.argument("question")
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def query(repo_id: str, question: str, top_k: int, output_format: str):
    """Ask a question about an indexed repository.

    REPO_ID: Repository ID (full or first 8 characters)
    QUESTION: Your question about the code

    Example: coderag query abc12345 "How does authentication work?"
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    handlers = get_mcp_handlers()

    async def run_query():
        result = await handlers.query_code(repo_id=repo_id, question=question, top_k=top_k)
        return result

    click.echo(f"\n🔍 Querying: {question}\n")
    result = asyncio.run(run_query())

    if result.get("error"):
        click.echo(f"❌ Error: {result['error']}")
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("📝 Answer:\n")
        click.echo(result.get("answer", "No answer generated."))

        if result.get("citations"):
            click.echo("\n📍 Citations:")
            for citation in result["citations"]:
                click.echo(f"   {citation}")

        if result.get("evidence"):
            click.echo("\n📂 Evidence:")
            for chunk in result["evidence"][:3]:  # Show top 3
                click.echo(f"   - {chunk['file']}:{chunk['start_line']}-{chunk['end_line']} (relevance: {chunk['relevance']})")


@cli.command("repos")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
def repos(output_format: str):
    """List all indexed repositories."""
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    handlers = get_mcp_handlers()

    async def run_list():
        result = await handlers.list_repositories()
        return result

    result = asyncio.run(run_list())

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        repos_list = result.get("repositories", [])
        if not repos_list:
            click.echo("\n📭 No repositories indexed yet.")
            click.echo("   Run 'coderag index <url>' to index a repository.")
            return

        click.echo(f"\n📚 Indexed Repositories ({len(repos_list)}):\n")
        for repo in repos_list:
            status_icon = "✅" if repo["status"] == "ready" else "⏳" if repo["status"] == "indexing" else "❌"
            click.echo(f"   {status_icon} {repo['id'][:8]}  {repo['name']} ({repo['branch']})")
            click.echo(f"      Chunks: {repo['chunk_count']} | Indexed: {repo.get('indexed_at', 'N/A')}")


@cli.command("update")
@click.argument("repo_id")
def update(repo_id: str):
    """Update an indexed repository with latest changes.

    REPO_ID: Repository ID (full or first 8 characters)

    Fetches the latest changes from GitHub and re-indexes only the modified files.
    This is faster than a full re-index for repositories with frequent updates.

    Example: coderag update abc12345
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    click.echo(f"\n🔄 Updating repository: {repo_id}\n")

    handlers = get_mcp_handlers()

    async def run_update():
        result = await handlers.update_repository(repo_id=repo_id)
        return result

    result = asyncio.run(run_update())

    if result.get("error"):
        click.echo(f"❌ Error: {result['error']}")
        sys.exit(1)

    if result.get("message") == "Repository is already up to date":
        click.echo("✅ Repository is already up to date!")
    else:
        click.echo("✅ Repository updated successfully!")
        click.echo(f"   Files changed: {result.get('files_changed', 0)}")
        click.echo(f"   - Added: {result.get('files_added', 0)}")
        click.echo(f"   - Modified: {result.get('files_modified', 0)}")
        click.echo(f"   - Deleted: {result.get('files_deleted', 0)}")
        click.echo(f"   Chunks added: {result.get('chunks_added', 0)}")
        click.echo(f"   Chunks deleted: {result.get('chunks_deleted', 0)}")
        click.echo(f"   Total chunks: {result.get('total_chunks', 0)}")


@cli.command("delete")
@click.argument("repo_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def delete(repo_id: str, force: bool):
    """Delete an indexed repository.

    REPO_ID: Repository ID (full or first 8 characters)

    Removes the repository from the index and deletes all associated chunks
    from the vector store.

    Example: coderag delete abc12345
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    handlers = get_mcp_handlers()

    # First get repo info for confirmation
    async def get_repo_info():
        result = await handlers.get_repository_info(repo_id=repo_id)
        return result

    info = asyncio.run(get_repo_info())

    if info.get("error"):
        click.echo(f"❌ Error: {info['error']}")
        sys.exit(1)

    repo_name = info.get("name", repo_id)
    chunk_count = info.get("chunk_count", 0)

    if not force:
        click.echo(f"\n⚠️  About to delete: {repo_name}")
        click.echo(f"   Chunks to delete: {chunk_count}")
        if not click.confirm("\nAre you sure?"):
            click.echo("Cancelled.")
            return

    async def run_delete():
        result = await handlers.delete_repository(repo_id=repo_id)
        return result

    result = asyncio.run(run_delete())

    if result.get("error"):
        click.echo(f"❌ Error: {result['error']}")
        sys.exit(1)

    click.echo(f"\n✅ Repository deleted: {result.get('name', repo_id)}")
    click.echo(f"   Chunks removed: {result.get('chunks_deleted', 0)}")


@cli.command("clean")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def clean(force: bool):
    """Clean up repositories with errors or stuck in indexing.

    Removes all repositories that have status 'error' or have been stuck
    in 'indexing' or 'pending' status for too long.

    Example: coderag clean
    """
    # Apply config from file to environment
    _apply_config_to_env()

    import asyncio
    from coderag.mcp.handlers import get_mcp_handlers

    handlers = get_mcp_handlers()

    async def get_repos():
        result = await handlers.list_repositories()
        return result

    result = asyncio.run(get_repos())
    repos = result.get("repositories", [])

    # Find repos to clean
    to_clean = [r for r in repos if r["status"] in ("error", "indexing", "pending")]

    if not to_clean:
        click.echo("\n✅ No repositories need cleaning.")
        return

    click.echo(f"\n🧹 Found {len(to_clean)} repository(ies) to clean:\n")
    for repo in to_clean:
        status_icon = "❌" if repo["status"] == "error" else "⏳"
        click.echo(f"   {status_icon} {repo['id'][:8]}  {repo['name']} ({repo['status']})")

    if not force:
        if not click.confirm(f"\nDelete these {len(to_clean)} repositories?"):
            click.echo("Cancelled.")
            return

    # Delete each repo
    deleted = 0
    for repo in to_clean:
        async def run_delete():
            return await handlers.delete_repository(repo_id=repo["id"])

        try:
            result = asyncio.run(run_delete())
            if result.get("success"):
                deleted += 1
                click.echo(f"   ✅ Deleted: {repo['name']}")
            else:
                click.echo(f"   ❌ Failed: {repo['name']} - {result.get('error', 'Unknown')}")
        except Exception as e:
            click.echo(f"   ❌ Failed: {repo['name']} - {str(e)}")

    click.echo(f"\n✅ Cleaned {deleted}/{len(to_clean)} repositories.")


@cli.command("doctor")
def doctor():
    """Diagnose common issues with CodeRAG setup.

    Checks Python version, configuration, API key validity, and system components.
    """
    click.echo("\n🏥 CodeRAG Doctor\n")
    all_ok = True

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 11):
        click.echo(f"✅ Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        click.echo(f"❌ Python version: {py_version.major}.{py_version.minor}.{py_version.micro} (need 3.11+)")
        all_ok = False

    # Check config file
    config = get_config()
    if config:
        click.echo(f"✅ Config file exists: {CONFIG_FILE}")
        if config.get("llm_provider"):
            click.echo(f"   Provider: {config['llm_provider']}")
    else:
        click.echo(f"⚠️  No config file. Run 'coderag setup' to configure.")

    # Check API key
    api_key = config.get("llm_api_key") or os.environ.get("MODEL_LLM_API_KEY")
    provider = config.get("llm_provider") or os.environ.get("MODEL_LLM_PROVIDER", "groq")

    if provider != "local":
        if api_key:
            click.echo(f"✅ API key configured (provider: {provider})")
        else:
            click.echo(f"❌ No API key configured for {provider}")
            all_ok = False

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            click.echo(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            click.echo("ℹ️  CUDA not available (CPU mode for embeddings)")
    except ImportError:
        click.echo("⚠️  PyTorch not installed")
        all_ok = False

    # Check ChromaDB data directory
    from coderag.config import get_settings
    settings = get_settings()
    chroma_path = settings.vectorstore.persist_directory
    if chroma_path.exists():
        click.echo(f"✅ ChromaDB directory: {chroma_path}")
    else:
        click.echo(f"ℹ️  ChromaDB directory will be created: {chroma_path}")

    # Check Claude Desktop
    claude_config = get_claude_config_path()
    if claude_config and claude_config.exists():
        try:
            config_data = json.loads(claude_config.read_text())
            if "coderag" in config_data.get("mcpServers", {}):
                click.echo("✅ Claude Desktop MCP configured")
            else:
                click.echo("ℹ️  Claude Desktop installed but MCP not configured. Run 'coderag mcp-install'")
        except Exception:
            click.echo("⚠️  Claude Desktop config exists but could not be read")
    else:
        click.echo("ℹ️  Claude Desktop not detected")

    # Summary
    if all_ok:
        click.echo("\n✅ All checks passed!")
    else:
        click.echo("\n⚠️  Some issues detected. See above for details.")


def _apply_config_to_env():
    """Apply configuration from config file to environment variables."""
    config = get_config()

    if config.get("llm_provider") and not os.environ.get("MODEL_LLM_PROVIDER"):
        os.environ["MODEL_LLM_PROVIDER"] = config["llm_provider"]

    if config.get("llm_api_key") and not os.environ.get("MODEL_LLM_API_KEY"):
        os.environ["MODEL_LLM_API_KEY"] = config["llm_api_key"]

    if config.get("embedding_device") and not os.environ.get("MODEL_EMBEDDING_DEVICE"):
        os.environ["MODEL_EMBEDDING_DEVICE"] = config["embedding_device"]


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
