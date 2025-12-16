#!/usr/bin/env python3
"""Validate CodeRAG setup and dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_imports():
    """Check if all modules can be imported."""
    print("🔍 Checking module imports...")

    modules_to_test = [
        ("coderag.config", "get_settings"),
        ("coderag.logging", "get_logger"),
        ("coderag.models.repository", "Repository"),
        ("coderag.models.document", "Document"),
        ("coderag.models.chunk", "Chunk"),
        ("coderag.models.response", "Response"),
        ("coderag.ingestion.validator", "GitHubURLValidator"),
        ("coderag.ingestion.loader", "RepositoryLoader"),
        ("coderag.ingestion.filter", "FileFilter"),
        ("coderag.ingestion.chunker", "CodeChunker"),
        ("coderag.indexing.embeddings", "EmbeddingGenerator"),
        ("coderag.indexing.vectorstore", "VectorStore"),
        ("coderag.retrieval.retriever", "Retriever"),
        ("coderag.generation.prompts", "SYSTEM_PROMPT"),
        ("coderag.generation.citations", "CitationParser"),
        ("coderag.generation.generator", "ResponseGenerator"),
        ("coderag.api.schemas", "QueryRequest"),
        ("coderag.api.routes", "router"),
    ]

    failed = []
    for module_name, attr_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            getattr(module, attr_name)
            print(f"  ✓ {module_name}.{attr_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{attr_name}: {e}")
            failed.append((module_name, str(e)))

    if failed:
        print(f"\n❌ {len(failed)} imports failed")
        return False

    print(f"\n✅ All {len(modules_to_test)} imports successful")
    return True


def check_dependencies():
    """Check if critical dependencies are installed."""
    print("\n🔍 Checking dependencies...")

    dependencies = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "gradio",
        "transformers",
        "torch",
        "sentence_transformers",
        "chromadb",
        "git",
        "structlog",
    ]

    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError as e:
            print(f"  ✗ {dep}: Not installed")
            failed.append(dep)

    if failed:
        print(f"\n❌ {len(failed)} dependencies missing: {', '.join(failed)}")
        return False

    print(f"\n✅ All {len(dependencies)} dependencies installed")
    return True


def check_config():
    """Check if configuration loads correctly."""
    print("\n🔍 Checking configuration...")

    try:
        from coderag.config import get_settings

        settings = get_settings()
        print(f"  ✓ App name: {settings.app_name}")
        print(f"  ✓ Version: {settings.app_version}")
        print(f"  ✓ LLM model: {settings.models.llm_name}")
        print(f"  ✓ Embedding model: {settings.models.embedding_name}")
        print(f"  ✓ Data dir: {settings.data_dir}")

        print("\n✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        return False


def check_directories():
    """Check if required directories are created."""
    print("\n🔍 Checking directories...")

    from coderag.config import get_settings
    settings = get_settings()

    settings.ensure_directories()

    dirs_to_check = [
        settings.data_dir,
        settings.vectorstore.persist_directory,
        settings.ingestion.repos_cache_dir,
    ]

    for directory in dirs_to_check:
        if directory.exists():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - Missing")
            return False

    print("\n✅ All directories exist")
    return True


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("CodeRAG Setup Validation")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Directories", check_directories),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n🎉 All validation checks passed!")
        return 0
    else:
        print("\n⚠️  Some validation checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
