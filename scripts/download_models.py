#!/usr/bin/env python3
"""Download required models for CodeRAG."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def download_models():
    """Download Qwen2.5-Coder and nomic-embed-text models."""
    models = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "nomic-ai/nomic-embed-text-v1.5",
    ]

    cache_dir = Path("./data/hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        print(f"\nDownloading {model_name}...")
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
            )
            print(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            sys.exit(1)

    print("\n✓ All models downloaded successfully!")
    print(f"Models cached in: {cache_dir.absolute()}")


if __name__ == "__main__":
    download_models()
