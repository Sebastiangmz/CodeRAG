"""Hugging Face Spaces entry point for CodeRAG.

This file is used by Hugging Face Spaces to launch the Gradio demo.
It's configured to work without GPU (embeddings on CPU, LLM via Groq).
"""

import os

# Configure for HF Spaces environment
os.environ.setdefault("MODEL_LLM_PROVIDER", "groq")
os.environ.setdefault("MODEL_EMBEDDING_DEVICE", "cpu")

# Use HF Spaces secrets for API key
if "GROQ_API_KEY" in os.environ and "MODEL_LLM_API_KEY" not in os.environ:
    os.environ["MODEL_LLM_API_KEY"] = os.environ["GROQ_API_KEY"]

# Import and launch the Gradio app
from coderag.ui.app import create_gradio_app

demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch()
