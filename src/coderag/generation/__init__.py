"""Generation module: LLM inference and response generation with citations."""

from coderag.generation.citations import CitationParser, CitationVerifier
from coderag.generation.prompts import SYSTEM_PROMPT, build_prompt

__all__ = ["CitationParser", "CitationVerifier", "ResponseGenerator", "SYSTEM_PROMPT", "build_prompt"]


def __getattr__(name: str) -> object:
    if name == "ResponseGenerator":
        from coderag.generation.generator import ResponseGenerator

        return ResponseGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
