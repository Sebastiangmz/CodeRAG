"""Generation module: LLM inference and response generation with citations."""

from coderag.generation.generator import ResponseGenerator
from coderag.generation.prompts import SYSTEM_PROMPT, build_prompt
from coderag.generation.citations import CitationParser

__all__ = ["ResponseGenerator", "SYSTEM_PROMPT", "build_prompt", "CitationParser"]
