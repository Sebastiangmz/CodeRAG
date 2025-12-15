"""System prompts for grounded code Q&A."""

SYSTEM_PROMPT = """You are a code assistant that answers questions about a repository.

STRICT RULES:
1. Only answer based on the provided code chunks below
2. Every factual claim MUST include a citation [file_path:start_line-end_line]
3. If information is NOT in the chunks, respond: "I could not find information about this in the indexed repository"
4. NEVER invent code, functions, files, or behaviors not in the chunks
5. Be concise and structured
6. When multiple relevant chunks exist, cite all of them

CITATION FORMAT: [file_path:start_line-end_line]
Example: [src/auth.py:45-78]

RESPONSE FORMAT:
- Start with a direct answer to the question
- Include citations inline with your statements
- If showing code, quote it exactly from the chunks
- End with a brief summary if the answer is complex"""


def build_prompt(question: str, context: str) -> str:
    """Build the full prompt with context and question.

    Args:
        question: User's question
        context: Retrieved code chunks formatted as context

    Returns:
        Complete prompt for the LLM
    """
    return f"""Based on the following code chunks from the repository, answer the question.

## Retrieved Code Chunks

{context}

## Question

{question}

## Answer

"""


def build_no_context_response() -> str:
    """Response when no relevant context is found."""
    return "I could not find information about this in the indexed repository."


def build_clarification_prompt(question: str, ambiguities: list[str]) -> str:
    """Build prompt asking for clarification."""
    ambiguity_list = "\n".join(f"- {a}" for a in ambiguities)
    return f"""Your question "{question}" is ambiguous. Could you clarify:

{ambiguity_list}

Please provide more specific details so I can give you an accurate answer."""
