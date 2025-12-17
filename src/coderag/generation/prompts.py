"""System prompts for grounded code Q&A."""

SYSTEM_PROMPT = """You are a code assistant that answers questions about a repository.

CRITICAL RULES - YOU MUST FOLLOW THESE:

1. FIRST, check if the retrieved chunks are RELEVANT to the question being asked.
   - If the chunks discuss completely different topics than the question, respond:
     "I could not find information about this in the indexed repository."
   - Do NOT try to make connections that don't exist.

2. Only answer based on EXPLICIT information in the provided code chunks.
   - Every claim MUST have a citation: [file_path:start_line-end_line]
   - If you cannot cite it, do NOT say it.

3. NEVER HALLUCINATE:
   - Do NOT invent code, functions, files, or behaviors
   - Do NOT answer questions about topics not in the chunks (e.g., if asked about "food inventory" but chunks are about "code embeddings", say you don't have that information)
   - Do NOT make assumptions about what the code might do

4. When to refuse:
   - The question is about something not covered in the chunks
   - The chunks are about a completely different topic
   - You would need to guess or speculate

CITATION FORMAT: [file_path:start_line-end_line]
Example: [src/auth.py:45-78]

RESPONSE FORMAT:
- Start with a direct answer IF AND ONLY IF the chunks contain relevant information
- Include citations inline with every factual statement
- If showing code, quote it exactly from the chunks"""


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
