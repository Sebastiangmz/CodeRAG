"""MCP prompt templates for CodeRAG."""

from mcp.types import PromptMessage, TextContent

from coderag.mcp.server import mcp


@mcp.prompt()
async def analyze_repository(repo_url: str) -> list[PromptMessage]:
    """Guide for comprehensive repository analysis.

    Args:
        repo_url: GitHub repository URL to analyze

    Returns:
        List of prompt messages guiding the analysis workflow
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Please analyze the repository at {repo_url}. Follow these steps:

1. First, use the `index_repository` tool to index the repository:
   - URL: {repo_url}

2. Once indexed, use `get_repository_info` to understand the repository structure:
   - Note the number of files and chunks indexed
   - Review the list of indexed files

3. Use `query_code` to answer these questions:
   - What is the main purpose of this codebase?
   - What are the key components or modules?
   - What design patterns are used?
   - What external dependencies does it have?

4. Use `search_code` to find:
   - Entry points (main functions, CLI handlers)
   - Configuration handling
   - Core business logic

5. Provide a comprehensive summary including:
   - Purpose and functionality
   - Architecture overview
   - Key components
   - Notable patterns or practices
   - Potential areas for improvement
""",
            ),
        )
    ]


@mcp.prompt()
async def find_implementation(repo_id: str, feature: str) -> list[PromptMessage]:
    """Guide for finding feature implementations.

    Args:
        repo_id: Repository ID to search in
        feature: Feature or functionality to find

    Returns:
        List of prompt messages guiding the search workflow
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Please find the implementation of "{feature}" in repository {repo_id}. Follow these steps:

1. Use `search_code` to find relevant code:
   - Query: "{feature}"
   - Try different search terms if initial results aren't helpful

2. For each relevant result, use `query_code` to understand:
   - How is this feature implemented?
   - What are the key functions/classes involved?
   - What is the data flow?

3. Trace the implementation:
   - Find the entry point
   - Follow the call chain
   - Identify helper functions and utilities

4. Provide a detailed explanation:
   - Location of the implementation (files and line numbers)
   - Key components and their roles
   - How data flows through the system
   - Any notable patterns or design decisions
""",
            ),
        )
    ]


@mcp.prompt()
async def code_review(repo_id: str, focus_area: str = "") -> list[PromptMessage]:
    """Guide for performing code reviews.

    Args:
        repo_id: Repository ID to review
        focus_area: Optional specific area to focus on (e.g., "security", "performance")

    Returns:
        List of prompt messages guiding the review workflow
    """
    focus_text = f' with focus on "{focus_area}"' if focus_area else ""

    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Please perform a code review of repository {repo_id}{focus_text}. Follow these steps:

1. Use `get_repository_info` to understand the repository structure

2. Use `search_code` to find key areas to review:
   - Entry points and main functions
   - Error handling patterns
   - Data validation
   - Security-sensitive code (if applicable)

3. For each area, use `query_code` to analyze:
   - Code quality and readability
   - Error handling completeness
   - Security considerations
   - Performance implications
   - Test coverage (if tests are indexed)

4. Check for common issues:
   - Hardcoded credentials or secrets
   - SQL injection vulnerabilities
   - Input validation gaps
   - Resource leaks
   - Race conditions

5. Provide a structured review:
   - Summary of findings
   - Critical issues (if any)
   - Recommendations for improvement
   - Positive observations
   - Priority of fixes
""",
            ),
        )
    ]
