# Feature Specification: CodeRAG Q&A System

**Feature Branch**: `001-coderag-qa-system`
**Created**: 2025-12-15
**Status**: Draft
**Input**: Sistema RAG grounded para Q&A sobre repositorios de código con citas verificables, chunking semántico Python, modelos locales, interfaz web, y deployment containerizado.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Index a GitHub Repository (Priority: P1)

As a developer, I want to index a public GitHub repository so that I can later ask questions about its code and documentation.

**Why this priority**: Without indexing, there is no knowledge base to query. This is the foundational capability that enables all other features.

**Independent Test**: Can be fully tested by providing a GitHub URL and verifying that the repository is cloned, processed into chunks, and stored in a searchable index. Success is confirmed when the system reports the number of chunks indexed.

**Acceptance Scenarios**:

1. **Given** the system is running, **When** I enter a valid public GitHub URL (e.g., `https://github.com/owner/repo`) and click "Index", **Then** the system clones the repository and displays a progress indicator.

2. **Given** a repository is being indexed, **When** the indexing completes, **Then** the system displays a success message with the total number of chunks indexed (e.g., "342 chunks indexed").

3. **Given** I enter an invalid URL or a private repository URL, **When** I click "Index", **Then** the system displays a clear error message explaining the issue.

4. **Given** I have previously indexed a repository, **When** I index the same repository again, **Then** the system updates the existing index with any new or changed content.

---

### User Story 2 - Ask Questions About Code (Priority: P1)

As a developer, I want to ask natural language questions about an indexed repository and receive accurate answers with citations to the source code.

**Why this priority**: This is the core value proposition - answering questions about code with verifiable citations. Equally critical as indexing since without Q&A, indexing has no purpose.

**Independent Test**: Can be tested by indexing a known repository, asking questions with known answers, and verifying that responses include correct citations pointing to the actual source locations.

**Acceptance Scenarios**:

1. **Given** a repository has been indexed, **When** I ask "Where is the function X defined?", **Then** the system responds with the file path, line numbers, and a brief explanation, formatted as citations like `[src/module.py:45-78]`.

2. **Given** a repository has been indexed, **When** I ask "What does function Y do?", **Then** the system provides an explanation based on the code and documentation, with citations to the relevant source files.

3. **Given** a repository has been indexed, **When** I ask a question about something not present in the repository, **Then** the system responds with "I could not find information about this in the indexed repository" instead of making up an answer.

4. **Given** a repository has been indexed, **When** I ask an ambiguous question, **Then** the system asks for clarification or explains the ambiguity before attempting to answer.

---

### User Story 3 - View Retrieved Evidence (Priority: P2)

As a developer, I want to see the code chunks that were used to generate an answer so that I can verify the response accuracy.

**Why this priority**: Transparency builds trust. Showing evidence allows users to validate that answers are grounded in actual code, not hallucinated.

**Independent Test**: Can be tested by asking a question and verifying that the response includes a visible "Evidence" section showing the retrieved chunks with their source locations.

**Acceptance Scenarios**:

1. **Given** I have asked a question, **When** the system generates a response, **Then** I can see an "Evidence" section listing all code chunks used, including file paths and line ranges.

2. **Given** the evidence section is displayed, **When** I review the chunks, **Then** each chunk shows its relevance score and the actual code/text content.

3. **Given** a response includes multiple citations, **When** I view the evidence, **Then** the chunks are ordered by relevance to my question.

---

### User Story 4 - Track Indexing Progress (Priority: P2)

As a developer, I want to see real-time progress while indexing a repository so that I know how long the process will take.

**Why this priority**: Large repositories can take several minutes to index. Progress feedback prevents user frustration and abandonment.

**Independent Test**: Can be tested by indexing a medium-sized repository and observing the progress indicator updating through distinct stages.

**Acceptance Scenarios**:

1. **Given** I have started indexing a repository, **When** the process is running, **Then** I see a progress bar with percentage and current stage (e.g., "Cloning repository... 10%", "Processing chunks... 65%").

2. **Given** indexing is in progress, **When** each stage completes, **Then** the progress indicator updates to reflect the new stage and percentage.

3. **Given** indexing encounters a non-fatal issue (e.g., unparseable file), **When** the process continues, **Then** the system logs a warning but does not stop the entire indexing process.

---

### User Story 5 - Configure Indexing Options (Priority: P3)

As a developer, I want to customize indexing options (branch, file filters, retrieval parameters) so that I can focus on relevant parts of a repository.

**Why this priority**: While defaults work for most cases, power users need flexibility to handle specific use cases like indexing only documentation or excluding test files.

**Independent Test**: Can be tested by indexing with custom options (e.g., specific branch, exclude patterns) and verifying that only the expected files are indexed.

**Acceptance Scenarios**:

1. **Given** I am about to index a repository, **When** I expand "Advanced Options", **Then** I can specify a branch name other than the default.

2. **Given** I am configuring indexing options, **When** I set file inclusion/exclusion patterns, **Then** only matching files are processed during indexing.

3. **Given** I am configuring options, **When** I adjust the "Top-K" parameter for retrieval, **Then** subsequent queries return the specified number of most relevant chunks.

---

### User Story 6 - Persist Indexed Repositories (Priority: P3)

As a developer, I want my indexed repositories to persist between sessions so that I do not have to re-index every time I restart the system.

**Why this priority**: Re-indexing large repositories is time-consuming. Persistence improves user experience by maintaining state across sessions.

**Independent Test**: Can be tested by indexing a repository, restarting the system, and verifying that the repository remains available for queries without re-indexing.

**Acceptance Scenarios**:

1. **Given** I have indexed a repository and then restart the system, **When** I open the application, **Then** the previously indexed repository is still available for querying.

2. **Given** multiple repositories have been indexed, **When** I view the repository list, **Then** I see all indexed repositories with their index date and chunk counts.

3. **Given** I want to remove an indexed repository, **When** I delete it from the list, **Then** the associated index data is removed and storage is freed.

---

### Edge Cases

- **Empty repository**: When indexing a repository with no code files (only binary files or empty), the system displays a message explaining that no indexable content was found.
- **Very large files**: Files exceeding the maximum size limit (500KB default) are skipped with a warning logged.
- **Encoding issues**: Files with unsupported encodings are skipped gracefully without crashing the indexing process.
- **Network interruption**: If network is lost during repository cloning, the system displays an error and allows retry.
- **Insufficient resources**: If the system runs out of memory during model loading, it displays a clear error suggesting a lighter model alternative.
- **Repository access revoked**: If a previously public repository becomes private, queries fail gracefully with an explanation.

## Requirements *(mandatory)*

### Functional Requirements

#### Repository Indexing

- **FR-001**: System MUST accept public GitHub repository URLs in the format `https://github.com/owner/repo`.
- **FR-002**: System MUST validate that the provided URL is a valid, accessible GitHub repository before attempting to clone.
- **FR-003**: System MUST clone the specified branch (default: main/master) of the repository to local storage.
- **FR-004**: System MUST filter files based on configurable inclusion/exclusion patterns, excluding binary files, lock files, and vendored dependencies by default.
- **FR-005**: System MUST divide code files into semantic chunks, preserving logical units (functions, classes) for supported languages.
- **FR-006**: System MUST generate vector embeddings for each chunk and store them in a searchable index.
- **FR-007**: System MUST display indexing progress with stage information and percentage completion.
- **FR-008**: System MUST report the total number of chunks indexed upon completion.

#### Question Answering

- **FR-009**: System MUST accept natural language questions about the currently selected repository (queries do not span multiple repositories).
- **FR-010**: System MUST retrieve the most relevant chunks from the index based on semantic similarity to the question.
- **FR-011**: System MUST generate responses that are grounded in the retrieved chunks, not fabricated.
- **FR-012**: System MUST include citations in format `[file_path:start_line-end_line]` for every factual claim in the response.
- **FR-013**: System MUST refuse to answer when retrieved chunks do not contain sufficient information, responding with "I could not find information about this in the indexed repository."
- **FR-014**: System MUST display the retrieved chunks as evidence alongside the generated response.

#### User Interface

- **FR-015**: System MUST provide a web-based interface accessible via browser.
- **FR-016**: System MUST provide separate views for indexing repositories and asking questions.
- **FR-016b**: System MUST allow users to select which indexed repository to query when multiple repositories exist.
- **FR-017**: System MUST display clear error messages when operations fail.
- **FR-018**: System MUST allow configuration of advanced options (branch, filters, top-k) through an expandable panel.

#### Persistence

- **FR-019**: System MUST persist indexed repository data across application restarts.
- **FR-020**: System MUST allow users to view a list of all indexed repositories.
- **FR-021**: System MUST allow users to delete indexed repositories and free associated storage.

#### Security & Privacy

- **FR-022**: System MUST NOT index files likely to contain secrets (`.env`, credentials files, API keys).
- **FR-023**: System MUST sanitize all user inputs to prevent injection attacks.
- **FR-024**: System MUST operate entirely locally without sending data to external services.

#### Observability

- **FR-025**: System MUST provide structured logging with info, warning, and error levels to console and/or file.
- **FR-026**: System MUST log key operations: indexing start/completion, query processing, errors, and skipped files.

### Key Entities

- **Repository**: Represents an indexed GitHub repository. Attributes: URL, branch, clone date, status, chunk count.
- **Chunk**: A semantic unit of code or documentation. Attributes: content, file path, start line, end line, chunk type (function/class/text), embedding vector.
- **Query**: A user's question about a repository. Attributes: question text, timestamp, associated repository.
- **Response**: The system's answer to a query. Attributes: answer text, citations list, retrieved chunks, confidence score, grounded flag.
- **Citation**: A reference to source code. Attributes: file path, start line, end line.

### Assumptions

- Users have stable internet connectivity for cloning repositories.
- Target repositories are public (no authentication required for MVP).
- Host machine has an NVIDIA GPU with at least 8GB VRAM for optimal performance.
- Users understand basic Git concepts (branches, repositories).

## Clarifications

### Session 2025-12-15

- Q: How do queries behave when multiple repositories are indexed? → A: Queries target only the selected/active repository.
- Q: What level of logging/observability is required? → A: Structured logging to console/file (info, warning, error levels).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can index a typical repository (1000 files) in under 5 minutes.
- **SC-002**: Users receive answers to questions in under 10 seconds.
- **SC-003**: 90% of responses to "where is X defined?" questions correctly identify the file and line range.
- **SC-004**: System successfully abstains from answering (instead of hallucinating) for 95% of questions about content not in the repository.
- **SC-005**: All citations in responses point to actual locations that contain relevant content when verified.
- **SC-006**: System operates successfully on a machine with 8GB GPU memory without crashing.
- **SC-007**: Users can restart the application and access previously indexed repositories without re-indexing.
- **SC-008**: 80% of users can successfully index a repository and ask a question on their first attempt without documentation.
