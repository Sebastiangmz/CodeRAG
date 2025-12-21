# CodeRAG MCP Server - Especificacion Completa de Implementacion

## 1. CONTEXTO DEL PROYECTO EXISTENTE

### 1.1 Arquitectura Actual del RAG

El proyecto CodeRAG es un sistema RAG (Retrieval Augmented Generation) para codigo con la siguiente estructura:

```
src/coderag/
├── main.py              # Entry point FastAPI + Gradio UI
├── config.py            # Configuracion centralizada (pydantic-settings)
├── logging.py           # Structured logging con structlog
├── api/
│   ├── routes.py        # REST API endpoints existentes
│   └── schemas.py       # Pydantic schemas para API
├── ui/
│   ├── app.py           # Gradio UI
│   └── handlers.py      # UIHandlers - LOGICA PRINCIPAL DE NEGOCIO
├── ingestion/
│   ├── validator.py     # GitHubURLValidator - valida URLs
│   ├── loader.py        # RepositoryLoader - clona repos
│   ├── filter.py        # FileFilter - filtra archivos
│   └── chunker.py       # CodeChunker - tree-sitter/texto
├── indexing/
│   ├── embeddings.py    # EmbeddingGenerator - nomic-embed-text
│   └── vectorstore.py   # VectorStore - ChromaDB
├── retrieval/
│   └── retriever.py     # Retriever - busqueda semantica
├── generation/
│   ├── generator.py     # ResponseGenerator - LLM local/remoto
│   ├── prompts.py       # System prompts
│   └── citations.py     # CitationParser
└── models/
    ├── chunk.py         # Chunk, ChunkMetadata, ChunkType
    ├── document.py      # Document
    ├── query.py         # Query
    ├── response.py      # Response, Citation, RetrievedChunk
    └── repository.py    # Repository, RepositoryStatus
```

### 1.2 Componentes Clave a Reutilizar

#### UIHandlers (src/coderag/ui/handlers.py)
Clase principal que orquesta toda la logica de negocio:
- `index_repository(url, branch, include_patterns, exclude_patterns)` -> Iterator[str] (streaming)
- `index_repository_incremental(repo_id)` -> str
- `ask_question(repo_id, question, top_k)` -> tuple[answer, evidence, status]
- `get_repositories()` -> lista de repos disponibles
- `delete_repository(repo_id)` -> tuple[status, table]

#### ResponseGenerator (src/coderag/generation/generator.py)
Genera respuestas con citas:
- Soporta providers: local, openai, groq, anthropic, openrouter, together
- Metodo `generate(query: Query) -> Response`

#### VectorStore (src/coderag/indexing/vectorstore.py)
Operaciones ChromaDB:
- `query(embedding, repo_id, top_k, threshold)` -> chunks con scores
- `get_all_repo_ids()` -> lista de repo_ids
- `get_repo_chunk_count(repo_id)` -> int
- `get_indexed_files(repo_id)` -> set de file paths

### 1.3 Stack Tecnologico
- Python 3.11+
- FastAPI + Uvicorn
- ChromaDB (persistente)
- nomic-embed-text-v1.5 (sentence-transformers)
- Qwen2.5-Coder-3B (local) o APIs remotas
- Tree-sitter para parsing Python
- GitPython para clonacion

### 1.4 Configuracion Existente (config.py)
```python
Settings
├── ModelSettings      # llm_provider, llm_name, embedding_name, etc.
├── VectorStoreSettings # persist_directory, collection_name
├── IngestionSettings   # chunk_size, max_files, batch_size, patterns
├── RetrievalSettings   # default_top_k, similarity_threshold
└── ServerSettings      # host, port
```

---

## 2. ESPECIFICACION DEL MCP SERVER

### 2.1 Objetivo
Exponer las capacidades del RAG como un servidor MCP (Model Context Protocol) que permita a clientes como Claude Desktop, Cursor, o cualquier cliente MCP compatible:
1. Indexar repositorios de GitHub
2. Hacer preguntas sobre el codigo con respuestas citadas
3. Listar y gestionar repositorios indexados
4. Acceder a chunks de codigo como recursos

### 2.2 Transporte
- **Primario**: Streamable HTTP montado en `/mcp` sobre la app FastAPI existente
- **Secundario**: stdio para compatibilidad con Claude Desktop via `docker exec`

### 2.3 Dependencias a Agregar
```toml
# En pyproject.toml
dependencies = [
    # ... existentes ...
    "mcp>=1.0.0",
]
```

---

## 3. HERRAMIENTAS MCP (Tools)

### 3.1 Tool: index_repository

**Proposito**: Indexar un repositorio de GitHub para consultas posteriores.

**Schema de Entrada**:
```json
{
  "name": "index_repository",
  "description": "Indexa un repositorio de GitHub para poder hacer preguntas sobre su codigo. Clona el repositorio, extrae chunks semanticos del codigo, genera embeddings y los almacena en la base de datos vectorial.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "URL del repositorio de GitHub. Formatos validos: https://github.com/owner/repo, owner/repo"
      },
      "branch": {
        "type": "string",
        "description": "Rama a indexar. Si no se especifica, usa la rama por defecto del repositorio (main/master).",
        "default": ""
      },
      "include_patterns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Patrones glob para incluir archivos (ej: ['*.py', '*.js']). Si esta vacio, usa los patrones por defecto.",
        "default": []
      },
      "exclude_patterns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Patrones glob para excluir archivos (ej: ['**/tests/**']). Si esta vacio, usa los patrones por defecto.",
        "default": []
      }
    },
    "required": ["url"]
  }
}
```

**Schema de Salida**:
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "repo_id": {"type": "string", "description": "ID unico del repositorio indexado"},
    "repo_name": {"type": "string", "description": "Nombre completo owner/repo"},
    "files_processed": {"type": "integer"},
    "chunks_indexed": {"type": "integer"},
    "message": {"type": "string"}
  }
}
```

**Implementacion** (usar UIHandlers.index_repository pero sin streaming):
```python
@mcp.tool()
async def index_repository(
    url: str,
    branch: str = "",
    include_patterns: list[str] = [],
    exclude_patterns: list[str] = []
) -> dict:
    """
    Indexa un repositorio de GitHub para consultas de codigo.

    Args:
        url: URL del repositorio (https://github.com/owner/repo o owner/repo)
        branch: Rama a indexar (opcional, usa default del repo)
        include_patterns: Patrones glob para incluir (opcional)
        exclude_patterns: Patrones glob para excluir (opcional)

    Returns:
        Diccionario con repo_id, estadisticas y estado
    """
    # Implementacion detallada en seccion 5
```

### 3.2 Tool: query_code

**Proposito**: Hacer preguntas sobre el codigo de un repositorio indexado.

**Schema de Entrada**:
```json
{
  "name": "query_code",
  "description": "Hace una pregunta sobre el codigo de un repositorio indexado y devuelve una respuesta fundamentada con citas exactas al codigo fuente. Cada afirmacion incluye referencias [archivo:linea-linea].",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_id": {
        "type": "string",
        "description": "ID del repositorio a consultar. Obtener de list_repositories o del resultado de index_repository."
      },
      "question": {
        "type": "string",
        "description": "Pregunta sobre el codigo. Ejemplos: 'Donde esta definida la funcion X?', 'Como funciona la autenticacion?', 'Que hace la clase Y?'"
      },
      "top_k": {
        "type": "integer",
        "description": "Numero de chunks de codigo a recuperar para contexto (1-20).",
        "default": 5,
        "minimum": 1,
        "maximum": 20
      }
    },
    "required": ["repo_id", "question"]
  }
}
```

**Schema de Salida**:
```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "Respuesta a la pregunta con citas en formato [archivo:linea-linea]"
    },
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "start_line": {"type": "integer"},
          "end_line": {"type": "integer"}
        }
      },
      "description": "Lista de citas extraidas de la respuesta"
    },
    "evidence": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "start_line": {"type": "integer"},
          "end_line": {"type": "integer"},
          "content": {"type": "string"},
          "relevance_score": {"type": "number"},
          "chunk_type": {"type": "string"},
          "name": {"type": "string"}
        }
      },
      "description": "Chunks de codigo recuperados como evidencia"
    },
    "grounded": {
      "type": "boolean",
      "description": "True si la respuesta tiene citas verificables"
    }
  }
}
```

**Implementacion**:
```python
@mcp.tool()
async def query_code(
    repo_id: str,
    question: str,
    top_k: int = 5
) -> dict:
    """
    Pregunta sobre el codigo de un repositorio indexado.

    Args:
        repo_id: ID del repositorio a consultar
        question: Pregunta sobre el codigo
        top_k: Numero de chunks a recuperar (1-20)

    Returns:
        Respuesta con citas y evidencia
    """
    # Implementacion detallada en seccion 5
```

### 3.3 Tool: list_repositories

**Proposito**: Listar todos los repositorios indexados disponibles.

**Schema de Entrada**:
```json
{
  "name": "list_repositories",
  "description": "Lista todos los repositorios de GitHub que han sido indexados y estan disponibles para consultas.",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**Schema de Salida**:
```json
{
  "type": "object",
  "properties": {
    "repositories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "repo_id": {"type": "string"},
          "name": {"type": "string", "description": "owner/repo"},
          "url": {"type": "string"},
          "branch": {"type": "string"},
          "chunk_count": {"type": "integer"},
          "status": {"type": "string", "enum": ["ready", "indexing", "error", "pending"]},
          "indexed_at": {"type": "string", "format": "date-time"}
        }
      }
    },
    "count": {"type": "integer"}
  }
}
```

### 3.4 Tool: get_repository_info

**Proposito**: Obtener informacion detallada de un repositorio especifico.

**Schema de Entrada**:
```json
{
  "name": "get_repository_info",
  "description": "Obtiene informacion detallada de un repositorio indexado, incluyendo estadisticas y archivos indexados.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_id": {
        "type": "string",
        "description": "ID del repositorio (completo o primeros 8 caracteres)"
      }
    },
    "required": ["repo_id"]
  }
}
```

**Schema de Salida**:
```json
{
  "type": "object",
  "properties": {
    "repo_id": {"type": "string"},
    "name": {"type": "string"},
    "url": {"type": "string"},
    "branch": {"type": "string"},
    "chunk_count": {"type": "integer"},
    "status": {"type": "string"},
    "indexed_at": {"type": "string"},
    "indexed_files": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Lista de archivos indexados"
    },
    "last_commit": {"type": "string"}
  }
}
```

### 3.5 Tool: delete_repository

**Proposito**: Eliminar un repositorio indexado.

**Schema de Entrada**:
```json
{
  "name": "delete_repository",
  "description": "Elimina un repositorio indexado y todos sus datos asociados (chunks, embeddings, cache).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_id": {
        "type": "string",
        "description": "ID del repositorio a eliminar"
      }
    },
    "required": ["repo_id"]
  }
}
```

### 3.6 Tool: update_repository

**Proposito**: Actualizar un repositorio con cambios incrementales.

**Schema de Entrada**:
```json
{
  "name": "update_repository",
  "description": "Actualiza un repositorio existente de forma incremental, solo procesando archivos modificados desde la ultima indexacion.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_id": {
        "type": "string",
        "description": "ID del repositorio a actualizar"
      }
    },
    "required": ["repo_id"]
  }
}
```

### 3.7 Tool: search_code

**Proposito**: Busqueda semantica directa sin generacion de respuesta LLM.

**Schema de Entrada**:
```json
{
  "name": "search_code",
  "description": "Busca fragmentos de codigo semanticamente similares a una consulta, sin generar una respuesta elaborada. Util para exploracion rapida.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_id": {
        "type": "string",
        "description": "ID del repositorio"
      },
      "query": {
        "type": "string",
        "description": "Texto de busqueda semantica"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      },
      "file_filter": {
        "type": "string",
        "description": "Filtrar por patron de archivo (ej: '*.py')"
      },
      "chunk_type": {
        "type": "string",
        "enum": ["function", "class", "method", "text", "all"],
        "default": "all"
      }
    },
    "required": ["repo_id", "query"]
  }
}
```

---

## 4. RECURSOS MCP (Resources)

### 4.1 Resource: repository://

**URI Template**: `repository://{repo_id}`

**Descripcion**: Acceso a los metadatos de un repositorio indexado.

```python
@mcp.resource("repository://{repo_id}")
async def get_repository_resource(repo_id: str) -> Resource:
    """Retorna metadatos del repositorio como recurso."""
```

### 4.2 Resource: code://

**URI Template**: `code://{repo_id}/{file_path}`

**Descripcion**: Acceso al contenido de un archivo indexado.

```python
@mcp.resource("code://{repo_id}/{file_path}")
async def get_code_resource(repo_id: str, file_path: str) -> Resource:
    """Retorna el contenido de un archivo del repositorio."""
```

### 4.3 Resource: chunk://

**URI Template**: `chunk://{chunk_id}`

**Descripcion**: Acceso a un chunk especifico por su ID.

```python
@mcp.resource("chunk://{chunk_id}")
async def get_chunk_resource(chunk_id: str) -> Resource:
    """Retorna un chunk especifico con sus metadatos."""
```

---

## 5. IMPLEMENTACION DETALLADA

### 5.1 Estructura de Archivos a Crear

```
src/coderag/
├── mcp/
│   ├── __init__.py
│   ├── server.py        # Servidor MCP principal (FastMCP)
│   ├── tools.py         # Definicion de herramientas
│   ├── resources.py     # Definicion de recursos
│   ├── prompts.py       # Prompts MCP predefinidos
│   └── handlers.py      # MCPHandlers - adaptador de UIHandlers
└── main.py              # Modificar para montar MCP
```

### 5.2 Archivo: src/coderag/mcp/__init__.py

```python
"""MCP Server module for CodeRAG."""

from coderag.mcp.server import create_mcp_server, mcp

__all__ = ["create_mcp_server", "mcp"]
```

### 5.3 Archivo: src/coderag/mcp/server.py

```python
"""MCP Server configuration and setup."""

from mcp.server.fastmcp import FastMCP

from coderag.config import get_settings
from coderag.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Crear instancia del servidor MCP
mcp = FastMCP(
    name="CodeRAG",
    version=settings.app_version,
    description="RAG-based Q&A system for code repositories with verifiable citations",
)


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with all tools and resources."""
    # Importar para registrar tools y resources
    from coderag.mcp import tools  # noqa: F401
    from coderag.mcp import resources  # noqa: F401
    from coderag.mcp import prompts  # noqa: F401

    logger.info("MCP server created", tools=len(mcp._tools), resources=len(mcp._resources))
    return mcp
```

### 5.4 Archivo: src/coderag/mcp/handlers.py

```python
"""MCP-specific handlers that wrap UIHandlers for non-streaming use."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from coderag.config import get_settings
from coderag.generation.generator import ResponseGenerator
from coderag.indexing.embeddings import EmbeddingGenerator
from coderag.indexing.vectorstore import VectorStore
from coderag.ingestion.chunker import CodeChunker
from coderag.ingestion.filter import FileFilter
from coderag.ingestion.loader import RepositoryLoader
from coderag.ingestion.validator import GitHubURLValidator, ValidationError
from coderag.logging import get_logger
from coderag.models.document import Document
from coderag.models.query import Query
from coderag.models.repository import Repository, RepositoryStatus

logger = get_logger(__name__)


class MCPHandlers:
    """
    Handlers optimizados para MCP (sin streaming, respuestas completas).
    Reutiliza la logica de UIHandlers pero adaptada para el protocolo MCP.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = GitHubURLValidator()
        self.loader = RepositoryLoader()
        self.filter = FileFilter()
        self.chunker = CodeChunker()
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()
        self.generator: Optional[ResponseGenerator] = None

        # Repository metadata storage
        self.repos_file = self.settings.data_dir / "repositories.json"
        self.repositories: dict[str, Repository] = self._load_repositories()

    def _load_repositories(self) -> dict[str, Repository]:
        """Load repositories from disk."""
        if self.repos_file.exists():
            try:
                data = json.loads(self.repos_file.read_text())
                return {r["id"]: Repository.from_dict(r) for r in data}
            except Exception as e:
                logger.error("Failed to load repositories", error=str(e))
        return {}

    def _save_repositories(self) -> None:
        """Save repositories to disk."""
        self.repos_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.repositories.values()]
        self.repos_file.write_text(json.dumps(data, indent=2))

    def _find_repository(self, repo_id: str) -> Optional[Repository]:
        """Find repository by full or partial ID."""
        for rid, repo in self.repositories.items():
            if rid == repo_id or rid.startswith(repo_id):
                return repo
        return None

    async def index_repository(
        self,
        url: str,
        branch: str = "",
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
    ) -> dict:
        """
        Index a GitHub repository (non-streaming version for MCP).

        Returns dict with:
            - success: bool
            - repo_id: str
            - repo_name: str
            - files_processed: int
            - chunks_indexed: int
            - message: str
            - error: str (if failed)
        """
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []

        try:
            # Validate URL
            logger.info("MCP: Starting indexing", url=url, branch=branch)
            repo_info = self.validator.parse_url(url)
            branch = branch.strip() or repo_info.branch or "main"

            # Create repository record
            repo = Repository(
                url=repo_info.url,
                branch=branch,
                status=RepositoryStatus.CLONING,
            )
            self.repositories[repo.id] = repo
            self._save_repositories()

            # Clone repository
            logger.info("MCP: Cloning repository", url=url, branch=branch)
            repo_path = self.loader.clone_repository(repo_info, branch)
            repo.clone_path = repo_path
            repo.status = RepositoryStatus.INDEXING
            self._save_repositories()

            # Setup filter
            file_filter = FileFilter(
                include_patterns=include_patterns if include_patterns else None,
                exclude_patterns=exclude_patterns if exclude_patterns else None,
            )

            # Process files
            logger.info("MCP: Filtering files", repo_path=str(repo_path))
            files = list(file_filter.filter_files(repo_path))
            file_count = len(files)
            logger.info("MCP: Files to process", count=file_count)

            # Check limits
            if file_count > self.settings.ingestion.max_files_per_repo:
                raise ValueError(
                    f"Repository exceeds file limit ({file_count} > {self.settings.ingestion.max_files_per_repo})"
                )

            # Delete existing chunks
            self.vectorstore.delete_repo_chunks(repo.id)

            # Process documents and chunks
            total_chunks = 0
            batch: list = []
            batch_size = self.settings.ingestion.batch_size

            for file_path in files:
                try:
                    doc = Document.from_file(file_path, repo_path, repo.id)
                    for chunk in self.chunker.chunk_document(doc):
                        chunk.repo_id = repo.id
                        batch.append(chunk)

                        if len(batch) >= batch_size:
                            embedded = self.embedder.embed_chunks(batch, show_progress=False)
                            self.vectorstore.add_chunks(embedded)
                            total_chunks += len(batch)
                            batch = []
                except Exception as e:
                    logger.warning("MCP: Failed to process file", path=str(file_path), error=str(e))

            # Process final batch
            if batch:
                embedded = self.embedder.embed_chunks(batch, show_progress=False)
                self.vectorstore.add_chunks(embedded)
                total_chunks += len(batch)

            # Save commit for incremental updates
            try:
                from git import Repo as GitRepo
                git_repo = GitRepo(repo_path)
                repo.last_commit = git_repo.head.commit.hexsha
            except Exception:
                repo.last_commit = None

            # Update repository status
            repo.chunk_count = total_chunks
            repo.indexed_at = datetime.now()
            repo.status = RepositoryStatus.READY
            self._save_repositories()

            logger.info("MCP: Indexing complete", repo_id=repo.id, chunks=total_chunks)

            return {
                "success": True,
                "repo_id": repo.id,
                "repo_name": repo_info.full_name,
                "files_processed": file_count,
                "chunks_indexed": total_chunks,
                "message": f"Successfully indexed {repo_info.full_name}",
            }

        except ValidationError as e:
            logger.error("MCP: Validation error", error=str(e))
            return {
                "success": False,
                "repo_id": "",
                "repo_name": "",
                "files_processed": 0,
                "chunks_indexed": 0,
                "message": "Validation failed",
                "error": str(e),
            }
        except Exception as e:
            logger.error("MCP: Indexing failed", error=str(e), exc_info=True)
            if "repo" in locals():
                repo.status = RepositoryStatus.ERROR
                repo.error_message = str(e)
                self._save_repositories()
            return {
                "success": False,
                "repo_id": repo.id if "repo" in locals() else "",
                "repo_name": "",
                "files_processed": 0,
                "chunks_indexed": 0,
                "message": "Indexing failed",
                "error": str(e),
            }

    async def query_code(
        self,
        repo_id: str,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Query code in a repository.

        Returns dict with:
            - answer: str
            - citations: list[dict]
            - evidence: list[dict]
            - grounded: bool
            - error: str (if failed)
        """
        repo = self._find_repository(repo_id)
        if not repo:
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": f"Repository not found: {repo_id}",
            }

        if repo.status != RepositoryStatus.READY:
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": f"Repository not ready (status: {repo.status.value})",
            }

        try:
            # Lazy load generator
            if self.generator is None:
                self.generator = ResponseGenerator()

            query = Query(
                question=question.strip(),
                repo_id=repo.id,
                top_k=min(max(top_k, 1), 20),
            )

            response = self.generator.generate(query)

            return {
                "answer": response.answer,
                "citations": [
                    {
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                    }
                    for c in response.citations
                ],
                "evidence": [
                    {
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "content": c.content,
                        "relevance_score": c.relevance_score,
                        "chunk_type": c.chunk_type,
                        "name": c.name,
                    }
                    for c in response.retrieved_chunks
                ],
                "grounded": response.grounded,
            }

        except Exception as e:
            logger.error("MCP: Query failed", error=str(e), exc_info=True)
            return {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "error": str(e),
            }

    async def list_repositories(self) -> dict:
        """List all indexed repositories."""
        repos = []
        for repo in self.repositories.values():
            repos.append({
                "repo_id": repo.id,
                "name": repo.full_name,
                "url": repo.url,
                "branch": repo.branch,
                "chunk_count": repo.chunk_count,
                "status": repo.status.value,
                "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            })
        return {
            "repositories": repos,
            "count": len(repos),
        }

    async def get_repository_info(self, repo_id: str) -> dict:
        """Get detailed info about a repository."""
        repo = self._find_repository(repo_id)
        if not repo:
            return {"error": f"Repository not found: {repo_id}"}

        # Get indexed files from vectorstore
        indexed_files = list(self.vectorstore.get_indexed_files(repo.id))

        return {
            "repo_id": repo.id,
            "name": repo.full_name,
            "url": repo.url,
            "branch": repo.branch,
            "chunk_count": repo.chunk_count,
            "status": repo.status.value,
            "indexed_at": repo.indexed_at.isoformat() if repo.indexed_at else None,
            "indexed_files": sorted(indexed_files),
            "last_commit": repo.last_commit,
            "error_message": repo.error_message,
        }

    async def delete_repository(self, repo_id: str) -> dict:
        """Delete a repository and its data."""
        repo = self._find_repository(repo_id)
        if not repo:
            return {"success": False, "error": f"Repository not found: {repo_id}"}

        try:
            # Delete from vector store
            deleted_chunks = self.vectorstore.delete_repo_chunks(repo.id)

            # Delete cached repo
            try:
                repo_info_stub = type("RepoInfo", (), {"owner": repo.owner, "name": repo.name})()
                self.loader.delete_cache(repo_info_stub)
            except Exception:
                pass  # Cache may not exist

            # Remove from records
            del self.repositories[repo.id]
            self._save_repositories()

            return {
                "success": True,
                "message": f"Deleted {repo.full_name}",
                "chunks_deleted": deleted_chunks,
            }

        except Exception as e:
            logger.error("MCP: Delete failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def update_repository(self, repo_id: str) -> dict:
        """Update repository incrementally."""
        repo = self._find_repository(repo_id)
        if not repo:
            return {"success": False, "error": f"Repository not found: {repo_id}"}

        if not repo.last_commit:
            return {"success": False, "error": "No previous indexing found. Please re-index."}

        if not repo.clone_path or not Path(repo.clone_path).exists():
            return {"success": False, "error": "Repository cache not found. Please re-index."}

        try:
            from git import Repo as GitRepo

            repo_path = Path(repo.clone_path)

            # Update local repository
            self.loader._update_repository(repo_path, repo.branch, None)

            git_repo = GitRepo(repo_path)
            current_commit = git_repo.head.commit.hexsha

            if current_commit == repo.last_commit:
                return {
                    "success": True,
                    "message": "Repository is already up to date",
                    "files_changed": 0,
                    "chunks_added": 0,
                }

            # Get changed files
            diff = git_repo.commit(repo.last_commit).diff(current_commit)

            added = set()
            modified = set()
            deleted = set()

            for d in diff:
                if d.new_file:
                    added.add(d.b_path)
                elif d.deleted_file:
                    deleted.add(d.a_path)
                elif d.renamed:
                    deleted.add(d.a_path)
                    added.add(d.b_path)
                else:
                    modified.add(d.b_path or d.a_path)

            # Delete chunks for deleted/modified files
            for file_path in deleted | modified:
                self.vectorstore.delete_file_chunks(repo.id, file_path)

            # Index new/modified files
            files_to_index = []
            file_filter = FileFilter()
            for file_path in added | modified:
                full_path = repo_path / file_path
                if full_path.exists() and file_filter.should_include(full_path, repo_path):
                    files_to_index.append(full_path)

            new_chunks = 0
            if files_to_index:
                batch = []
                for file_path in files_to_index:
                    try:
                        doc = Document.from_file(file_path, repo_path, repo.id)
                        for chunk in self.chunker.chunk_document(doc):
                            chunk.repo_id = repo.id
                            batch.append(chunk)
                    except Exception:
                        continue

                if batch:
                    embedded = self.embedder.embed_chunks(batch, show_progress=False)
                    self.vectorstore.add_chunks(embedded)
                    new_chunks = len(batch)

            # Update metadata
            repo.last_commit = current_commit
            repo.indexed_at = datetime.now()
            repo.chunk_count = self.vectorstore.get_repo_chunk_count(repo.id)
            self._save_repositories()

            return {
                "success": True,
                "message": "Incremental update complete",
                "files_changed": len(added | modified | deleted),
                "files_added": len(added),
                "files_modified": len(modified),
                "files_deleted": len(deleted),
                "chunks_added": new_chunks,
                "total_chunks": repo.chunk_count,
            }

        except Exception as e:
            logger.error("MCP: Update failed", error=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_code(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        file_filter: str = None,
        chunk_type: str = "all",
    ) -> dict:
        """Semantic code search without LLM generation."""
        repo = self._find_repository(repo_id)
        if not repo:
            return {"results": [], "error": f"Repository not found: {repo_id}"}

        if repo.status != RepositoryStatus.READY:
            return {"results": [], "error": f"Repository not ready: {repo.status.value}"}

        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query, is_query=True)

            # Search vectorstore
            results = self.vectorstore.query(
                query_embedding=query_embedding,
                repo_id=repo.id,
                top_k=min(top_k, 50),
                similarity_threshold=0.0,
            )

            # Filter and format results
            formatted = []
            for chunk, score in results:
                # Apply file filter if specified
                if file_filter:
                    import fnmatch
                    if not fnmatch.fnmatch(chunk.file_path, file_filter):
                        continue

                # Apply chunk type filter
                if chunk_type != "all" and chunk.chunk_type.value != chunk_type:
                    continue

                formatted.append({
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "relevance_score": score,
                    "chunk_type": chunk.chunk_type.value,
                    "name": chunk.name,
                    "citation": chunk.citation,
                })

            return {
                "results": formatted,
                "count": len(formatted),
                "query": query,
            }

        except Exception as e:
            logger.error("MCP: Search failed", error=str(e))
            return {"results": [], "error": str(e)}


# Singleton instance
_mcp_handlers: Optional[MCPHandlers] = None


def get_mcp_handlers() -> MCPHandlers:
    """Get or create the MCPHandlers singleton."""
    global _mcp_handlers
    if _mcp_handlers is None:
        _mcp_handlers = MCPHandlers()
    return _mcp_handlers
```

### 5.5 Archivo: src/coderag/mcp/tools.py

```python
"""MCP Tool definitions for CodeRAG."""

from coderag.mcp.server import mcp
from coderag.mcp.handlers import get_mcp_handlers


@mcp.tool()
async def index_repository(
    url: str,
    branch: str = "",
    include_patterns: list[str] = [],
    exclude_patterns: list[str] = [],
) -> dict:
    """
    Indexa un repositorio de GitHub para poder hacer preguntas sobre su codigo.

    Clona el repositorio, extrae chunks semanticos del codigo usando tree-sitter,
    genera embeddings vectoriales y los almacena para busqueda semantica.

    Args:
        url: URL del repositorio (https://github.com/owner/repo o owner/repo)
        branch: Rama a indexar (opcional, usa la rama por defecto del repo)
        include_patterns: Patrones glob para incluir archivos ['*.py', '*.js']
        exclude_patterns: Patrones glob para excluir archivos ['**/tests/**']

    Returns:
        dict con repo_id, files_processed, chunks_indexed y mensaje de estado
    """
    handlers = get_mcp_handlers()
    return await handlers.index_repository(url, branch, include_patterns, exclude_patterns)


@mcp.tool()
async def query_code(
    repo_id: str,
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Hace una pregunta sobre el codigo de un repositorio indexado.

    Devuelve una respuesta fundamentada con citas exactas al codigo fuente.
    Cada afirmacion incluye referencias en formato [archivo:linea_inicio-linea_fin].

    Args:
        repo_id: ID del repositorio (de list_repositories o index_repository)
        question: Pregunta sobre el codigo (ej: "Donde esta definida la funcion X?")
        top_k: Numero de chunks a recuperar para contexto (1-20, default 5)

    Returns:
        dict con answer (respuesta), citations (citas), evidence (chunks) y grounded (bool)
    """
    handlers = get_mcp_handlers()
    return await handlers.query_code(repo_id, question, top_k)


@mcp.tool()
async def list_repositories() -> dict:
    """
    Lista todos los repositorios de GitHub indexados.

    Returns:
        dict con repositories (lista) y count (total)
    """
    handlers = get_mcp_handlers()
    return await handlers.list_repositories()


@mcp.tool()
async def get_repository_info(repo_id: str) -> dict:
    """
    Obtiene informacion detallada de un repositorio indexado.

    Args:
        repo_id: ID del repositorio (completo o primeros 8 caracteres)

    Returns:
        dict con metadatos, estadisticas y lista de archivos indexados
    """
    handlers = get_mcp_handlers()
    return await handlers.get_repository_info(repo_id)


@mcp.tool()
async def delete_repository(repo_id: str) -> dict:
    """
    Elimina un repositorio indexado y todos sus datos.

    Args:
        repo_id: ID del repositorio a eliminar

    Returns:
        dict con success y mensaje
    """
    handlers = get_mcp_handlers()
    return await handlers.delete_repository(repo_id)


@mcp.tool()
async def update_repository(repo_id: str) -> dict:
    """
    Actualiza un repositorio de forma incremental.

    Solo procesa archivos modificados desde la ultima indexacion,
    haciendo el proceso mucho mas rapido que una re-indexacion completa.

    Args:
        repo_id: ID del repositorio a actualizar

    Returns:
        dict con estadisticas de cambios procesados
    """
    handlers = get_mcp_handlers()
    return await handlers.update_repository(repo_id)


@mcp.tool()
async def search_code(
    repo_id: str,
    query: str,
    top_k: int = 10,
    file_filter: str = None,
    chunk_type: str = "all",
) -> dict:
    """
    Busqueda semantica de codigo sin generar respuesta LLM.

    Util para explorar rapidamente el codigo y encontrar fragmentos relevantes.

    Args:
        repo_id: ID del repositorio
        query: Texto de busqueda semantica
        top_k: Numero de resultados (max 50)
        file_filter: Filtrar por patron de archivo (ej: '*.py')
        chunk_type: Filtrar por tipo: function, class, method, text, all

    Returns:
        dict con results (lista de chunks con scores) y count
    """
    handlers = get_mcp_handlers()
    return await handlers.search_code(repo_id, query, top_k, file_filter, chunk_type)
```

### 5.6 Archivo: src/coderag/mcp/resources.py

```python
"""MCP Resource definitions for CodeRAG."""

import json
from mcp.types import Resource, TextResourceContents

from coderag.mcp.server import mcp
from coderag.mcp.handlers import get_mcp_handlers


@mcp.resource("repository://{repo_id}")
async def get_repository_resource(repo_id: str) -> Resource:
    """
    Recurso: Metadatos de un repositorio indexado.

    URI: repository://{repo_id}
    """
    handlers = get_mcp_handlers()
    info = await handlers.get_repository_info(repo_id)

    if "error" in info:
        return Resource(
            uri=f"repository://{repo_id}",
            name=f"Repository {repo_id}",
            mimeType="application/json",
            text=json.dumps({"error": info["error"]}),
        )

    return Resource(
        uri=f"repository://{repo_id}",
        name=info.get("name", repo_id),
        description=f"Repository {info.get('name')} with {info.get('chunk_count', 0)} chunks",
        mimeType="application/json",
        text=json.dumps(info, indent=2),
    )


@mcp.resource("repositories://list")
async def list_repositories_resource() -> Resource:
    """
    Recurso: Lista de todos los repositorios indexados.

    URI: repositories://list
    """
    handlers = get_mcp_handlers()
    repos = await handlers.list_repositories()

    return Resource(
        uri="repositories://list",
        name="All Indexed Repositories",
        description=f"{repos['count']} repositories indexed",
        mimeType="application/json",
        text=json.dumps(repos, indent=2),
    )
```

### 5.7 Archivo: src/coderag/mcp/prompts.py

```python
"""MCP Prompt definitions for CodeRAG."""

from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent

from coderag.mcp.server import mcp


@mcp.prompt()
def analyze_repository(repo_url: str) -> list[PromptMessage]:
    """
    Prompt para analizar un repositorio completo.

    Guia al LLM para indexar y hacer un analisis inicial del repositorio.
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Por favor analiza el repositorio: {repo_url}

Pasos a seguir:
1. Primero, indexa el repositorio usando la herramienta index_repository
2. Una vez indexado, usa query_code para responder estas preguntas:
   - Cual es la estructura general del proyecto?
   - Cuales son los archivos/modulos principales?
   - Que patrones de diseno se utilizan?
   - Como esta organizado el codigo?

Proporciona un resumen ejecutivo del repositorio."""
            ),
        )
    ]


@mcp.prompt()
def find_implementation(repo_id: str, feature: str) -> list[PromptMessage]:
    """
    Prompt para encontrar la implementacion de una funcionalidad.
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""En el repositorio {repo_id}, necesito encontrar como esta implementado: {feature}

Por favor:
1. Usa search_code para encontrar codigo relacionado
2. Usa query_code para entender la implementacion
3. Explica el flujo de datos y la logica
4. Incluye todas las citas relevantes al codigo"""
            ),
        )
    ]


@mcp.prompt()
def code_review(repo_id: str, focus_area: str = "") -> list[PromptMessage]:
    """
    Prompt para hacer code review de un repositorio.
    """
    focus = f" con enfoque en {focus_area}" if focus_area else ""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""Realiza un code review del repositorio {repo_id}{focus}.

Analiza:
1. Calidad del codigo y mejores practicas
2. Posibles bugs o problemas
3. Oportunidades de refactorizacion
4. Documentacion y comentarios
5. Manejo de errores
6. Testing (si existe)

Usa query_code y search_code para fundamentar tus observaciones con citas al codigo."""
            ),
        )
    ]
```

### 5.8 Modificacion: src/coderag/main.py

```python
"""CodeRAG main application entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coderag.config import get_settings
from coderag.logging import setup_logging, get_logger

# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.server.log_level.upper())
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting CodeRAG",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    yield
    logger.info("Shutting down CodeRAG")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG-based Q&A system for code repositories with verifiable citations",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    # Register API routes
    from coderag.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1")

    # Mount MCP Server (Streamable HTTP)
    try:
        from coderag.mcp.server import create_mcp_server
        mcp_server = create_mcp_server()

        # Mount MCP on /mcp path
        app.mount("/mcp", mcp_server.streamable_http_app())
        logger.info("MCP server mounted at /mcp")
    except ImportError as e:
        logger.warning("MCP server not available", error=str(e))
    except Exception as e:
        logger.error("Failed to mount MCP server", error=str(e))

    # Mount Gradio UI
    try:
        from coderag.ui.app import create_gradio_app
        import gradio as gr

        gradio_app = create_gradio_app()
        app = gr.mount_gradio_app(app, gradio_app, path="/")
        logger.info("Gradio UI mounted at /")
    except ImportError as e:
        logger.warning("Gradio UI not available", error=str(e))
    except Exception as e:
        logger.error("Failed to mount Gradio UI", error=str(e))

    return app


def main() -> None:
    """Run the application."""
    app = create_app()

    logger.info(
        "Starting server",
        host=settings.server.host,
        port=settings.server.port,
    )

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_level=settings.server.log_level,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application crashed", error=str(e), exc_info=True)
        import traceback
        print("\n" + "="*80)
        print("FATAL ERROR:")
        print("="*80)
        traceback.print_exc()
        print("="*80)
        input("Press Enter to close...")
```

### 5.9 Archivo: src/coderag/mcp/stdio.py (Soporte para Claude Desktop)

```python
"""stdio transport for MCP server (for Claude Desktop compatibility)."""

import asyncio
import sys

from coderag.mcp.server import create_mcp_server
from coderag.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)


async def run_stdio():
    """Run MCP server with stdio transport."""
    logger.info("Starting MCP server in stdio mode")
    mcp = create_mcp_server()

    # Run with stdio transport
    async with mcp.run_stdio() as streams:
        await streams.wait_closed()


def main():
    """Entry point for stdio mode."""
    try:
        asyncio.run(run_stdio())
    except KeyboardInterrupt:
        logger.info("MCP stdio server interrupted")
    except Exception as e:
        logger.error("MCP stdio server crashed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 6. CONFIGURACION DE CLIENTES

### 6.1 Claude Desktop (stdio via Docker)

Archivo: `~/.config/claude/claude_desktop_config.json` (Linux) o equivalente

```json
{
  "mcpServers": {
    "coderag": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "coderag-container",
        "python",
        "-m",
        "coderag.mcp.stdio"
      ],
      "env": {}
    }
  }
}
```

### 6.2 Claude Desktop (stdio local, sin Docker)

```json
{
  "mcpServers": {
    "coderag": {
      "command": "python",
      "args": ["-m", "coderag.mcp.stdio"],
      "cwd": "/path/to/coderag",
      "env": {
        "PYTHONPATH": "/path/to/coderag/src"
      }
    }
  }
}
```

### 6.3 Clientes HTTP (Streamable HTTP)

Endpoint: `http://localhost:8000/mcp`

Ejemplo con curl para listar herramientas:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

---

## 7. TESTING

### 7.1 Archivo: tests/test_mcp_tools.py

```python
"""Tests for MCP tools."""

import pytest
from unittest.mock import AsyncMock, patch

from coderag.mcp.handlers import MCPHandlers


@pytest.fixture
def handlers():
    """Create MCPHandlers instance for testing."""
    with patch.object(MCPHandlers, '_load_repositories', return_value={}):
        return MCPHandlers()


@pytest.mark.asyncio
async def test_list_repositories_empty(handlers):
    """Test listing repositories when none exist."""
    result = await handlers.list_repositories()
    assert result["count"] == 0
    assert result["repositories"] == []


@pytest.mark.asyncio
async def test_query_code_repo_not_found(handlers):
    """Test querying non-existent repository."""
    result = await handlers.query_code("nonexistent", "test question")
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_delete_repository_not_found(handlers):
    """Test deleting non-existent repository."""
    result = await handlers.delete_repository("nonexistent")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# Integration tests require actual dependencies
@pytest.mark.integration
@pytest.mark.asyncio
async def test_index_small_repo():
    """Integration test: index a small repository."""
    handlers = MCPHandlers()
    result = await handlers.index_repository(
        url="https://github.com/keleshev/mini",  # Small test repo
        branch="master"
    )
    assert result["success"] is True
    assert result["chunks_indexed"] > 0
```

### 7.2 Archivo: tests/test_mcp_server.py

```python
"""Tests for MCP server integration."""

import pytest
from fastapi.testclient import TestClient

from coderag.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_mcp_endpoint_exists(client):
    """Test that MCP endpoint is mounted."""
    # MCP uses POST for all operations
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    )
    assert response.status_code == 200


def test_health_check(client):
    """Test health endpoint still works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## 8. DEPENDENCIAS ACTUALIZADAS

### pyproject.toml (agregar):

```toml
dependencies = [
    # ... existentes ...
    "mcp>=1.0.0",
]
```

---

## 9. CHECKLIST DE IMPLEMENTACION

### Fase 1: Estructura Base
- [ ] Crear directorio `src/coderag/mcp/`
- [ ] Crear `__init__.py`
- [ ] Crear `server.py` con FastMCP
- [ ] Crear `handlers.py` con MCPHandlers
- [ ] Agregar dependencia `mcp` a pyproject.toml

### Fase 2: Tools
- [ ] Implementar `tools.py` con todas las herramientas
- [ ] Probar cada tool individualmente
- [ ] Verificar manejo de errores

### Fase 3: Resources
- [ ] Implementar `resources.py`
- [ ] Probar acceso a recursos

### Fase 4: Prompts
- [ ] Implementar `prompts.py`
- [ ] Probar prompts predefinidos

### Fase 5: Integracion
- [ ] Modificar `main.py` para montar MCP
- [ ] Crear `stdio.py` para Claude Desktop
- [ ] Probar Streamable HTTP
- [ ] Probar stdio transport

### Fase 6: Testing
- [ ] Crear tests unitarios
- [ ] Crear tests de integracion
- [ ] Documentar configuracion de clientes

### Fase 7: Docker
- [ ] Actualizar Dockerfile si es necesario
- [ ] Actualizar docker-compose.yaml
- [ ] Probar con Claude Desktop via docker exec

---

## 10. ERRORES COMUNES Y SOLUCIONES

### Error: "Module mcp not found"
```bash
pip install mcp
```

### Error: "Repository not ready"
El repositorio aun esta siendo indexado. Esperar a que `status` sea `ready`.

### Error: "No previous indexing found"
Para usar `update_repository`, el repo debe haber sido indexado previamente con la version que guarda `last_commit`.

### Error en stdio con Claude Desktop
Verificar que el contenedor Docker esta corriendo y el nombre es correcto:
```bash
docker ps  # Verificar nombre del contenedor
docker exec -it <container> python -m coderag.mcp.stdio  # Probar manualmente
```

---

## 11. EJEMPLO DE USO COMPLETO

### Flujo tipico desde Claude Desktop:

1. **Usuario**: "Indexa el repo fastapi/fastapi y explicame como funciona el routing"

2. **Claude** (internamente):
   - Llama `index_repository(url="fastapi/fastapi")`
   - Espera resultado: `{success: true, repo_id: "abc123", chunks_indexed: 5000}`
   - Llama `query_code(repo_id="abc123", question="Como funciona el routing?")`
   - Recibe respuesta con citas

3. **Claude** (al usuario):
   "He indexado el repositorio fastapi/fastapi (5000 chunks).

   El routing en FastAPI funciona mediante decoradores... [src/fastapi/routing.py:45-120]

   La clase `APIRouter` gestiona... [src/fastapi/routing.py:200-250]"

---

Este documento contiene toda la especificacion necesaria para implementar el servidor MCP completo para CodeRAG.
