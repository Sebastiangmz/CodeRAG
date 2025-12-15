# RAG Grounded para Repositorios de Código - Documento de Arquitectura

## Información del Proyecto

- **Nombre del Proyecto**: CodeRAG (RAG Grounded para Codebases)
- **Tipo**: Sistema de Retrieval-Augmented Generation especializado en repositorios de código
- **Lenguaje Principal de Soporte**: Python (con capacidad de extensión a otros lenguajes)
- **Fase Actual**: MVP - Modo Q&A
- **Evolución Planificada**: Modo Patch/Diff (generación de cambios de código)
- **Interfaz**: Gradio montado en FastAPI
- **Deployment**: Docker Compose (100% reproducible local)
- **Alcance MVP**: Solo repositorios públicos de GitHub (sin autenticación)
- **Hardware Objetivo**: GPU NVIDIA RTX 4060 8GB VRAM

---

## 1. Visión General del Proyecto

### 1.1 Qué es RAG Grounded

RAG (Retrieval-Augmented Generation) funciona "aumentando" el prompt del usuario con pasajes recuperados desde una base de conocimiento externa. El LLM genera la respuesta usando ese contexto recuperado en lugar de inventar información.

**Beneficio principal**: Reducir alucinaciones porque el modelo se apoya en información recuperada (documentación, código fuente) en vez de generar respuestas sin fundamento.

### 1.2 Objetivo del Sistema

Construir un asistente de Q&A sobre repositorios de código que:

1. **Responda preguntas** sobre el código de forma precisa
2. **Cite las fuentes** (archivo + rango de líneas) para cada afirmación
3. **Rechace responder** cuando no hay evidencia en los chunks recuperados
4. **Se adapte** a casi cualquier repositorio, con optimización especial para Python

### 1.3 Casos de Uso MVP (Modo Q&A)

El usuario puede hacer preguntas como:

- "¿Dónde se define la clase `UserAuthentication`?"
- "¿Qué hace la función `process_payment()`?"
- "¿Cómo se configura el logger en este proyecto?"
- "¿Qué dependencias usa este módulo?"
- "¿Cuál es el flujo de datos en el endpoint `/api/users`?"

**Criterio de éxito**: El sistema encuentra el lugar correcto en el repo y la respuesta está sustentada por el contexto recuperado con citas verificables.

### 1.4 Evolución Futura (Modo Patch/Diff)

En fases posteriores, el usuario podrá pedir:

- "Agrega soporte para paginación en este endpoint"
- "Refactoriza esta función para usar async/await"
- "Arregla este bug en el manejo de errores"

El sistema responderá con un **diff aplicable** que indica qué líneas agregar/quitar en qué archivo.

**Criterio de éxito adicional**: El cambio debe compilar, pasar tests, y estar justificado por el contexto del repo.

---

## 2. Modelos de IA y Licencias

### 2.1 Modelos Seleccionados

Este proyecto utiliza modelos **100% locales y gratuitos** que corren en una RTX 4060 (8GB VRAM) usando cuantización 4-bit.

| Rol | Modelo | Licencia | Justificación |
|-----|--------|----------|---------------|
| **LLM Generator** | Qwen2.5-Coder-7B-Instruct | Apache 2.0 | Especializado en código, buen rendimiento en Q&A sobre repos |
| **Embeddings** | nomic-embed-text v1.5 | Apache 2.0 | Embeddings de calidad, open source, bajo consumo |
| **Alternativa ligera** | Llama-3.2-3B-Instruct | Llama 3.2 Community | Más rápido, menos VRAM (revisar licencia) |

### 2.2 Configuración de Modelos

```yaml
# configs/models.yaml
llm:
  model_name: "Qwen/Qwen2.5-Coder-7B-Instruct"
  quantization: "4bit"  # GPTQ o bitsandbytes
  max_new_tokens: 1024
  temperature: 0.1
  device_map: "auto"

embeddings:
  model_name: "nomic-ai/nomic-embed-text-v1.5"
  device: "cuda"
  normalize_embeddings: true
  dimensions: 768
```

### 2.3 Licencias y Permisos

Ambos modelos principales están bajo **Apache License 2.0**, que permite:
- ✅ Uso comercial y personal
- ✅ Modificación y redistribución
- ✅ Crear obras derivadas (como adaptadores LoRA)
- ⚠️ Requiere mantener avisos de copyright si redistribuyes

**Links oficiales de licencias (pinnear estas versiones):**
- [Qwen2.5-Coder-7B-Instruct LICENSE](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE)
- [nomic-embed-text v1.5 LICENSE](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### 2.4 Mejores Prácticas para Repositorio Público

Dado que el objetivo es tener el código público para portafolio, se deben seguir estas reglas:

#### ✅ SÍ publicar en el repo:
- Todo el código fuente (pipeline, UI, API)
- Scripts de entrenamiento (`train_qlora.py`)
- Configuraciones (YAML/JSON con hiperparámetros)
- README con métricas y resultados
- Docker files y docker-compose
- Datasets de evaluación (preguntas de prueba)

#### ❌ NO publicar en el repo:
- Checkpoints de modelos (`*.safetensors`, `*.bin`)
- Adaptadores LoRA entrenados (`adapter_model.safetensors`, `adapter_config.json`)
- Índices vectoriales pre-construidos (ChromaDB dumps)
- Datasets procesados de repos de terceros
- Archivos `.env` con API keys

#### Estructura de .gitignore recomendada:

```gitignore
# Artefactos de modelos y fine-tuning
artifacts/
checkpoints/
adapters/
*.safetensors
*.bin
*.pt
*.pth

# Base de datos vectorial
vectorstore/
chroma_db/
data/

# Cache de repositorios clonados
repos/

# Entorno y secretos
.env
.env.local
*.env

# Python
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
*.egg-info/
venv/
.venv/

# IDE
.vscode/
.idea/
```

### 2.5 ¿Qué son los Adaptadores LoRA?

Los **adaptadores LoRA** (Low-Rank Adaptation) son pequeños conjuntos de pesos que se entrenan para adaptar un modelo base a una tarea específica, sin modificar los pesos originales del modelo.

**En este proyecto:**
- El modelo base (Qwen2.5-Coder) se descarga desde Hugging Face y se mantiene congelado
- El fine-tuning con QLoRA produce archivos pequeños (~100MB): `adapter_model.safetensors` + `adapter_config.json`
- En inferencia, el adaptador se "enchufa" al modelo base para producir el comportamiento entrenado

**Distribución de artefactos derivados:**
- Si subes los archivos `adapter_*` a GitHub, estás redistribuyendo un "artefacto derivado" del modelo base
- Esto activa obligaciones de licencia adicionales
- **Recomendación**: El pipeline descarga el modelo base y genera adaptadores localmente; no se suben al repo público

---

## 3. Arquitectura del Sistema

### 3.1 Pipeline Principal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE RAG                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  CLONAR/ │   │ FILTRAR  │   │ CHUNKING │   │EMBEDDINGS│   │  ÍNDICE  │  │
│  │  CARGAR  │──▶│ ARCHIVOS │──▶│          │──▶│  nomic   │──▶│ ChromaDB │  │
│  │   REPO   │   │          │   │          │   │          │   │          │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                                 │
│  │ CONSULTA │   │RECUPERAR │   │ GENERAR  │                                 │
│  │ USUARIO  │──▶│  TOP-K   │──▶│  Qwen2.5 │──▶ Respuesta con citas          │
│  │          │   │  CHUNKS  │   │  Coder   │                                 │
│  └──────────┘   └──────────┘   └──────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Componentes del Sistema

#### 3.2.1 Módulo de Carga de Repositorios

**Responsabilidad**: Clonar o cargar repositorios Git y convertir archivos a documentos procesables.

**Implementación sugerida**:
- Usar `GitLoader` de LangChain o implementación propia
- Soportar: URL de Git, path local, o archivo ZIP

**Salida**: Lista de documentos con contenido y metadatos (path, nombre, extensión)

#### 3.2.2 Módulo de Filtrado de Archivos

**Responsabilidad**: Filtrar archivos relevantes y excluir basura (binarios, lockfiles, dependencias vendorizadas).

**Reglas por defecto**:

```yaml
include_patterns:
  - "README*"
  - "docs/**"
  - "src/**"
  - "*.py"
  - "*.md"
  - "*.rst"
  - "pyproject.toml"
  - "requirements*.txt"
  - "setup.py"
  - "setup.cfg"
  - "*.yaml"
  - "*.yml"
  - "*.json"  # Solo configs, no data dumps
  - "*.toml"

exclude_patterns:
  - ".git/**"
  - "venv/**"
  - ".venv/**"
  - "__pycache__/**"
  - "dist/**"
  - "build/**"
  - "node_modules/**"
  - "*.egg-info/**"
  - ".tox/**"
  - ".pytest_cache/**"
  - ".mypy_cache/**"
  - "*.png"
  - "*.jpg"
  - "*.jpeg"
  - "*.gif"
  - "*.ico"
  - "*.pdf"
  - "*.zip"
  - "*.tar.gz"
  - "*.whl"
  - "*.pyc"
  - "*.pyo"
  - "*.so"
  - "*.dylib"
  - "*.dll"
  - "poetry.lock"
  - "package-lock.json"
  - "yarn.lock"
  - "Pipfile.lock"
```

**Configuración**: Permitir override por proyecto vía archivo `.coderagignore` o config.

#### 3.2.3 Módulo de Chunking

**Responsabilidad**: Dividir archivos en chunks coherentes para indexación.

**Estrategia dual**:

1. **Chunking Semántico (Python)**: Usar AST para extraer unidades coherentes
2. **Chunking por Texto (Fallback)**: Para archivos sin parser disponible

##### Chunking Semántico para Python

**Tecnología**: Tree-sitter (py-tree-sitter)

**Unidades a extraer**:
- `function_definition` → Funciones completas
- `class_definition` → Clases completas (o métodos individuales si son muy grandes)
- `decorated_definition` → Funciones/clases con decoradores

**Metadatos por chunk**:
```python
{
    "file_path": "src/auth/handlers.py",
    "chunk_type": "function",
    "name": "authenticate_user",
    "start_line": 45,
    "end_line": 78,
    "parent_class": "AuthHandler",  # Si aplica
    "decorators": ["@require_auth", "@log_access"],
    "signature": "def authenticate_user(self, username: str, password: str) -> bool:",
    "docstring": "Authenticate user with credentials...",
    "imports_used": ["hashlib", "datetime"],
    "commit_hash": "abc123..."  # Opcional
}
```

##### Chunking por Texto (Fallback)

**Parámetros**:
- `chunk_size`: 1000-1500 tokens (configurable)
- `chunk_overlap`: 100-200 tokens
- Preservar límites de línea cuando sea posible

**Metadatos por chunk**:
```python
{
    "file_path": "docs/installation.md",
    "chunk_type": "text",
    "start_line": 1,
    "end_line": 45,
    "section_title": "Installation Guide",  # Si se puede inferir
    "commit_hash": "abc123..."
}
```

#### 3.2.4 Módulo de Embeddings

**Responsabilidad**: Generar embeddings vectoriales para cada chunk.

**Modelo seleccionado**: `nomic-embed-text v1.5`

```python
from sentence_transformers import SentenceTransformer

# Inicialización
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    device="cuda"
)

# Generar embeddings
embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
```

**Consideraciones**:
- Dimensión: 768
- El modelo debe ser consistente entre indexación y consulta
- Normalizar embeddings para usar similitud coseno

#### 3.2.5 Base de Datos Vectorial

**Responsabilidad**: Almacenar embeddings e índices para búsqueda por similitud.

**Tecnología seleccionada**: ChromaDB

**Justificación**:
- Simple de configurar y usar
- Buen DX (Developer Experience)
- Ideal para desarrollo local y MVP
- Persistencia en disco fácil de configurar
- Integración nativa con LangChain

**Configuración básica**:

```python
import chromadb
from chromadb.config import Settings

# Cliente persistente
client = chromadb.PersistentClient(
    path="./data/chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Crear o obtener colección
collection = client.get_or_create_collection(
    name="coderag_chunks",
    metadata={"hnsw:space": "cosine"}  # Similitud coseno
)
```

**Índice debe almacenar**:
- Vector embedding
- Contenido del chunk (texto)
- Todos los metadatos del chunk
- ID único del chunk

#### 3.2.6 Módulo de Recuperación (Retrieval)

**Responsabilidad**: Dado un query, recuperar los K chunks más relevantes.

**Parámetros**:
- `top_k`: 5-10 chunks (configurable)
- `similarity_threshold`: 0.7 mínimo (opcional, para filtrar chunks poco relevantes)

**Estrategias de mejora** (para fases posteriores):
- Hybrid search (vector + keyword BM25)
- Reranking con modelo cross-encoder
- Query expansion/reformulation

#### 3.2.7 Módulo de Generación

**Responsabilidad**: Generar respuesta basada en chunks recuperados, con citas.

**Modelo**: Qwen2.5-Coder-7B-Instruct (cuantizado 4-bit)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuración de cuantización
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)
```

**Formato de citas**:
```
[archivo:línea_inicio-línea_fin]
```

Ejemplos:
- `[src/auth/handlers.py:45-78]`
- `[README.md:12-25]`
- `[docs/api.md:100-115]`

**Prompt del sistema** (comportamiento requerido):

```
Eres un asistente de código que responde preguntas sobre un repositorio.

REGLAS ESTRICTAS:
1. Solo responde basándote en los chunks de código/documentación proporcionados
2. Cada afirmación importante DEBE incluir una cita en formato [archivo:líneas]
3. Si la información NO está en los chunks, responde: "No encontré información sobre esto en el repositorio indexado"
4. NO inventes código, funciones, o archivos que no estén en los chunks
5. Si la pregunta es ambigua, pide clarificación
6. Responde de forma concisa y estructurada

FORMATO DE RESPUESTA:
- Respuestas en bullets cuando sea apropiado
- Cita después de cada afirmación relevante
- Incluye snippets de código solo si son cortos y relevantes
```

---

## 4. Interfaz de Usuario (Gradio + FastAPI)

### 4.1 Arquitectura de la UI

La interfaz se implementa con **Gradio montado dentro de FastAPI** usando `gradio.mount_gradio_app()`. Esto permite tener UI y API en el mismo proceso, accesible en `http://localhost:8000`.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI App                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Gradio UI         │    │        API REST                 │ │
│  │   /gradio           │    │        /api/v1/*                │ │
│  │                     │    │                                 │ │
│  │  - Indexar repos    │    │  - POST /repos/index            │ │
│  │  - Chat Q&A         │    │  - POST /query                  │ │
│  │  - Ver progreso     │    │  - GET /repos                   │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Diseño de la Interfaz

#### Panel de Indexación

```
┌─────────────────────────────────────────────────────────────────┐
│                    📦 INDEXAR REPOSITORIO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  URL del Repositorio (GitHub):                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ https://github.com/owner/repo                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ─────────────── Opciones Avanzadas (expandible) ─────────────  │
│                                                                  │
│  Branch:              Top-K:           Filtros:                  │
│  ┌──────────────┐    ┌──────────┐     ☑ Incluir tests           │
│  │ main         │    │ 5        │     ☐ Solo documentación      │
│  └──────────────┘    └──────────┘                               │
│                                                                  │
│              ┌─────────────────────────┐                        │
│              │      🚀 INDEXAR         │                        │
│              └─────────────────────────┘                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ████████████████████░░░░░░░░░░  65% - Procesando chunks...  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Panel de Chat Q&A

```
┌─────────────────────────────────────────────────────────────────┐
│                    💬 PREGUNTAR SOBRE EL CÓDIGO                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Repositorio activo: langchain (342 chunks indexados)           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ¿Dónde se define la clase BaseRetriever?                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│              ┌─────────────────────────┐                        │
│              │      🔍 PREGUNTAR       │                        │
│              └─────────────────────────┘                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ RESPUESTA:                                                   ││
│  │                                                              ││
│  │ La clase `BaseRetriever` se define en el módulo de          ││
│  │ retrievers [src/retrievers/base.py:23-89]:                  ││
│  │                                                              ││
│  │ - Es una clase abstracta que define la interfaz común       ││
│  │   [src/retrievers/base.py:25-30]                            ││
│  │ - El método principal es `get_relevant_documents()`         ││
│  │   [src/retrievers/base.py:45-67]                            ││
│  │                                                              ││
│  │ ─────────────────────────────────────────────────────────── ││
│  │ 📎 EVIDENCIA:                                                ││
│  │ • src/retrievers/base.py (líneas 23-89)                     ││
│  │ • src/retrievers/__init__.py (líneas 5-12)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Componentes Gradio

```python
import gradio as gr
from fastapi import FastAPI

app = FastAPI()

# Definición de la interfaz Gradio
with gr.Blocks(title="CodeRAG - Q&A sobre Repositorios") as demo:
    gr.Markdown("# 🔍 CodeRAG - Asistente de Código con RAG")
    
    with gr.Tab("📦 Indexar"):
        repo_url = gr.Textbox(
            label="URL del Repositorio (GitHub)",
            placeholder="https://github.com/owner/repo"
        )
        
        with gr.Accordion("Opciones Avanzadas", open=False):
            branch = gr.Textbox(label="Branch", value="main")
            top_k = gr.Slider(minimum=1, maximum=20, value=5, label="Top-K chunks")
            include_tests = gr.Checkbox(label="Incluir tests", value=False)
            docs_only = gr.Checkbox(label="Solo documentación", value=False)
        
        index_btn = gr.Button("🚀 Indexar", variant="primary")
        index_progress = gr.Progress()
        index_output = gr.Textbox(label="Estado", interactive=False)
        
        index_btn.click(
            fn=index_repository,
            inputs=[repo_url, branch, top_k, include_tests, docs_only],
            outputs=[index_output]
        )
    
    with gr.Tab("💬 Preguntar"):
        repo_status = gr.Markdown("*No hay repositorio indexado*")
        question = gr.Textbox(
            label="Tu pregunta",
            placeholder="¿Dónde se define la función X?"
        )
        ask_btn = gr.Button("🔍 Preguntar", variant="primary")
        
        answer_output = gr.Markdown(label="Respuesta")
        evidence_output = gr.JSON(label="Evidencia (chunks recuperados)")
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question],
            outputs=[answer_output, evidence_output]
        )

# Montar Gradio en FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
```

### 4.4 Flujo de Usuario

#### Flujo 1: Indexar Repositorio

```
1. Usuario abre http://localhost:8000
2. En tab "Indexar", pega URL: https://github.com/owner/repo
3. (Opcional) Ajusta branch, top-k, filtros
4. Click en "Indexar"
5. Sistema muestra barra de progreso:
   - "Clonando repositorio..." (10%)
   - "Filtrando archivos..." (20%)
   - "Procesando chunks..." (40-80%)
   - "Generando embeddings..." (80-95%)
   - "Guardando índice..." (95-100%)
6. Mensaje: "✅ Repositorio indexado: 342 chunks"
7. Tab "Preguntar" se habilita
```

#### Flujo 2: Hacer Pregunta

```
1. Usuario va a tab "Preguntar"
2. Ve: "Repositorio activo: repo-name (342 chunks)"
3. Escribe pregunta: "¿Cómo se configura el logging?"
4. Click en "Preguntar"
5. Sistema:
   a. Genera embedding de la pregunta (nomic-embed)
   b. Busca top-k chunks similares (ChromaDB)
   c. Construye prompt con contexto
   d. Qwen2.5-Coder genera respuesta con citas
6. Muestra respuesta + evidencia (archivos, líneas, snippets)
```

### 4.5 Manejo de Progreso con Gradio

Gradio soporta `gr.Progress` para mostrar avance en tareas largas:

```python
def index_repository(repo_url: str, branch: str, progress=gr.Progress()):
    """Indexa un repositorio con feedback de progreso."""
    
    progress(0, desc="Validando URL...")
    validate_github_url(repo_url)
    
    progress(0.1, desc="Clonando repositorio...")
    repo_path = clone_repository(repo_url, branch)
    
    progress(0.2, desc="Filtrando archivos...")
    files = filter_files(repo_path)
    
    progress(0.3, desc="Procesando chunks...")
    chunks = []
    for i, file in enumerate(files):
        chunks.extend(chunk_file(file))
        progress(0.3 + (0.4 * i / len(files)), desc=f"Chunking: {file.name}")
    
    progress(0.7, desc="Generando embeddings (nomic-embed)...")
    embeddings = generate_embeddings(chunks)
    
    progress(0.9, desc="Guardando índice (ChromaDB)...")
    save_index(embeddings, chunks)
    
    progress(1.0, desc="¡Completado!")
    return f"✅ Repositorio indexado: {len(chunks)} chunks"
```

---

## 5. Infraestructura Docker

### 5.1 Arquitectura de Contenedores

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    coderag-app                               ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ ││
│  │  │  FastAPI    │  │   Gradio    │  │   RAG Pipeline      │ ││
│  │  │  :8000      │  │   UI        │  │   (indexing/query)  │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Volúmenes (locales, no en Git)           ││
│  │  ./data:/app/data       (ChromaDB - índices vectoriales)    ││
│  │  ./repos:/app/repos     (repos clonados - cache)            ││
│  │  ./models:/app/models   (cache modelos HuggingFace)         ││
│  │  ./adapters:/app/adapters (LoRA adapters - si se entrenan)  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Dockerfile (con soporte CUDA)

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Instalar Python y dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY configs/ ./configs/

# Crear directorios para datos persistentes
RUN mkdir -p /app/data /app/repos /app/models /app/adapters

# Variables de entorno
ENV PYTHONPATH=/app/src
ENV DATA_DIR=/app/data
ENV REPOS_DIR=/app/repos
ENV HF_HOME=/app/models
ENV ADAPTERS_DIR=/app/adapters

# Puerto de la aplicación
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "coderag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  coderag:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      # Persistencia de índices vectoriales (ChromaDB)
      - ./data:/app/data
      # Cache de repositorios clonados
      - ./repos:/app/repos
      # Cache de modelos de HuggingFace (evita re-descargar)
      - ./models:/app/models
      # Adaptadores LoRA entrenados (si aplica)
      - ./adapters:/app/adapters
    environment:
      # Configuración de modelos (locales, sin API keys)
      - LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
      - EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
      # Configuración
      - LOG_LEVEL=INFO
      - GRADIO_SERVER_NAME=0.0.0.0
      # CUDA
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  data:
  repos:
  models:
  adapters:
```

### 5.4 Archivo .env

```bash
# .env.example (copiar a .env)
# NO INCLUIR ESTE ARCHIVO EN GIT SI TIENE DATOS SENSIBLES

# Modelos locales (no requieren API keys)
LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Configuración de retrieval
TOP_K_DEFAULT=5
SIMILARITY_THRESHOLD=0.7

# Configuración de chunking
CHUNK_SIZE=1500
CHUNK_OVERLAP=150

# Configuración de generación
MAX_NEW_TOKENS=1024
TEMPERATURE=0.1

# HuggingFace (opcional, para descargas privadas)
# HF_TOKEN=hf_...

# Logging
LOG_LEVEL=INFO
```

### 5.5 Comandos de Uso

```bash
# Construir y levantar el servicio (primera vez descarga modelos ~15GB)
docker compose up --build

# Levantar en segundo plano
docker compose up -d

# Ver logs
docker compose logs -f coderag

# Parar el servicio
docker compose down

# Limpiar datos (reset completo - NO borra modelos descargados)
docker compose down -v
rm -rf ./data ./repos

# Limpiar TODO incluyendo modelos (re-descargará ~15GB)
docker compose down -v
rm -rf ./data ./repos ./models ./adapters
```

### 5.6 Persistencia y Cache

**Volumen `./data`**: Almacena los índices vectoriales (ChromaDB). Permite que un repositorio indexado persista entre reinicios del contenedor.

**Volumen `./repos`**: Cache de repositorios clonados. Evita re-clonar el mismo repo si ya existe localmente.

**Volumen `./models`**: Cache de modelos de HuggingFace. **IMPORTANTE**: Evita re-descargar ~15GB cada vez que se reconstruye el contenedor.

**Volumen `./adapters`**: Almacena adaptadores LoRA entrenados. Se mantienen fuera de Git.

```python
# Lógica de cache de repos
def get_or_clone_repo(repo_url: str, branch: str) -> Path:
    """Obtiene repo del cache o lo clona si no existe."""
    repo_id = hash_repo_url(repo_url, branch)
    cache_path = Path(os.environ["REPOS_DIR"]) / repo_id
    
    if cache_path.exists():
        # Actualizar repo existente (git pull)
        update_repo(cache_path)
        return cache_path
    else:
        # Clonar nuevo repo
        return clone_repo(repo_url, branch, cache_path)
```

---

## 6. Requisitos de Hardware

### 6.1 Configuración Mínima (MVP)

| Componente | Especificación | Notas |
|------------|----------------|-------|
| **GPU** | NVIDIA RTX 4060 8GB | Cuantización 4-bit requerida |
| **RAM** | 16GB | 32GB recomendado para repos grandes |
| **Storage** | 50GB SSD | Modelos (~15GB) + índices + repos |
| **CUDA** | 12.1+ | Requerido para inferencia GPU |

### 6.2 Uso de VRAM Estimado

| Componente | VRAM |
|------------|------|
| Qwen2.5-Coder-7B (4-bit) | ~4.5GB |
| nomic-embed-text v1.5 | ~0.5GB |
| Overhead CUDA | ~1GB |
| **Total** | **~6GB** |

**Margen disponible**: ~2GB para batches de embeddings y contextos largos.

### 6.3 Alternativa Ligera

Si la VRAM es insuficiente, usar Llama-3.2-3B-Instruct:

| Componente | VRAM |
|------------|------|
| Llama-3.2-3B (4-bit) | ~2GB |
| nomic-embed-text v1.5 | ~0.5GB |
| Overhead | ~1GB |
| **Total** | **~3.5GB** |

---

## 7. Decisiones de Alcance MVP

### 7.1 Incluido en MVP

| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| Repos públicos | Solo GitHub público | Simplifica auth, suficiente para demo |
| Interfaz Gradio | UI web con botones | UX simple, rápido de implementar |
| Q&A con citas | Preguntas → respuestas citadas | Core value proposition |
| Chunking Python | AST-aware para .py | Diferenciador técnico |
| Docker local | 100% reproducible | Portafolio profesional |
| Persistencia | Índices en volumen | No re-indexar cada vez |
| Modelos locales | Qwen2.5-Coder + nomic-embed | Sin costos de API |

### 7.2 Excluido del MVP (Futuro)

| Feature | Razón de exclusión | Fase futura |
|---------|-------------------|-------------|
| Repos privados | Requiere GITHUB_TOKEN, complejidad auth | v1.1 |
| Modo Patch/Diff | Requiere validación de código, tests | v2.0 |
| Multi-repo | Un índice por repo es más simple | v1.2 |
| Fine-tuning | Primero validar RAG base funciona | v1.5 |
| Reranking | Optimización, no core | v1.3 |
| Auth de usuarios | No necesario para demo local | v2.0 |

### 7.3 Validación de URL GitHub

```python
import re
from urllib.parse import urlparse

def validate_github_url(url: str) -> tuple[str, str]:
    """
    Valida URL de GitHub y extrae owner/repo.
    
    Args:
        url: URL del repositorio (ej: https://github.com/owner/repo)
        
    Returns:
        Tuple de (owner, repo_name)
        
    Raises:
        ValueError: Si la URL no es válida o no es de GitHub
    """
    parsed = urlparse(url)
    
    # Validar dominio
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise ValueError(f"Solo se soportan repos de GitHub. Dominio recibido: {parsed.netloc}")
    
    # Extraer owner/repo del path
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"URL inválida. Formato esperado: https://github.com/owner/repo")
    
    owner, repo = path_parts[0], path_parts[1]
    
    # Limpiar .git si existe
    repo = repo.removesuffix(".git")
    
    # Validar caracteres
    if not re.match(r"^[\w\-\.]+$", owner) or not re.match(r"^[\w\-\.]+$", repo):
        raise ValueError(f"Nombre de owner o repo contiene caracteres inválidos")
    
    return owner, repo
```

---

## 8. Esquema de Datos

### 8.1 Documento (Pre-chunking)

```python
@dataclass
class Document:
    content: str
    metadata: DocumentMetadata

@dataclass
class DocumentMetadata:
    file_path: str          # Ruta relativa al root del repo
    file_name: str          # Nombre del archivo
    extension: str          # Extensión (.py, .md, etc.)
    size_bytes: int         # Tamaño del archivo
    last_modified: datetime # Última modificación
    repo_url: str           # URL del repositorio (si aplica)
    commit_hash: str        # Hash del commit (opcional)
    branch: str             # Branch (opcional)
```

### 8.2 Chunk (Post-chunking)

```python
@dataclass
class Chunk:
    id: str                 # UUID único
    content: str            # Contenido del chunk
    embedding: List[float]  # Vector embedding (768 dims para nomic)
    metadata: ChunkMetadata

@dataclass
class ChunkMetadata:
    file_path: str          # Ruta del archivo origen
    start_line: int         # Línea de inicio
    end_line: int           # Línea de fin
    chunk_type: str         # "function", "class", "text", etc.
    
    # Para código Python
    name: Optional[str]             # Nombre de función/clase
    parent_class: Optional[str]     # Clase padre si es método
    signature: Optional[str]        # Firma de la función
    docstring: Optional[str]        # Docstring si existe
    decorators: List[str]           # Lista de decoradores
    
    # Para documentación
    section_title: Optional[str]    # Título de sección
    heading_level: Optional[int]    # Nivel de heading (h1, h2, etc.)
    
    # Trazabilidad
    commit_hash: Optional[str]
    indexed_at: datetime
```

### 8.3 Query y Respuesta

```python
@dataclass
class Query:
    text: str               # Pregunta del usuario
    top_k: int = 5          # Número de chunks a recuperar
    filters: Dict = None    # Filtros opcionales (por archivo, tipo, etc.)

@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float
    rank: int

@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    
    def __str__(self):
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"

@dataclass
class Response:
    answer: str                         # Respuesta generada
    citations: List[Citation]           # Lista de citas usadas
    retrieved_chunks: List[RetrievedChunk]  # Chunks recuperados
    confidence: float                   # Confianza en la respuesta (0-1)
    grounded: bool                      # True si está fundamentada en chunks
```

---

## 9. Estrategia de Evaluación

### 9.1 Prueba Cerrada (Respuestas en Docs)

**Objetivo**: Verificar que el sistema recupera el chunk correcto y la respuesta es fiel al texto.

**Metodología**:
1. Crear set de preguntas donde la respuesta está literalmente en el repo
2. Para cada pregunta, definir:
   - Archivo(s) esperado(s)
   - Rango de líneas esperado
   - Contenido clave que debe aparecer en la respuesta

**Métricas**:
- **Retrieval Accuracy**: ¿El chunk correcto está en top-k?
- **Faithfulness**: ¿La respuesta se mantiene fiel al contenido recuperado?
- **Citation Accuracy**: ¿Las citas apuntan a los lugares correctos?

### 9.2 Prueba Abierta (Fuera de Docs)

**Objetivo**: Verificar que el modelo dice "no está en la base" cuando corresponde.

**Metodología**:
1. Crear preguntas sobre cosas que NO están en el repo
2. El sistema debe responder indicando que no tiene información

**Métricas**:
- **Abstention Rate**: ¿Con qué frecuencia se abstiene correctamente?
- **Hallucination Detection**: ¿Inventa información cuando no debería?

### 9.3 Dataset de Evaluación (Formato JSONL)

```jsonl
{"id": "q001", "type": "closed", "question": "¿Dónde se define la función authenticate_user?", "expected_files": ["src/auth/handlers.py"], "expected_line_range": [45, 78], "expected_keywords": ["authenticate_user", "password", "hash"]}
{"id": "q002", "type": "closed", "question": "¿Qué parámetros recibe process_payment?", "expected_files": ["src/payments/processor.py"], "expected_line_range": [112, 145], "expected_keywords": ["amount", "currency", "card_token"]}
{"id": "q003", "type": "open", "question": "¿Cómo se conecta a MongoDB?", "expected_behavior": "abstain", "reason": "No hay conexión a MongoDB en este repo"}
{"id": "q004", "type": "open", "question": "¿Cuál es el endpoint para eliminar usuarios?", "expected_behavior": "abstain", "reason": "No existe endpoint DELETE /users"}
```

### 9.4 Framework de Evaluación Automatizada

**Métricas a implementar**:

1. **Retrieval Metrics**:
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)

2. **Generation Metrics**:
   - Faithfulness Score (usando modelo evaluador)
   - Answer Relevancy
   - Citation Precision/Recall

3. **End-to-End Metrics**:
   - Correctness (respuesta correcta)
   - Groundedness (fundamentada en contexto)
   - Abstention Accuracy (se abstiene cuando debe)

---

## 10. Fine-tuning para Comportamiento Grounded

### 10.1 Objetivo del Fine-tune

**NO** entrenar conocimiento del repo (eso viene del retrieval).
**SÍ** entrenar el comportamiento:
- Incluir citas por afirmación
- Responder en formato estructurado
- Abstenerse cuando no hay evidencia

### 10.2 Técnica: QLoRA

**Por qué QLoRA**:
- Permite ajustar modelo cuantizado (4-bit)
- Solo entrena adaptadores LoRA (~100MB de salida)
- Baja memoria, viable en RTX 4060 8GB
- No toca todos los pesos del modelo base

**Configuración típica**:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rango de las matrices LoRA
    lora_alpha=32,           # Factor de escalado
    target_modules=[         # Qué módulos adaptar
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 10.3 Dataset de Entrenamiento (Formato)

```jsonl
{
  "instruction": "Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado. Incluye citas en formato [archivo:líneas] para cada afirmación.",
  "context": "[Chunk 1: src/auth/handlers.py:45-78]\ndef authenticate_user(self, username: str, password: str) -> bool:\n    \"\"\"Authenticate user with username and password.\"\"\"\n    hashed = hashlib.sha256(password.encode()).hexdigest()\n    return self.db.verify_credentials(username, hashed)\n\n[Chunk 2: src/models/user.py:12-35]\nclass User:\n    def __init__(self, username, email):\n        self.username = username\n        self.email = email",
  "question": "¿Cómo se verifica la contraseña del usuario?",
  "response": "La verificación de contraseña se realiza en la función `authenticate_user` [src/auth/handlers.py:45-78]:\n\n- Se hashea la contraseña usando SHA-256 [src/auth/handlers.py:48]\n- Se verifica contra la base de datos llamando a `db.verify_credentials()` [src/auth/handlers.py:49]\n\nEl hash se genera con `hashlib.sha256(password.encode()).hexdigest()`."
}
```

```jsonl
{
  "instruction": "Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está disponible, indícalo claramente.",
  "context": "[Chunk 1: src/auth/handlers.py:45-78]\ndef authenticate_user(...)...\n[Chunk 2: src/models/user.py:12-35]\nclass User:...",
  "question": "¿Cómo se envían notificaciones por email?",
  "response": "No encontré información sobre el envío de notificaciones por email en el contexto proporcionado. Los chunks recuperados solo contienen información sobre autenticación de usuarios y el modelo User, pero no hay código relacionado con notificaciones o emails."
}
```

### 10.4 Contrato de Comportamiento Post Fine-tune

El modelo fine-tuneado debe:

1. **Siempre citar**: Cada afirmación factual → cita
2. **Formato consistente**: `[archivo:línea_inicio-línea_fin]`
3. **Abstenerse apropiadamente**: Si no hay evidencia → decir que no está
4. **No alucinar**: Nunca inventar archivos, funciones o código
5. **Ser conciso**: Respuestas estructuradas, no verbosas

### 10.5 Artefactos de Fine-tuning (NO publicar)

El entrenamiento QLoRA genera estos archivos que **NO deben subirse a Git**:

```
adapters/
├── qwen-coder-grounded/
│   ├── adapter_model.safetensors  # ~100MB - NO publicar
│   ├── adapter_config.json        # Config del adaptador
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
```

**En el repo SÍ publicar**:
- `scripts/train_qlora.py` - Script de entrenamiento
- `configs/qlora_config.yaml` - Hiperparámetros
- `README.md` con métricas antes/después del fine-tune

---

## 11. Stack Tecnológico

### 11.1 Backend / Core

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Framework | FastAPI | Async, moderno, buena documentación |
| Orquestación RAG | LangChain o LlamaIndex | Ecosistema maduro, integraciones |
| Parser Python | Tree-sitter (py-tree-sitter) | Chunking semántico preciso |
| **Embeddings** | **nomic-embed-text v1.5** | Local, Apache 2.0, 768 dims |
| **Vector DB** | **ChromaDB** | Simple para MVP, buena DX |
| **LLM** | **Qwen2.5-Coder-7B-Instruct** | Local, Apache 2.0, especializado en código |

### 11.2 Dependencias Python Principales

```toml
[project]
dependencies = [
    # Web Framework
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    
    # UI
    "gradio>=4.0.0",
    
    # LLM Local
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    "bitsandbytes>=0.43.0",  # Cuantización 4-bit
    "torch>=2.2.0",
    
    # Embeddings
    "sentence-transformers>=2.5.0",
    
    # Vector DB
    "chromadb>=0.4.0",
    
    # RAG (opcional, para utilidades)
    "langchain>=0.1.0",
    "langchain-community>=0.0.20",
    
    # Code Parsing
    "tree-sitter>=0.20.0",
    "tree-sitter-python>=0.20.0",
    
    # Git
    "gitpython>=3.1.0",
    
    # Utils
    "tiktoken>=0.5.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.25.0",
]

[project.optional-dependencies]
finetune = [
    "peft>=0.10.0",           # LoRA/QLoRA
    "trl>=0.8.0",             # Trainer para LLMs
    "datasets>=2.18.0",
    "wandb>=0.16.0",          # Logging (opcional)
]

dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```

### 11.3 Estructura de Proyecto

```
coderag/
├── src/
│   └── coderag/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app + Gradio mount
│       ├── config.py               # Configuración (env vars, defaults)
│       │
│       ├── ui/                     # Interfaz Gradio
│       │   ├── __init__.py
│       │   ├── app.py              # Definición de la UI Gradio
│       │   ├── components.py       # Componentes reutilizables
│       │   └── handlers.py         # Handlers de eventos (indexar, preguntar)
│       │
│       ├── ingestion/              # Carga y procesamiento de repos
│       │   ├── __init__.py
│       │   ├── loader.py           # Carga de repositorios (clone GitHub)
│       │   ├── filter.py           # Filtrado de archivos
│       │   ├── chunker.py          # Chunking (AST + texto)
│       │   └── validator.py        # Validación de URLs GitHub
│       │
│       ├── indexing/               # Embeddings y vector DB
│       │   ├── __init__.py
│       │   ├── embeddings.py       # Generación con nomic-embed
│       │   └── vectorstore.py      # Interacción con ChromaDB
│       │
│       ├── retrieval/              # Recuperación de chunks
│       │   ├── __init__.py
│       │   ├── retriever.py        # Lógica de retrieval
│       │   └── reranker.py         # Reranking (futuro)
│       │
│       ├── generation/             # Generación de respuestas
│       │   ├── __init__.py
│       │   ├── generator.py        # Generación con Qwen2.5-Coder
│       │   ├── prompts.py          # Templates de prompts
│       │   └── citations.py        # Parsing/formateo de citas
│       │
│       ├── evaluation/             # Framework de evaluación
│       │   ├── __init__.py
│       │   ├── metrics.py          # Métricas de evaluación
│       │   ├── evaluator.py        # Evaluador principal
│       │   └── datasets.py         # Carga de datasets de eval
│       │
│       └── api/                    # Endpoints API REST
│           ├── __init__.py
│           ├── routes.py           # Rutas de API
│           └── schemas.py          # Schemas Pydantic
│
├── scripts/                        # Scripts de utilidad
│   ├── train_qlora.py              # Script de fine-tuning (SÍ publicar)
│   ├── evaluate.py                 # Script de evaluación
│   └── download_models.py          # Pre-descarga de modelos
│
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_validator.py
│   └── fixtures/
│
├── eval_datasets/                  # Datasets de evaluación (SÍ publicar)
│   ├── closed_questions.jsonl
│   └── open_questions.jsonl
│
├── configs/                        # Configuraciones (SÍ publicar)
│   ├── default.yaml
│   ├── models.yaml
│   ├── filters.yaml
│   └── qlora_config.yaml
│
├── data/                           # ChromaDB (NO publicar - volumen Docker)
├── repos/                          # Cache repos (NO publicar - volumen Docker)
├── models/                         # Cache HF (NO publicar - volumen Docker)
├── adapters/                       # LoRA adapters (NO publicar - volumen Docker)
│
├── Dockerfile
├── docker-compose.yml
├── .env.example                    # Ejemplo de .env (SÍ publicar)
├── .dockerignore
├── .gitignore                      # Excluye data/, repos/, models/, adapters/
├── pyproject.toml
├── requirements.txt
└── README.md                       # Con sección "Model Licenses"
```

---

## 12. API Endpoints (MVP)

### 12.1 Indexación

```http
POST /api/v1/repos/index
Content-Type: application/json

{
  "source": "https://github.com/user/repo.git",
  "branch": "main",
  "filters": {
    "include": ["*.py", "*.md"],
    "exclude": ["tests/**"]
  }
}

Response 202:
{
  "job_id": "uuid",
  "status": "processing",
  "message": "Repository indexing started"
}
```

### 12.2 Consulta (Q&A)

```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "¿Dónde se define la autenticación de usuarios?",
  "repo_id": "uuid",
  "top_k": 5,
  "include_chunks": true
}

Response 200:
{
  "answer": "La autenticación de usuarios se define en el módulo `auth` [src/auth/handlers.py:45-78]...",
  "citations": [
    {
      "file": "src/auth/handlers.py",
      "start_line": 45,
      "end_line": 78,
      "snippet": "def authenticate_user(...)..."
    }
  ],
  "confidence": 0.92,
  "grounded": true,
  "retrieved_chunks": [...]
}
```

### 12.3 Estado de Repositorios

```http
GET /api/v1/repos

Response 200:
{
  "repositories": [
    {
      "id": "uuid",
      "url": "https://github.com/user/repo.git",
      "branch": "main",
      "indexed_at": "2024-01-15T10:30:00Z",
      "chunk_count": 342,
      "status": "ready"
    }
  ]
}
```

---

## 13. Plan de Desarrollo por Fases

### Fase 1: Setup + Docker + Core Pipeline (Semana 1-2)

**Objetivos**:
- [ ] Setup del proyecto (estructura, dependencias, configs)
- [ ] Dockerfile con soporte CUDA y docker-compose.yml funcionales
- [ ] Descarga y configuración de modelos locales (Qwen2.5-Coder + nomic-embed)
- [ ] Módulo de carga de repositorios (Git clone público)
- [ ] Validación de URLs de GitHub
- [ ] Filtrado de archivos con reglas por defecto
- [ ] Chunking básico por texto
- [ ] Integración con ChromaDB
- [ ] Embedding con nomic-embed-text e indexación básica
- [ ] Volúmenes Docker para persistencia

**Entregable**: `docker compose up` levanta el servicio y puede indexar un repo.

### Fase 2: Interfaz Gradio + Progreso (Semana 2-3)

**Objetivos**:
- [ ] UI Gradio básica con tabs (Indexar / Preguntar)
- [ ] Campo de URL + botón Indexar
- [ ] Barra de progreso durante indexación (`gr.Progress`)
- [ ] Opciones avanzadas (branch, top-k, filtros)
- [ ] Montar Gradio en FastAPI (`mount_gradio_app`)
- [ ] Feedback de estado (éxito/error)

**Entregable**: UI funcional donde el usuario puede pegar URL e indexar.

### Fase 3: Chunking Semántico Python (Semana 3-4)

**Objetivos**:
- [ ] Integrar Tree-sitter para Python
- [ ] Extraer funciones y clases como chunks
- [ ] Enriquecer metadatos (signature, docstring, decorators)
- [ ] Fallback a chunking por texto para otros archivos
- [ ] Tests unitarios del chunker

**Entregable**: Chunks semánticos de calidad para archivos Python.

### Fase 4: Generación con Citas + Chat UI (Semana 4-5)

**Objetivos**:
- [ ] Implementar módulo de generación con Qwen2.5-Coder (4-bit)
- [ ] Sistema de prompts para comportamiento grounded
- [ ] Parsing y formateo de citas `[archivo:líneas]`
- [ ] Tab "Preguntar" funcional en Gradio
- [ ] Mostrar respuesta + evidencia (chunks usados)
- [ ] Manejo de casos "no encontrado"

**Entregable**: Sistema funcional de Q&A con citas en UI.

### Fase 5: Evaluación y Refinamiento (Semana 5-6)

**Objetivos**:
- [ ] Crear dataset de evaluación (cerradas + abiertas)
- [ ] Implementar métricas de evaluación
- [ ] Benchmark del sistema
- [ ] Identificar áreas de mejora
- [ ] Documentación (README, docstrings)
- [ ] Demo video para portafolio

**Entregable**: Framework de evaluación funcional + documentación completa.

### Fase 6: Fine-tuning QLoRA (Semana 6+)

**Objetivos**:
- [ ] Preparar dataset de fine-tuning (ejemplos con citas)
- [ ] Configurar entrenamiento QLoRA
- [ ] Entrenar adaptador para comportamiento grounded
- [ ] Evaluar mejora post fine-tune
- [ ] Documentar resultados (métricas antes/después)
- [ ] (Opcional) Optimizar retrieval (reranking, hybrid search)

**Entregable**: Sistema optimizado listo para demo/portafolio + adaptador funcional.

---

## 14. Consideraciones Adicionales

### 14.1 Manejo de Errores

- Repositorio no accesible → Error claro con instrucciones
- Parser falla → Fallback a chunking por texto
- LLM no responde → Retry con backoff exponencial
- Chunks insuficientes → Indicar confianza baja
- GPU sin memoria → Mensaje claro, sugerir modelo más ligero

### 14.2 Configuración por Entorno

```yaml
# configs/default.yaml
ingestion:
  chunk_size: 1500
  chunk_overlap: 150
  max_file_size_kb: 500

retrieval:
  top_k: 5
  similarity_threshold: 0.7

generation:
  model: "Qwen/Qwen2.5-Coder-7B-Instruct"
  quantization: "4bit"
  temperature: 0.1
  max_new_tokens: 1024

embeddings:
  model: "nomic-ai/nomic-embed-text-v1.5"
  dimensions: 768
  normalize: true
```

### 14.3 Logging y Observabilidad

- Logs estructurados (JSON) con contexto de request
- Métricas de latencia por componente
- Tracking de uso de VRAM
- Trazas de retrieval para debugging

### 14.4 Seguridad

- No indexar archivos con secretos (.env, credentials)
- Sanitizar inputs de usuario
- Rate limiting en API
- Los volúmenes Docker quedan fuera del repo

---

## 15. Glosario

| Término | Definición |
|---------|------------|
| **RAG** | Retrieval-Augmented Generation: técnica que aumenta prompts con información recuperada |
| **Grounded** | Respuestas fundamentadas en evidencia, no alucinaciones |
| **Chunk** | Fragmento de texto/código indexado |
| **Embedding** | Representación vectorial de un texto |
| **Top-k** | Los K resultados más relevantes en una búsqueda |
| **QLoRA** | Quantized Low-Rank Adaptation: fine-tuning eficiente en memoria |
| **LoRA Adapter** | Pequeño conjunto de pesos que adapta un modelo base a una tarea |
| **AST** | Abstract Syntax Tree: representación estructural del código |
| **Tree-sitter** | Parser incremental para análisis de código |
| **Faithfulness** | Métrica de qué tan fiel es la respuesta al contexto |
| **Abstention** | Cuando el modelo se rehúsa a responder por falta de evidencia |
| **ChromaDB** | Base de datos vectorial open source para embeddings |
| **nomic-embed** | Modelo de embeddings open source bajo Apache 2.0 |
| **Qwen2.5-Coder** | LLM de Alibaba especializado en código, Apache 2.0 |
| **Gradio** | Framework Python para crear interfaces web para ML/AI |
| **FastAPI** | Framework web moderno y async para Python |
| **Docker Compose** | Herramienta para definir y ejecutar aplicaciones multi-contenedor |

---

## 16. Referencias y Recursos

### Documentación Técnica
- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Tree-sitter Documentation](https://tree-sitter.github.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (LoRA/QLoRA)](https://huggingface.co/docs/peft/)

### Modelos
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [nomic-embed-text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (alternativa)

### Papers Relevantes
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al.)

### Tutoriales Recomendados
- [Gradio + FastAPI Integration](https://www.gradio.app/guides/sharing-your-app#api-page)
- [Gradio Progress Bars](https://www.gradio.app/docs/gradio/progress)
- [QLoRA Fine-tuning Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

*Documento generado para servir como contexto completo del proyecto CodeRAG.*
*Última actualización: Diciembre 2024*
*Modelos: Qwen2.5-Coder-7B-Instruct (LLM) + nomic-embed-text v1.5 (Embeddings)*
*Vector DB: ChromaDB*
