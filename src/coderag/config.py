"""Application configuration using pydantic-settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """LLM and embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # LLM Provider: "local", "openai", "groq", "anthropic", "openrouter"
    # Default to "groq" (free tier available, no GPU required)
    llm_provider: str = "groq"

    # API settings (for remote providers)
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None  # Custom API base URL

    # Model name (local or remote)
    llm_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_top_p: float = 0.95

    # Local model settings
    llm_use_4bit: bool = True
    llm_device_map: str = "auto"

    embedding_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 8  # Reduced for 8GB VRAM GPUs
    embedding_device: str = "auto"  # "auto" detects CUDA, falls back to CPU


class VectorStoreSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_")

    persist_directory: Path = Path("./data/chroma_db")
    collection_name: str = "coderag_chunks"
    distance_metric: str = "cosine"
    anonymized_telemetry: bool = False


class IngestionSettings(BaseSettings):
    """Repository ingestion configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    repos_cache_dir: Path = Path("./data/repos")
    max_file_size_kb: int = 500
    default_branch: str = "main"
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Large repository handling
    max_files_per_repo: int = 5000
    max_total_chunks: int = 50000
    batch_size: int = 100
    stream_processing: bool = True

    # Warning thresholds
    warn_files_threshold: int = 1000
    warn_chunks_threshold: int = 10000

    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs", "*.c", "*.cpp", "*.h"]
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/*.min.js",
            "**/*.min.css",
            "**/package-lock.json",
            "**/yarn.lock",
            "**/poetry.lock",
            "**/.env",
            "**/.env.*",
            "**/credentials*",
            "**/*secret*",
            "**/*password*",
        ]
    )


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    default_top_k: int = 5
    max_top_k: int = 20
    similarity_threshold: float = 0.3


class ServerSettings(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CodeRAG"
    app_version: str = "0.1.0"
    debug: bool = False
    data_dir: Path = Path("./data")

    models: ModelSettings = Field(default_factory=ModelSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.persist_directory.mkdir(parents=True, exist_ok=True)
        self.ingestion.repos_cache_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
