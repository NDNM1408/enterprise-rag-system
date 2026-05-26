"""
Application settings loaded from environment variables / .env file.
All required fields will raise a clear error on startup if missing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Database
    PGSQL_SCHEMA: str = "public"
    DATABASE_URL: str = Field(..., description="PostgreSQL async connection URL")

    # Legacy individual DB parts (kept for backward compatibility, not used if DATABASE_URL is set)
    DB_DRIVER: Optional[str] = Field(None)
    DB_USERNAME: Optional[str] = Field(None)
    DB_PASSWORD: Optional[str] = Field(None)
    DB_HOST: Optional[str] = Field(None)
    DB_PORT: Optional[str] = Field(None)
    DB_NAME: Optional[str] = Field(None)

    # Message broker
    RABBITMQ_URL: str = Field(..., description="RabbitMQ AMQP connection URL")

    # S3 / MinIO
    PREPROCESS_BUCKET_NAME: str = Field(..., description="S3 bucket for raw documents")
    UPSERT_BUCKET_NAME: str = Field(..., description="S3 bucket for processed chunk text files")
    S3_ENDPOINT_URL: Optional[str] = Field(None, description="Custom S3 endpoint (e.g. MinIO)")
    AWS_DEFAULT_REGION: str = Field(default="ap-southeast-1")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None)
    AWS_SESSION_TOKEN: Optional[str] = Field(None)

    # External services
    DOCUMENT_AI_URL: str = Field(..., description="DocumentAI service URL for HTML parsing")
    GOOGLE_GENAI_API_KEY: Optional[str] = Field(None)

    # Chunking — markdown-aware parent-child splitter
    TIKTOKEN_MODEL_NAME: str = Field(default="gpt-4o-mini", description="tiktoken model name for token counting")
    # gemini-embedding-001 input cap is 2048 tokens; target keeps margin for
    # tokenizer drift between tiktoken (gpt-4o) and Gemini.
    RETRIEVE_MAX_TOKENS: int = Field(default=2048, ge=64, description="Hard upper bound for retrieve chunks (matches gemini-embedding-001 limit)")
    RETRIEVE_TARGET_TOKENS: int = Field(default=1800, ge=64, description="Greedy pack target for retrieve chunks")

    # Embedding API
    EMBEDDING_API_BASE: str = Field(
        default="http://localhost:4000/v1",
        description="OpenAI-compatible embedding API base URL"
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="rag-embedding-model",
        description="Embedding model name"
    )
    EMBEDDING_API_KEY: str = Field(default="fake")
    EMBEDDING_BATCH_SIZE: int = Field(default=4, ge=1, le=64)

    # Elasticsearch (llm-wiki rag_mode)
    ELASTICSEARCH_URL: str = Field(
        default="http://elasticsearch:9200",
        description="Elasticsearch endpoint for llm-wiki indexing/retrieval",
    )
    ELASTICSEARCH_USERNAME: str = Field(default="", description="ES username (empty = no auth)")
    ELASTICSEARCH_PASSWORD: str = Field(default="", description="ES password")
    ELASTICSEARCH_INDEX_PREFIX: str = Field(
        default="kb",
        description="Prefix for per-KB Elasticsearch indices (final name: <prefix>-<kb_id>)",
    )

    # Server (health-check FastAPI app)
    PORT: int = Field(default=8001, ge=1, le=65535)

    # Application mode
    MODE: str = Field(default="prod", description="Application mode: 'dev' or 'prod'")

    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.MODE.lower() == "dev"

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        valid_prefixes = (
            "postgresql://",
            "postgresql+asyncpg://",
            "postgresql+psycopg2://",
        )
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"DATABASE_URL must start with one of {valid_prefixes}, got: {v[:30]}..."
            )
        return v

    @field_validator("RABBITMQ_URL")
    @classmethod
    def validate_rabbitmq_url(cls, v: str) -> str:
        if not v.startswith(("amqp://", "amqps://")):
            raise ValueError(
                f"RABBITMQ_URL must start with amqp:// or amqps://, got: {v[:30]}..."
            )
        return v


settings = Settings()
