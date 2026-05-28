"""
Application settings loaded from environment variables / .env file.
All required fields will raise a clear error on startup if missing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
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

    # Message broker
    RABBITMQ_URL: str = Field(..., description="RabbitMQ AMQP connection URL")

    # S3 / MinIO
    BUCKET_NAME: str = Field(..., description="S3 bucket name for document storage")
    S3_ENDPOINT_URL: Optional[str] = Field(None, description="Custom S3 endpoint (e.g. MinIO)")
    AWS_DEFAULT_REGION: str = Field(default="ap-southeast-1")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None)
    AWS_SESSION_TOKEN: Optional[str] = Field(None)

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

    # LiteLLM (chatbot service)
    LITELLM_API_BASE: str = Field(
        default="http://localhost:4000/v1",
        description="OpenAI-compatible LLM API base URL for chatbot"
    )
    LITELLM_API_KEY: str = Field(default="fake", description="API key for LiteLLM")

    # Extract-then-answer: per-parent fact extraction with cache. When on, the
    # RAG node extracts atomic facts from each retrieved parent (cached by
    # parent_id), then answers from facts instead of raw parent text.
    ENABLE_FACT_EXTRACTION: bool = Field(
        default=True,
        description="Enable extract-then-answer pipeline with per-parent fact cache",
    )
    FACT_EXTRACTION_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Cheap/fast model for per-parent fact extraction (no thinking)",
    )

    # hier_v2 retrieval — final LLM selector picks the top_n most relevant
    # chunks (across mixed text_child / table_summary / table_segment) after
    # parent/table dedup. The /query endpoint returns the selector output.
    SELECTOR_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="LLM that picks the final top_n chunks for the search response",
    )
    SELECTOR_TOP_N: int = Field(
        default=10, ge=1, le=50,
        description="How many chunks the /query endpoint returns after selector",
    )
    SELECTOR_OVERFETCH_MULT: int = Field(
        default=3, ge=1, le=10,
        description="Over-fetch multiplier before dedup (raw fetch = top_k * this)",
    )

    # Data API self-reference (used by chatbot RAG to query its own KB endpoints)
    DATA_API_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL data-api is reachable at (used by chatbot RAG node)"
    )

    # Document parsing service
    DOCUMENT_PARSING_URL: str = Field(
        default="http://document-parsing:8002",
        description="Base URL of the document-parsing service (internal network)"
    )
    DOCUMENT_PARSING_CALLBACK_BASE: str = Field(
        default="http://data-api:8000",
        description="Base URL data-api is reachable at from document-parsing (used to build callback_url)"
    )

    # Server
    PORT: int = Field(default=8000, ge=1, le=65535)
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
