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

    # Chunking (tiktoken-based)
    TIKTOKEN_MODEL_NAME: str = Field(default="gpt-4o-mini", description="tiktoken model name for token counting")
    CHUNK_SIZE: int = Field(default=1200, ge=1, description="Maximum tokens per chunk")
    CHUNK_OVERLAP_SIZE: int = Field(default=100, ge=0, description="Overlapping tokens between consecutive chunks")

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

    # Neo4j (for graph storage)
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="neo4j", description="Neo4j password")
    NEO4J_DATABASE: str = Field(default="neo4j", description="Neo4j database name")

    # GraphRAG — LLM used for entity extraction
    GRAPHRAG_LLM_API_BASE: str = Field(
        default="http://localhost:4000/v1",
        description="OpenAI-compatible base URL for GraphRAG entity-extraction LLM"
    )
    GRAPHRAG_LLM_MODEL: str = Field(
        default="gemini/gemini-2.0-flash",
        description="LLM model name for GraphRAG entity extraction"
    )
    GRAPHRAG_LLM_API_KEY: str = Field(default="fake", description="API key for GraphRAG LLM")
    GRAPHRAG_WORKING_DIR: str = Field(
        default="/tmp/graphrag",
        description="Base directory for GraphRAG file-based caches"
    )

    # Postgres connection params for GraphRAG PGVector stores
    POSTGRES_HOST: str = Field(default="localhost", description="Postgres host for GraphRAG")
    POSTGRES_PORT: int = Field(default=5432, description="Postgres port for GraphRAG")
    POSTGRES_USER: str = Field(default="datahub", description="Postgres user for GraphRAG")
    POSTGRES_PASSWORD: str = Field(default="datahub", description="Postgres password for GraphRAG")
    POSTGRES_DATABASE: str = Field(default="datahub", description="Postgres database for GraphRAG")

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
