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

    # Database (shared with data-api)
    PGSQL_SCHEMA: str = "public"
    DATABASE_URL: str = Field(..., description="PostgreSQL async connection URL")

    # LiteLLM
    LITELLM_API_BASE: str = Field(
        default="http://localhost:4000/v1",
        description="OpenAI-compatible LLM API base URL"
    )
    LITELLM_API_KEY: str = Field(default="fake", description="API key for LiteLLM")

    # Data API
    DATA_API_URL: str = Field(
        default="http://localhost:8000",
        description="Data API base URL for KB queries"
    )

    # Server
    PORT: int = Field(default=8001, ge=1, le=65535)
    HOST: str = Field(default="0.0.0.0")

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


settings = Settings()
