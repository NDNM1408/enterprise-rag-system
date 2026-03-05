"""FastAPI dependency injection configuration."""

from functools import lru_cache

from app.configurations.settings import Settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    from app.configurations.settings import settings
    return settings
