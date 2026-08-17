"""Central configuration for HRA-RLM.

This module keeps runtime configuration in one place so API credentials,
model selection, and retrieval/recursion limits are not scattered throughout
the codebase.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_MODEL")
    max_recursion_depth: int = Field(default=5, alias="MAX_RECURSION_DEPTH")
    top_k_chunks: int = Field(default=20, alias="TOP_K_CHUNKS")
    rot_threshold: float = Field(default=0.4, alias="ROT_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()