"""Config module - loads .env via Pydantic Settings."""

import functools
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "sentence-transformers"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Add this line:
    GROQ_MODEL: str = "mixtral-8x7b-32768"

    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    MAX_RECURSION_DEPTH: int = 5
    TOP_K_CHUNKS: int = 20
    ROT_THRESHOLD: float = 0.4
    REPL_TIMEOUT_SECONDS: int = 10

    MAX_TOTAL_COST_USD: float = 1.0


@functools.lru_cache()
def get_settings() -> Settings:
    return Settings()