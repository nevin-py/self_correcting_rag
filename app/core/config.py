from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ENVIRONMENT: Literal["development", "production"] = "development"

    # Database
    DATABASE_URL: str

    # AI agent keys
    GROQ_KEY: str
    NOMIC_API_KEY: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = ""

    # Tooling
    TAVILY_API_KEY: str
    SEARXNG_URL: str = "http://localhost:8888"  # Local SearXNG instance for web search
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # OpenRouter (fallback when Groq hits rate limits)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_PLANNER_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_GENERATOR_MODEL: str = "deepseek/deepseek-v4-flash-0731"
    OPENROUTER_HALLUCINATION_MODEL: str = "deepseek/deepseek-v4-flash-0731"

    # Nomic embedding rate limit: 1200 requests / 5-minute rolling window / IP
    # Concurrency of 2 with a 0.25s gap → 4 req/s sustained, well under limit.
    NOMIC_CONCURRENCY: int = 2
    NOMIC_INTERVAL: float = 0.5
    NOMIC_MAX_RETRIES: int = 3

    # Graph execution guards — safety nets, not quality gates
    # Keep tight to avoid 5-6 minute query times
    MAX_GRAPH_STEPS: int = 12
    MAX_SEARCHES: int = 2
    MAX_RETRIEVALS: int = 3
    MAX_REGENERATIONS: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
