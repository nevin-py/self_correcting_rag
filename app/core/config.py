from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14
    ENVIRONMENT: Literal["development", "production"] = "development"

    # Comma-separated frontend origins for production CORS
    CORS_ORIGINS: str = ""

    # Database
    DATABASE_URL: str
    SQL_ECHO: bool = False

    # SMTP (OTP email verification / password reset)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""
    SMTP_TLS: bool = True
    OTP_TTL_MINUTES: int = 10
    OTP_MAX_ATTEMPTS: int = 5
    OTP_RESEND_COOLDOWN_SECONDS: int = 60

    # Anti-spam / quotas
    MAX_CHATS_PER_USER: int = 20
    MAX_CHAT_CREATES_PER_HOUR: int = 10
    MAX_QUERIES_PER_HOUR: int = 30
    MAX_INGEST_TOKENS_PER_DAY: int = 500_000
    MAX_TAVILY_CALLS_PER_DAY: int = 50
    MAX_FILE_TOKENS: int = 200_000
    MAX_CHUNKS_PER_FILE: int = 500

    # AI agent keys (system defaults; users may override LLM keys in Settings)
    GROQ_KEY: str = ""
    NOMIC_API_KEY: str

    # Google AI Studio (Gemini) — https://aistudio.google.com/apikey
    GOOGLE_AI_API_KEY: str = ""
    GOOGLE_AI_PLANNER_MODEL: str = "gemini-3.5-flash"
    GOOGLE_AI_GENERATOR_MODEL: str = "gemini-3.5-flash"
    GOOGLE_AI_HALLUCINATION_MODEL: str = "gemini-3.5-flash"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = ""

    # Tooling (system-only — not user-configurable)
    TAVILY_API_KEY: str
    SEARXNG_URL: str = "http://localhost:8888"
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_PLANNER_MODEL: str = "xiaomi/mimo-v2.5"
    OPENROUTER_GENERATOR_MODEL: str = "xiaomi/mimo-v2.5"
    OPENROUTER_HALLUCINATION_MODEL: str = "xiaomi/mimo-v2.5"

    # Nomic embedding rate limit
    NOMIC_CONCURRENCY: int = 2
    NOMIC_INTERVAL: float = 0.5
    NOMIC_MAX_RETRIES: int = 3

    # Graph execution guards
    MAX_GRAPH_STEPS: int = 20
    MAX_SEARCHES: int = 2
    MAX_RETRIEVALS: int = 3
    MAX_REGENERATIONS: int = 2

    USE_VERIFY_CASCADE: bool = True
    MAX_REPAIR_PASSES: int = 1
    NLI_ENTAIL_THRESHOLD: float = 0.7
    NLI_CONTRADICT_THRESHOLD: float = 0.7

    QUERY_TIMEOUT_SECONDS: int = 0
    STREAM_NODE_TIMEOUT_SECONDS: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_set(self) -> set[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return set()
        return {o.strip() for o in raw.split(",") if o.strip()}


settings = Settings()
