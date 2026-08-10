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
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()