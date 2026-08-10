# pyrefly: ignore [missing-import]
from joblib.numpy_pickle_compat import _CHUNK_SIZE
from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Literal
class Settings(BaseSettings):
    #security
    SECRET_KEY:str
    ALGORITHM:str="HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:int
    ENVIRONMENT:Literal["development","production"]='development'
    #db
    DATABASE_URL:str
    #AI agent keys
    GROQ_KEY:str
    NOMIC_API_KEY:str
        #ollama
    OLLAMA_BASE_URL:str
    OPENAI_API_KEY:str
        #tooling
    TAVILY_API_KEY:str
    CHUNK_SIZE:int
    CHUNK_OVERLAP:int
    model_config=SettingsConfigDict(env_file='.env',env_file_encoding="utf-8")
settings=Settings()