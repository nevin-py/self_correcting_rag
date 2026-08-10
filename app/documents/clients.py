# app/documents/clients.py
from groq import Groq
from langchain_groq import ChatGroq
from nomic import login
import chromadb
from app.core.config import settings
from tavily import TavilyClient

# 1. Define LLM first so it is instantly available to graph.py
chat_llm = ChatGroq(
    api_key=settings.GROQ_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
)

groq_client = Groq(api_key=settings.GROQ_KEY)
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

# 2. Delay the Chroma and Nomic initialization until called, 
# or keep it at the bottom below all global definitions
def chroma_setup():
    chroma_client = chromadb.PersistentClient(path="./data/chroma")
    login(settings.NOMIC_API_KEY)
    return chroma_client

chroma_client = chroma_setup()
