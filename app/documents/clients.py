import logging

from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── LLM + search clients (fail fast — these are required) ────────────────────

# Main LLM — answer generation (higher quality, higher token cost)
chat_llm = ChatGroq(
    api_key=settings.GROQ_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Routing LLM — planner + hallucination checker
# 70B is smart enough to follow instructions and not loop wastefully
# 8K TPM on free tier — sufficient with our truncation limits
routing_llm = ChatGroq(
    api_key=settings.GROQ_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0,
)

groq_client = Groq(api_key=settings.GROQ_KEY)
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

# ── OpenRouter fallback (used when Groq hits rate limits) ────────────────────
# Model assignments:
#   Planner:            GPT-4o-mini (strict JSON/tool calling, cheap)
#   Generator:          DeepSeek V4 Flash 0731 (1M context, 20+ providers)
#   Hallucination:      DeepSeek V4 Flash 0731 (fast fact verification)
#   Fallback array:     DeepSeek → MiMo → Gemini Flash

openrouter_planner_llm = None
openrouter_generator_llm = None
openrouter_hallucination_llm = None

if settings.OPENROUTER_API_KEY:
    # Planner: GPT-4o-mini — strict structured output, near-zero rate limits
    openrouter_planner_llm = ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=settings.OPENROUTER_PLANNER_MODEL,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://self-correcting-rag.local",
            "X-Title": "Self-Correcting RAG",
        },
    )

    # Generator: DeepSeek V4 Flash — 1M context, ultra-cheap, 20+ provider failover
    openrouter_generator_llm = ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=settings.OPENROUTER_GENERATOR_MODEL,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://self-correcting-rag.local",
            "X-Title": "Self-Correcting RAG",
        },
    )

    # Hallucination checker: DeepSeek V4 Flash — fast fact verification
    openrouter_hallucination_llm = ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=settings.OPENROUTER_HALLUCINATION_MODEL,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://self-correcting-rag.local",
            "X-Title": "Self-Correcting RAG",
        },
    )

    logger.info("OpenRouter configured: planner=%s, generator=%s, hallucination=%s",
                settings.OPENROUTER_PLANNER_MODEL,
                settings.OPENROUTER_GENERATOR_MODEL,
                settings.OPENROUTER_HALLUCINATION_MODEL)

# ── ChromaDB + Nomic (lazy init — app starts even if these fail) ─────────────

_chroma_client = None
_nomic_logged_in = False
_init_attempted = False


def _ensure_nomic():
    """Log in to Nomic on first use. Idempotent."""
    global _nomic_logged_in
    if _nomic_logged_in:
        return
    from nomic import login as nomic_login
    nomic_login(settings.NOMIC_API_KEY)
    _nomic_logged_in = True
    logger.info("Nomic login successful")


def get_chroma_client():
    """
    Return the ChromaDB client, initializing on first call.
    Returns None if initialization fails (caller must handle).
    """
    global _chroma_client, _init_attempted
    if _chroma_client is not None:
        return _chroma_client
    if _init_attempted:
        return None
    _init_attempted = True
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path="./data/chroma")
        _ensure_nomic()
        logger.info("ChromaDB + Nomic initialized successfully")
    except Exception:
        logger.exception("Failed to initialize ChromaDB or Nomic — vector search will be unavailable")
        _chroma_client = None
    return _chroma_client


# Backward-compatible module-level name — resolves lazily on first access.
# Usage: from app.documents.clients import chroma_client
# The name is NOT in __dict__ initially, so __getattr__ fires on first import.
def __getattr__(name: str):
    if name == "chroma_client":
        return get_chroma_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
