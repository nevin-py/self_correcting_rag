from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title="Self-Correcting RAG",
    description=(
        "Agentic retrieval-augmented generation with hallucination detection, "
        "self-repair, and web search fallback."
    ),
    version="0.1.0",
)

# CORS — restrict to known origins in production
# In development, allow localhost variants. In production, set explicit origins.
_dev_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_dev_origins if settings.ENVIRONMENT == "development" else [],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Include all API routes under /api/v1
app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
