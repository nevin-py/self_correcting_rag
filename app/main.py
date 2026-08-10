from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router

app = FastAPI(
    title="Self-Correcting RAG",
    description=(
        "Agentic retrieval-augmented generation with hallucination detection, "
        "self-repair, and web search fallback."
    ),
    version="0.1.0",
)

# CORS — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes under /api/v1
app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
