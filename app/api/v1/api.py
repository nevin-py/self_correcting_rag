from fastapi import APIRouter
from app.auth.router import router as auth_router
from app.agent.router import router as agent_router
from app.documents.router import router as documents_router
from app.settings.router import router as settings_router
from app.memory.router import router as memory_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
api_router.include_router(documents_router)
api_router.include_router(agent_router)
api_router.include_router(settings_router)
api_router.include_router(memory_router)
