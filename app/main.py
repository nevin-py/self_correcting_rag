import logging
from contextlib import asynccontextmanager
from typing import MutableMapping

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.v1.api import api_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.middleware import (
    RequestIDMiddleware,
    StructuredLoggingMiddleware,
    MetricsMiddleware,
)
from app.core.database import engine

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("Starting Self-Correcting RAG v%s", app.version)
    yield
    logging.getLogger(__name__).info("Shutting down — disposing DB engine")
    await engine.dispose()


# ── Pure ASGI CORS middleware ────────────────────────────────────────────────
# Starlette's CORSMiddleware does NOT add headers to exception-handler
# responses (they bypass the middleware chain).  This ASGI middleware sits
# at the very top of the stack and guarantees CORS headers on EVERY
# response — including 400/401/429 from exception handlers.

_dev_origins = {
    "http://localhost:3000", "http://localhost:3001",
    "http://localhost:5173", "http://localhost:8080",
    "http://127.0.0.1:3000", "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
}


class ASGICorsMiddleware:
    """Pure ASGI CORS — handles preflights AND adds headers to ALL responses."""

    def __init__(self, app: ASGIApp, allowed_origins: set[str], allow_credentials: bool = True):
        self.app = app
        self.allowed_origins = allowed_origins
        self.allow_credentials = allow_credentials

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers: MutableMapping[str, str] = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode() if b"origin" in headers else None
        method = scope.get("method", "")

        # Determine if origin is allowed
        allowed_origin = origin if origin and origin in self.allowed_origins else None

        # ── Handle OPTIONS preflight ────────────────────────────────────
        if method == "OPTIONS" and b"access-control-request-method" in headers:
            response_headers = [
                (b"access-control-allow-origin", allowed_origin.encode() if allowed_origin else b""),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                (b"access-control-allow-headers", b"*"),
                (b"access-control-max-age", b"86400"),
            ]
            if self.allow_credentials and allowed_origin:
                response_headers.append((b"access-control-allow-credentials", b"true"))

            await send({
                "type": "http.response.start",
                "status": 204,
                "headers": response_headers,
            })
            await send({"type": "http.response.body"})
            return

        # ── Normal request — wrap send to inject CORS headers ───────────
        async def send_wrapper(message: MutableMapping) -> None:
            if message["type"] == "http.response.start":
                resp_headers = list(message.get("headers", []))
                if allowed_origin:
                    resp_headers.append((b"access-control-allow-origin", allowed_origin.encode()))
                    if self.allow_credentials:
                        resp_headers.append((b"access-control-allow-credentials", b"true"))
                    resp_headers.append((b"access-control-expose-headers", b"X-Request-ID"))
                message["headers"] = resp_headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ── App ─────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="Self-Correcting RAG",
    description="Agentic RAG with hallucination detection and self-repair.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS is the outermost layer — pure ASGI, not Starlette middleware ────────
origins = _dev_origins if settings.ENVIRONMENT == "development" else set()
app.add_middleware(ASGICorsMiddleware, allowed_origins=origins, allow_credentials=True)

# ── Custom middleware (skip OPTIONS) ────────────────────────────────────────
app.add_middleware(RequestIDMiddleware)
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_router)
app.include_router(health_router)
