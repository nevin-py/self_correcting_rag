import logging
from contextlib import asynccontextmanager
from typing import MutableMapping

from fastapi import FastAPI, Request
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.v1.api import api_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.limiter import limiter
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
    logging.getLogger(__name__).info("Shutting down — disposing DB engine and HTTP client")
    from app.core.http_client import close_http_client
    await close_http_client()
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

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: set[str],
        allow_credentials: bool = True,
        allow_vercel_previews: bool = False,
    ):
        self.app = app
        self.allowed_origins = allowed_origins
        self.allow_credentials = allow_credentials
        self.allow_vercel_previews = allow_vercel_previews

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers: MutableMapping[str, str] = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode() if b"origin" in headers else None
        method = scope.get("method", "")

        # Determine if origin is allowed
        origin = headers.get(b"origin", b"").decode() if b"origin" in headers else None
        allowed_origin = None
        if origin:
            if origin in self.allowed_origins:
                allowed_origin = origin
            elif origin.endswith(".vercel.app") and self.allow_vercel_previews:
                # Vercel serves every deployment on its own *.vercel.app
                # subdomain (previews + prod aliases). Handy in DEVELOPMENT so
                # new deploys are never blocked — but in production the origin
                # list must be exact: anyone can register an available
                # <name>.vercel.app subdomain, so a wildcard there would let
                # attacker-controlled sites pass CORS.
                allowed_origin = origin

        # ── Handle OPTIONS preflight ────────────────────────────────────
        if method == "OPTIONS" and b"access-control-request-method" in headers:
            response_headers = [
                (b"access-control-allow-origin", allowed_origin.encode() if allowed_origin else b""),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                (b"access-control-max-age", b"86400"),
            ]
            if self.allow_credentials and allowed_origin:
                response_headers.append((b"access-control-allow-credentials", b"true"))
            # Echo the browser's requested headers instead of replying "*" —
            # a wildcard Access-Control-Allow-Headers is rejected by browsers
            # when credentials are allowed.
            requested = headers.get(b"access-control-request-headers")
            if requested:
                response_headers.append((b"access-control-allow-headers", requested))
            else:
                response_headers.append(
                    (b"access-control-allow-headers", b"Authorization, Content-Type, Accept, X-Request-ID")
                )

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

def _validate_production_settings() -> None:
    """Fail fast on unsafe production configuration."""
    if settings.ENVIRONMENT != "production":
        return
    problems: list[str] = []
    if not settings.cors_origin_set:
        problems.append("CORS_ORIGINS must list your frontend origin(s) in production")
    # Email backend: validate whichever transport is actually selected.
    email_backend = (settings.EMAIL_BACKEND or "smtp").strip().lower()
    if email_backend == "brevo":
        if not (settings.BREVO_API_KEY or "").strip() or not (settings.BREVO_FROM or "").strip():
            problems.append("BREVO_API_KEY and BREVO_FROM are required in production (EMAIL_BACKEND=brevo)")
    elif email_backend == "resend":
        if not (settings.RESEND_API_KEY or "").strip() or not (settings.RESEND_FROM or "").strip():
            problems.append("RESEND_API_KEY and RESEND_FROM are required in production (EMAIL_BACKEND=resend)")
    else:
        if not (settings.SMTP_HOST or "").strip() or not (settings.SMTP_FROM or "").strip():
            problems.append("SMTP_HOST and SMTP_FROM are required in production (EMAIL_BACKEND=smtp)")
        if (settings.SMTP_USER or "").strip() and not (settings.SMTP_PASSWORD or "").strip():
            problems.append("SMTP_PASSWORD is required when SMTP_USER is set")
    weak_secrets = {
        "",
        "your_secret_key_here",
        "changeme",
        "secret",
        "supersecretkey123",
    }
    if settings.SECRET_KEY.strip().lower() in weak_secrets or len(settings.SECRET_KEY.strip()) < 32:
        problems.append("SECRET_KEY must be a strong value (openssl rand -hex 32)")
    if (settings.ENCRYPTION_KEY or "").strip() == settings.SECRET_KEY.strip():
        # Forces new deployments onto a dedicated user-key-encryption key so
        # SECRET_KEY rotation never bricks stored provider API keys.
        problems.append("ENCRYPTION_KEY must be set (openssl rand -hex 32) and differ from SECRET_KEY")
    if problems:
        raise RuntimeError("Production config invalid: " + "; ".join(problems))


_validate_production_settings()

app = FastAPI(
    title="Self-Correcting RAG",
    description=(
        "Agentic RAG with hallucination detection and self-repair. "
        "Created by Nevin Sunil Oommen."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# NOTE: limits are enforced via explicit @limiter.limit decorators on sensitive
# routes (auth in app/auth/router.py, queries in app/agent/router.py).
# Deliberately NOT using SlowAPIMiddleware: it raises RateLimitExceeded in the
# middleware layer, above the exception-handler stack, so a 429 would surface
# as an unhandled 500.

# ── CORS is the outermost layer — pure ASGI, not Starlette middleware ────────
if settings.ENVIRONMENT == "development":
    origins = _dev_origins | settings.cors_origin_set
    # Preview deploys are a dev convenience only — production requires exact
    # origins in CORS_ORIGINS (enforced by _validate_production_settings).
    allow_vercel_previews = True
else:
    origins = settings.cors_origin_set
    allow_vercel_previews = False
app.add_middleware(
    ASGICorsMiddleware,
    allowed_origins=origins,
    allow_credentials=True,
    allow_vercel_previews=allow_vercel_previews,
)

# ── Custom middleware (skip OPTIONS) ────────────────────────────────────────
# Starlette runs the LAST-added middleware first (outermost). RequestID must
# wrap StructuredLogging, so it is added AFTER it — otherwise every log line
# reads request_id="unknown" because the ID was never set when logging ran.
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_router)
app.include_router(health_router)
