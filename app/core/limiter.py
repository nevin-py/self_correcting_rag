"""Shared slowapi rate limiter.

One Limiter instance for the whole app so that:
- per-route ``@limiter.limit(...)`` decorators (auth, agent) and any future
  global limits share a single storage and configuration;
- tests can disable all limiting at once via ``limiter.enabled = False``.

NOTE: ``Limiter(default_limits=...)`` never fires by itself — slowapi only
evaluates limits through route decorators (or SlowAPIMiddleware, which raises
RateLimitExceeded above the exception-handler layer and would surface as 500).
Sensitive endpoints therefore carry explicit ``@limiter.limit`` decorators.

Keying: ``get_rate_limit_key`` falls back to the leftmost ``X-Forwarded-For``
entry when ``TRUST_PROXY_HEADERS`` is enabled (reverse-proxy deployments);
otherwise it uses the socket peer address. The agent router reuses this for
its user-key fallback so both limiters agree on client identity.
"""

from fastapi import Request
from slowapi import Limiter

from app.core.config import settings


def get_client_ip(request) -> str | None:
    """Best-effort client IP, proxy-aware when TRUST_PROXY_HEADERS is on.

    Accepts a Request or a raw ASGI scope (dict). NOTE: the parameter MUST be
    named ``request`` — slowapi inspects the signature and only passes the
    request when it sees that exact name.
    """
    if hasattr(request, "headers"):
        headers = request.headers
        client = getattr(request, "client", None)
    else:
        headers = {
            k.decode("latin-1"): v.decode("latin-1")
            for k, v in request.get("headers", [])
        }
        client = request.get("client")

    if settings.TRUST_PROXY_HEADERS:
        fwd = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if fwd:
            # Leftmost entry = original client (set by our own proxy).
            leftmost = fwd.split(",")[0].strip()
            if leftmost:
                return leftmost
    return client.host if client else None


def get_rate_limit_key(request) -> str | None:
    return get_client_ip(request)


# Shared storage: in-memory by default (single process); set
# RATELIMIT_STORAGE_URI (e.g. redis://redis:6379/0) in production so limits
# survive restarts and are enforced across workers/instances.
_RATELIMIT_STORAGE_URI: str = settings.RATELIMIT_STORAGE_URI or "memory://"

limiter = Limiter(
    key_func=get_rate_limit_key,
    storage_uri=_RATELIMIT_STORAGE_URI,
    default_limits=["200/minute"],
)
