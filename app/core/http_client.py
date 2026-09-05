"""Process-wide shared httpx.AsyncClient.

One client = one connection pool: TLS connections are reused across requests
(latency win) and the pool size bounds concurrent sockets/buffers (RAM win
under 20 concurrent queries). Previously every fetch created a fresh
AsyncClient — a new pool and TLS handshake per call.

Per-request timeouts are still passed at the call site via `timeout=`.
"""
from __future__ import annotations

import httpx

from app.core.config import settings

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared client, creating it lazily (event-loop safe)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=settings.HTTPX_MAX_CONNECTIONS,
                max_keepalive_connections=max(2, settings.HTTPX_MAX_CONNECTIONS // 2),
            ),
            timeout=httpx.Timeout(15.0),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SelfCorrectingRAG/1.0)"},
        )
    return _client


async def close_http_client() -> None:
    """Dispose the shared client on app shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
