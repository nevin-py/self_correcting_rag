"""
Layer 6: Infrastructure tests.

Covers: health endpoint, OpenAPI spec, config loading, rate limiter presence.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.core.config import settings


@pytest.mark.asyncio
class TestHealth:
    """H1: Health endpoint."""

    async def test_health_returns_ok(self):
        """Health endpoint doesn't need auth or DB."""
        from app.main import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestOpenAPI:
    """H2: OpenAPI spec completeness."""

    def test_all_endpoints_registered(self):
        """H2: All endpoints appear in the OpenAPI spec."""
        schema = app.openapi()
        paths = set(schema["paths"].keys())
        expected = {
            "/api/v1/agent/chats",
            "/api/v1/agent/chats/{chat_id}",
            "/api/v1/agent/chats/{chat_id}/history",
            "/api/v1/agent/chats/{chat_id}/messages",
            "/api/v1/agent/chats/{chat_id}/query",
            "/api/v1/agent/chats/{chat_id}/query_stream",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/api/v1/auth/register",
            "/api/v1/documents/ingestions/{ingestion_id}",
            "/api/v1/documents/upload_file",
            "/health",
            "/health/ready",
            "/metrics",
        }
        assert expected <= paths

    def _refresh_token_for(self, client, registered_user) -> str | None:
        """Log in and return the refresh token issued to the user."""
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        return (resp.json() or {}).get("refresh_token")


    def test_auth_endpoints_have_post(self):
        """Register and login are POST."""
        schema = app.openapi()
        assert "post" in schema["paths"]["/api/v1/auth/register"]
        assert "post" in schema["paths"]["/api/v1/auth/login"]

    def test_agent_crud_endpoints(self):
        """Chats has POST + GET, single chat has GET + DELETE."""
        schema = app.openapi()
        assert "post" in schema["paths"]["/api/v1/agent/chats"]
        assert "get" in schema["paths"]["/api/v1/agent/chats"]
        assert "get" in schema["paths"]["/api/v1/agent/chats/{chat_id}"]
        assert "delete" in schema["paths"]["/api/v1/agent/chats/{chat_id}"]

    def test_upload_returns_202(self):
        """Upload endpoint declares 202 status code."""
        schema = app.openapi()
        upload_op = schema["paths"]["/api/v1/documents/upload_file"]["post"]
        assert "202" in upload_op.get("responses", {})


class TestConfig:
    """H3: Configuration loads correctly."""

    def test_settings_loaded(self):
        """Settings object is populated from .env."""
        assert settings.SECRET_KEY
        assert settings.DATABASE_URL
        assert settings.GROQ_KEY
        assert settings.NOMIC_API_KEY
        assert settings.TAVILY_API_KEY

    def test_guard_limits_positive(self):
        """All guard limits are positive integers."""
        assert settings.MAX_GRAPH_STEPS > 0
        assert settings.MAX_SEARCHES > 0
        assert settings.MAX_RETRIEVALS > 0
        assert settings.MAX_REGENERATIONS > 0

    def test_nomic_rate_limits(self):
        """Nomic rate limit settings are sane."""
        assert settings.NOMIC_CONCURRENCY >= 1
        assert settings.NOMIC_INTERVAL > 0
        assert settings.NOMIC_MAX_RETRIES >= 1
        effective_rate = settings.NOMIC_CONCURRENCY / settings.NOMIC_INTERVAL
        assert effective_rate <= 4.0, f"Effective rate {effective_rate} exceeds 4 req/s limit"


@pytest.mark.asyncio
class TestTokenRefresh:
    """Token refresh endpoint."""

    async def test_refresh_returns_valid_token(self, client, registered_user):
        login = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        refresh_token = login.json().get("refresh_token")
        assert refresh_token, "login must issue a refresh token"

        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        # Refreshed token should be usable for authenticated requests
        new_headers = {"Authorization": f"Bearer {data['access_token']}"}
        me_resp = await client.get("/api/v1/agent/chats", headers=new_headers)
        assert me_resp.status_code == 200

    async def test_refresh_requires_valid_token(self, client):
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": "invalid-token-value-12345"}
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestReadinessAndMetrics:
    """Health/ready and /metrics endpoints."""

    async def test_readiness(self, client):
        import socket

        from urllib.parse import urlparse

        from app.core.config import settings

        # Readiness probes the configured dev database. When it is not running
        # (common in CI), the endpoint must report failure via HTTP 503 so
        # orchestrators (Cloud Run / k8s / Docker) actually detect it — a 200
        # body saying "not ready" is invisible to probes.
        u = urlparse(settings.DATABASE_URL.replace("+asyncpg", ""))
        db_up = False
        try:
            with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=2):
                db_up = True
        except OSError:
            pass

        resp = await client.get("/health/ready")
        if not db_up:
            assert resp.status_code == 503
            assert resp.json()["status"] == "not ready"
        else:
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"

    async def test_metrics(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "http_requests_total" in text
        assert "process_uptime_seconds" in text


@pytest.mark.asyncio
class TestCORS:
    """CORS preflight and cross-origin behavior."""

    async def test_preflight_register_returns_200(self, client):
        """OPTIONS preflight on /register returns 200 with CORS headers."""
        resp = await client.options(
            "/api/v1/auth/register",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, Authorization",
            },
        )
        assert resp.status_code in (200, 204)
        assert resp.headers.get("access-control-allow-origin") is not None
        assert "POST" in resp.headers.get("access-control-allow-methods", "")

    async def test_preflight_agent_chats_returns_200(self, client):
        """OPTIONS preflight on /agent/chats returns 200 with CORS headers."""
        resp = await client.options(
            "/api/v1/agent/chats",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, Authorization",
            },
        )
        assert resp.status_code in (200, 204)
        assert resp.headers.get("access-control-allow-origin") is not None

    async def test_actual_request_includes_cors_headers(self, client):
        """GET /health from a browser-like origin includes CORS headers."""
        resp = await client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") is not None

    async def test_unlisted_origin_rejected(self, client):
        """Request from unlisted origin gets no CORS headers."""
        resp = await client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code == 200
        # In dev mode with restricted origins, evil.com should not get CORS headers
        assert resp.headers.get("access-control-allow-origin") is None


@pytest.mark.asyncio
class TestAuthenticatedRequests:
    """Verify authenticated requests work end-to-end through the middleware stack."""

    async def test_register_then_login_then_create_chat(self, client):
        """Full flow: register → login → create chat → verify ownership."""
        email = f"flow_{uuid.uuid4().hex[:8]}@example.com"

        # Register
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "testpassword123"},
        )
        assert reg.status_code == 201

        # Verify email via the deterministic test OTP (see conftest).
        ver = await client.post("/api/v1/auth/verify-email", json={"email": email, "code": "123456"})
        assert ver.status_code == 200
        token = ver.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create chat (this was returning 401 before the fix)
        chat = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Test Chat"},
            headers=headers,
        )
        assert chat.status_code == 201
        chat_id = chat.json()["chat_id"]

        # Verify the chat exists and belongs to this user
        get = await client.get(f"/api/v1/agent/chats/{chat_id}", headers=headers)
        assert get.status_code == 200
        assert get.json()["title"] == "Test Chat"

    async def test_unauthenticated_request_rejected(self, client):
        """Request without token → 401/403."""
        resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "No Auth"},
        )
        assert resp.status_code in (401, 403)

    async def test_expired_token_rejected(self, client):
        """Expired JWT → 401."""
        import jwt as pyjwt
        from app.core.config import settings
        from datetime import datetime, timedelta, UTC

        payload = {
            "sub": str(uuid.uuid4()),
            "exp": datetime.now(UTC) - timedelta(hours=1),
        }
        expired = pyjwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        resp = await client.post(
            "/api/v1/agent/chats",
            json={"title": "Expired"},
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert resp.status_code == 401
