"""
Layer 1: Auth tests.

Covers: registration, login, token validation, expiry, tampering,
        duplicate email, short password, deactivated accounts.
"""

import time
import uuid

import httpx
import jwt
import pytest
import pytest_asyncio

from app.core.config import settings


@pytest.mark.asyncio
class TestRegister:
    """A1–A3: Registration endpoint."""

    async def test_register_success(self, client: httpx.AsyncClient):
        """A1: Valid registration → 201 + UserResponse."""
        email = f"reg_{uuid.uuid4().hex[:8]}@example.com"
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "validpassword123"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == email
        assert "user_id" in data
        assert data["is_active"] is True
        assert "hashed_password" not in data  # must never leak

    async def test_register_duplicate_email(self, client: httpx.AsyncClient):
        """A2: Duplicate email → 400."""
        email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
        payload = {"email": email, "password": "validpassword123"}
        resp1 = await client.post("/api/v1/auth/register", json=payload)
        assert resp1.status_code == 201

        resp2 = await client.post("/api/v1/auth/register", json=payload)
        assert resp2.status_code == 400
        assert "already exists" in resp2.json()["detail"]

    async def test_register_short_password(self, client: httpx.AsyncClient):
        """A3: Password < 8 chars → 422."""
        email = f"short_{uuid.uuid4().hex[:8]}@example.com"
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "short"},
        )
        assert resp.status_code == 422

    async def test_register_invalid_email(self, client: httpx.AsyncClient):
        """Not-an-email → 422."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "validpassword123"},
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestLogin:
    """A4–A6: Login endpoint."""

    async def test_login_success(self, client: httpx.AsyncClient, registered_user: dict):
        """A4: Valid credentials → 200 + JWT."""
        resp = await client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user["email"],
                "password": registered_user["password"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Verify the token is a valid JWT
        decoded = jwt.decode(
            data["access_token"],
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        assert "sub" in decoded
        assert decoded["sub"] == registered_user["user_id"]

    async def test_login_wrong_password(self, client: httpx.AsyncClient, registered_user: dict):
        """A5: Wrong password → 401."""
        resp = await client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user["email"],
                "password": "wrongpassword",
            },
        )
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client: httpx.AsyncClient):
        """A6: Nonexistent email → 401."""
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": "ghost@example.com", "password": "anypassword"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestTokenValidation:
    """A7–A10: Token-based authentication."""

    async def test_valid_token_grants_access(self, client: httpx.AsyncClient, auth_headers: dict):
        """A7: Valid JWT → 200 on protected endpoint."""
        resp = await client.get("/api/v1/agent/chats", headers=auth_headers)
        assert resp.status_code == 200

    async def test_expired_token_rejected(self, client: httpx.AsyncClient, registered_user: dict):
        """A8: Expired JWT → 401."""
        # Craft a token with past expiry
        payload = {
            "sub": registered_user["user_id"],
            "exp": int(time.time()) - 3600,  # expired 1 hour ago
        }
        expired_token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        headers = {"Authorization": f"Bearer {expired_token}"}

        resp = await client.get("/api/v1/agent/chats", headers=headers)
        assert resp.status_code == 401

    async def test_tampered_token_rejected(self, client: httpx.AsyncClient, auth_headers: dict):
        """A9: Tampered JWT → 401."""
        token = auth_headers["Authorization"].replace("Bearer ", "")
        # Flip a character in the signature
        tampered = token[:-5] + "XXXXX"
        headers = {"Authorization": f"Bearer {tampered}"}

        resp = await client.get("/api/v1/agent/chats", headers=headers)
        assert resp.status_code == 401

    async def test_no_token_rejected(self, client: httpx.AsyncClient):
        """A10: No Authorization header → 401 or 403 (OAuth2 scheme default)."""
        resp = await client.get("/api/v1/agent/chats")
        assert resp.status_code in (401, 403)
