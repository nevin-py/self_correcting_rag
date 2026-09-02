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
        assert data["email_verified"] is False
        assert "user_id" not in data  # identity only revealed after verification

    async def test_register_duplicate_email(self, client: httpx.AsyncClient):
        """A2: Duplicate email → 400."""
        email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
        payload = {"email": email, "password": "validpassword123"}
        resp1 = await client.post("/api/v1/auth/register", json=payload)
        assert resp1.status_code == 201

        resp2 = await client.post("/api/v1/auth/register", json=payload)
        # Unverified duplicate: re-registration resets the password and resends the OTP.
        assert resp2.status_code == 201

        # A verified duplicate is rejected.
        await client.post("/api/v1/auth/verify-email", json={"email": email, "code": "123456"})
        resp3 = await client.post("/api/v1/auth/register", json=payload)
        assert resp3.status_code == 400
        assert "already exists" in resp3.json()["detail"]

    async def test_register_echoes_otp_when_delivery_fails(self, client: httpx.AsyncClient):
        """Local dev: when mail cannot be delivered (non-production), the code
        is echoed as debug_otp so registration is testable without SMTP."""
        from unittest.mock import patch

        email = f"echo_{uuid.uuid4().hex[:8]}@example.com"
        with patch("app.auth.otp.send_email", lambda *a, **k: False):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "validpassword123"},
            )
        assert resp.status_code == 201
        assert resp.json()["debug_otp"], "undelivered OTP must be echoed in non-production"

        # And the echoed code actually verifies.
        ver = await client.post(
            "/api/v1/auth/verify-email",
            json={"email": email, "code": resp.json()["debug_otp"]},
        )
        assert ver.status_code == 200

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


@pytest.mark.asyncio
class TestVerifyEmailNoBypass:
    """Regression (Sprint 1): /verify-email must NEVER return a token pair
    without validating the OTP. The old code returned tokens for an already
    verified email without checking the code at all — full account takeover
    for anyone who knew a victim's email address."""

    async def test_already_verified_with_wrong_code_gets_no_tokens(
        self, client: httpx.AsyncClient, registered_user: dict
    ):
        resp = await client.post(
            "/api/v1/auth/verify-email",
            json={"email": registered_user["email"], "code": "999999"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert "access_token" not in body
        assert "refresh_token" not in body

    async def test_wrong_code_for_unverified_gets_no_tokens(
        self, client: httpx.AsyncClient
    ):
        email = f"noby_{uuid.uuid4().hex[:8]}@example.com"
        resp = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": "validpassword123"}
        )
        assert resp.status_code == 201

        resp = await client.post(
            "/api/v1/auth/verify-email", json={"email": email, "code": "000000"}
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "access_token" not in body
        assert "refresh_token" not in body
