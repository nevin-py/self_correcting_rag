"""Rate limiting on auth endpoints (Sprint 1 security fix).

The conftest disables the shared limiter so the rest of the suite isn't
throttled; these tests re-enable it with a fresh in-memory storage window.
"""

import uuid

import httpx
import pytest

from app.core.limiter import limiter


@pytest.fixture
def limiter_enabled():
    """Enable the app limiter with a clean storage for the duration of one test."""
    limiter.enabled = True
    limiter.limiter.storage.reset()
    yield limiter
    limiter.enabled = False
    limiter.limiter.storage.reset()


@pytest.mark.asyncio
async def test_login_brute_force_blocked(
    client: httpx.AsyncClient, registered_user: dict, limiter_enabled
):
    """5 login attempts/min — the 6th must be 429, even with a valid email."""
    last = None
    for _ in range(5):
        last = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": "wrongpassword"},
        )
        assert last.status_code == 401

    sixth = await client.post(
        "/api/v1/auth/login",
        data={"username": registered_user["email"], "password": "wrongpassword"},
    )
    assert sixth.status_code == 429


@pytest.mark.asyncio
async def test_register_rate_limited(client: httpx.AsyncClient, limiter_enabled):
    """5 registrations/min — the 6th must be 429 (spam/OTP-exhaustion guard)."""
    last = None
    for _ in range(5):
        email = f"rl_{uuid.uuid4().hex[:8]}@example.com"
        last = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": "validpassword123"}
        )
        assert last.status_code == 201, last.text

    sixth = await client.post(
        "/api/v1/auth/register",
        json={"email": f"rl_{uuid.uuid4().hex[:8]}@example.com", "password": "validpassword123"},
    )
    assert sixth.status_code == 429


@pytest.mark.asyncio
async def test_resend_otp_rate_limited(client: httpx.AsyncClient, limiter_enabled):
    """3 resends/min — the 4th must be 429 (OTP request-flood guard)."""
    email = f"otp_{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "validpassword123"}
    )
    assert resp.status_code == 201

    # purpose=reset_password: no prior OTP exists for this purpose, so the
    # verify_email cooldown triggered by register doesn't interfere.
    first = await client.post("/api/v1/auth/resend-otp", json={"email": email, "purpose": "reset_password"})
    assert first.status_code == 200

    # Rapid re-requests: the endpoint's 60s cooldown now skips SILENTLY
    # (200, generic body — a 429 here would confirm the email exists), and
    # the @limiter.limit("3/minute") decorator 429s from the 4th request.
    second = await client.post("/api/v1/auth/resend-otp", json={"email": email, "purpose": "reset_password"})
    assert second.status_code == 200
    assert "debug_otp" not in second.json() or second.json()["debug_otp"] is None

    statuses = []
    for _ in range(3):
        resp = await client.post("/api/v1/auth/resend-otp", json={"email": email, "purpose": "reset_password"})
        statuses.append(resp.status_code)
    # Out of requests #3, #4, #5 at least the last must be limiter-blocked.
    assert statuses[-1] == 429, statuses
