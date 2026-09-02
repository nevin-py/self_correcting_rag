"""Sprint 2 security hardening tests.

Covers:
- Refresh-token reuse detection (family revocation)
- Session revocation on password change / reset
- Email-enumeration fixes (forgot-password, resend-otp)
- /auth/me identity probe
- Purge no longer resets quota ledgers (UsageEvents kept)
- Dedicated ENCRYPTION_KEY with legacy fallback decrypt
"""

import hashlib
import uuid

import httpx
import pytest

from app.core.config import settings


@pytest.mark.asyncio
class TestRefreshReuseDetection:
    """Presenting a revoked refresh token must revoke the WHOLE family."""

    async def _login(self, client, registered_user):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        assert resp.status_code == 200
        return resp.json()

    async def test_reused_token_kills_family(self, client, registered_user):
        pair_a = await self._login(client, registered_user)

        # Rotate A → B (normal flow).
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": pair_a["refresh_token"]}
        )
        assert resp.status_code == 200
        pair_b = resp.json()

        # Replay the already-rotated token A — theft signal.
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": pair_a["refresh_token"]}
        )
        assert resp.status_code == 401

        # The replacement token B must ALSO be dead now (family revoked).
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": pair_b["refresh_token"]}
        )
        assert resp.status_code == 401

        # And a fresh login still works (user not locked out).
        pair_c = await self._login(client, registered_user)
        assert pair_c["access_token"]


@pytest.mark.asyncio
class TestPasswordChangeRevokesSessions:
    async def test_change_password_kills_refresh_tokens(self, client, registered_user):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        old_refresh = resp.json()["refresh_token"]

        resp = await client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
            json={"current_password": registered_user["password"], "new_password": "newpassword456"},
        )
        assert resp.status_code == 200

        # The pre-change refresh token must be dead.
        resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert resp.status_code == 401

        # And the new password must work.
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": "newpassword456"},
        )
        assert resp.status_code == 200

    async def test_reset_password_kills_refresh_tokens(self, client, registered_user):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        old_refresh = resp.json()["refresh_token"]

        resp = await client.post("/api/v1/auth/forgot-password", json={"email": registered_user["email"]})
        assert resp.status_code == 200

        resp = await client.post(
            "/api/v1/auth/reset-password",
            json={
                "email": registered_user["email"],
                "code": "123456",  # deterministic OTP from conftest
                "new_password": "resetpassword789",
            },
        )
        assert resp.status_code == 200

        resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestEmailEnumerationFixes:
    async def test_forgot_password_cooldown_is_not_a_429(self, client, registered_user):
        """A cooldown hit must look identical to a success — 429 only for
        existing users confirmed which emails exist."""
        payload = {"email": registered_user["email"]}
        r1 = await client.post("/api/v1/auth/forgot-password", json=payload)
        assert r1.status_code == 200
        r2 = await client.post("/api/v1/auth/forgot-password", json=payload)
        assert r2.status_code == 200  # cooldown skipped the send silently
        assert r1.json()["detail"] == r2.json()["detail"]

    async def test_forgot_password_nonexistent_email_identical(self, client, registered_user):
        existing = await client.post(
            "/api/v1/auth/forgot-password", json={"email": registered_user["email"]}
        )
        nonexistent = await client.post(
            "/api/v1/auth/forgot-password", json={"email": f"ghost_{uuid.uuid4().hex[:6]}@example.com"}
        )
        assert existing.json() == nonexistent.json()

    async def test_resend_otp_verified_user_gets_generic_response(self, client, registered_user):
        resp = await client.post(
            "/api/v1/auth/resend-otp",
            json={"email": registered_user["email"], "purpose": "verify_email"},
        )
        assert resp.status_code == 200
        assert resp.json()["detail"] == "If that email exists, a code was sent."
        assert "verified" not in resp.json()["detail"].lower()


@pytest.mark.asyncio
class TestAuthMe:
    async def test_me_returns_identity(self, client, registered_user, auth_headers):
        resp = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == registered_user["user_id"]
        assert body["email"] == registered_user["email"]
        assert body["email_verified"] is True

    async def test_me_requires_auth(self, client):
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)


@pytest.mark.asyncio
class TestPurgeKeepsQuotaLedger:
    async def test_usage_events_survive_purge(self, client, auth_headers, registered_user, db_session):
        # Creating a chat records a chat_create usage event.
        resp = await client.post(
            "/api/v1/agent/chats", json={"title": f"ledger-{uuid.uuid4().hex[:6]}"}, headers=auth_headers
        )
        assert resp.status_code == 201

        from sqlalchemy import select
        from app.auth.models import UsageEvent

        session, _engine = db_session
        uid = uuid.UUID(registered_user["user_id"])
        before = len((await session.execute(select(UsageEvent).where(UsageEvent.user_id == uid))).scalars().all())
        assert before > 0

        resp = await client.post("/api/v1/agent/chats/purge", headers=auth_headers)
        assert resp.status_code == 200

        session.expire_all()
        after = len((await session.execute(select(UsageEvent).where(UsageEvent.user_id == uid))).scalars().all())
        assert after == before  # quota ledger must NOT be wiped


class TestEncryptionKeySeparation:
    """User API keys must survive SECRET_KEY rotation via ENCRYPTION_KEY."""

    def test_encrypt_under_new_key_decrypts(self, monkeypatch):
        from app.core import secrets as sec

        monkeypatch.setattr(settings, "ENCRYPTION_KEY", "a" * 64)
        token = sec.encrypt_secret("sk-or-user-key-123")
        assert sec.decrypt_secret(token) == "sk-or-user-key-123"

    def test_adopting_encryption_key_preserves_legacy_data(self, monkeypatch):
        """Legacy era: rows encrypted under the SECRET_KEY-derived key (before
        ENCRYPTION_KEY existed). After adopting a dedicated ENCRYPTION_KEY,
        those rows must still decrypt; new rows use the dedicated key."""
        from app.core import secrets as sec

        # Legacy era: data encrypted under the SECRET_KEY-derived key.
        monkeypatch.setattr(settings, "ENCRYPTION_KEY", "")
        legacy_token = sec.encrypt_secret("sk-or-legacy-key")

        # ENCRYPTION_KEY introduced afterwards (SECRET_KEY unchanged).
        monkeypatch.setattr(settings, "ENCRYPTION_KEY", "b" * 64)

        # Legacy fallback decrypt must still succeed.
        assert sec.decrypt_secret(legacy_token) == "sk-or-legacy-key"
        # New data goes out under the dedicated key.
        new_token = sec.encrypt_secret("sk-or-new-key")
        assert sec.decrypt_secret(new_token) == "sk-or-new-key"
        # And the legacy token re-wrapped under the current key round-trips.
        rewrapped = sec.encrypt_secret(sec.decrypt_secret(legacy_token))
        assert rewrapped != legacy_token
        assert sec.decrypt_secret(rewrapped) == "sk-or-legacy-key"

    def test_garbage_still_fails(self, monkeypatch):
        from app.core import secrets as sec

        monkeypatch.setattr(settings, "ENCRYPTION_KEY", "c" * 64)
        with pytest.raises(ValueError):
            sec.decrypt_secret("not-a-fernet-token")
