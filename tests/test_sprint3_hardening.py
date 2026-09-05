"""Sprint 3 hardening tests.

Covers:
- CORS: exact-origin matching, vercel-preview wildcard only in dev,
  preflight header echoing (never "*" with credentials)
- httpOnly refresh cookie: set on login, refresh works cookie-only,
  logout clears it
- Metrics: route-template labels (no UUID cardinality explosion)
"""

import uuid

import httpx
import pytest

from app.core.middleware import metrics  # noqa: F401  (sanity: middleware module imports)
from app.main import ASGICorsMiddleware


def _drive_cors(middleware_factory):
    """Build the middleware with a recording inner app; return (caller, captured)."""
    import asyncio

    captured = {}

    async def inner_app(scope, receive, send):
        captured["inner_called"] = True
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b"{}"})

    mw = middleware_factory(inner_app)

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
            captured["headers"] = {
                k.decode().lower(): v.decode()
                for k, v in message.get("headers", [])
            }

    def run(scope):
        captured.pop("inner_called", None)
        asyncio.run(mw(scope, receive, send))
        return captured

    return run


class TestCorsMiddleware:
    def _mw(self, *, vercel: bool):
        def factory(inner_app):
            return ASGICorsMiddleware(
                app=inner_app,
                allowed_origins={"https://app.example.com"},
                allow_credentials=True,
                allow_vercel_previews=vercel,
            )
        return factory

    def _scope(self, *, origin=None, method="GET", preflight=False):
        headers = []
        if origin:
            headers.append((b"origin", origin.encode()))
        if preflight:
            headers.append((b"access-control-request-method", b"POST"))
            headers.append((b"access-control-request-headers", b"authorization,content-type"))
        return {"type": "http", "method": method, "headers": headers, "path": "/"}

    def test_exact_origin_allowed(self):
        run = _drive_cors(self._mw(vercel=False))
        resp = run(self._scope(origin="https://app.example.com"))
        assert resp["headers"]["access-control-allow-origin"] == "https://app.example.com"
        assert resp["headers"]["access-control-allow-credentials"] == "true"

    def test_unknown_origin_gets_no_cors_headers(self):
        run = _drive_cors(self._mw(vercel=False))
        resp = run(self._scope(origin="https://evil.example.com"))
        assert "access-control-allow-origin" not in resp["headers"]

    def test_vercel_preview_blocked_when_disabled(self):
        """Production: anyone can register an available *.vercel.app name —
        the wildcard must not pass CORS."""
        run = _drive_cors(self._mw(vercel=False))
        resp = run(self._scope(origin="https://attacker-site.vercel.app"))
        assert "access-control-allow-origin" not in resp["headers"]

    def test_vercel_preview_allowed_when_enabled(self):
        run = _drive_cors(self._mw(vercel=True))
        resp = run(self._scope(origin="https://my-preview.vercel.app"))
        assert resp["headers"]["access-control-allow-origin"] == "https://my-preview.vercel.app"

    def test_preflight_echoes_requested_headers(self):
        """With credentials, Allow-Headers must be concrete — "*" is rejected
        by browsers in credentialed mode."""
        run = _drive_cors(self._mw(vercel=False))
        resp = run(self._scope(origin="https://app.example.com", method="OPTIONS", preflight=True))
        assert resp["status"] == 204
        assert resp["headers"]["access-control-allow-headers"] == "authorization,content-type"


@pytest.mark.asyncio
class TestRefreshCookie:
    async def test_login_sets_httponly_cookie(self, client, registered_user):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        assert resp.status_code == 200
        set_cookie = resp.headers.get("set-cookie", "")
        assert "refresh_token=" in set_cookie
        assert "httponly" in set_cookie.lower()
        assert "path=/api/v1/auth" in set_cookie.lower().replace(" path=", "path=")

    async def test_refresh_works_cookie_only(self, client, registered_user):
        await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        # No body token — the httpOnly cookie must carry the refresh token.
        resp = await client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code == 200, resp.text
        assert "access_token" in resp.json()
        # The rotated cookie must be re-set.
        assert "refresh_token=" in resp.headers.get("set-cookie", "")

    async def test_refresh_without_cookie_or_body_rejected(self, client):
        resp = await client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code == 401

    async def test_logout_clears_cookie_and_revokes(self, client, registered_user):
        await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        resp = await client.post("/api/v1/auth/logout", json={})
        assert resp.status_code == 200
        # Cookie deletion still arrives as a Set-Cookie for refresh_token.
        assert "refresh_token=" in resp.headers.get("set-cookie", "")
        # …and the old cookie token is revoked: refreshing again fails.
        resp = await client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code == 401

    async def test_logout_kills_all_user_sessions(self, client, registered_user):
        """Logout must revoke EVERY live refresh token of the user.

        Regression: a concurrent tab's in-flight rotation could commit after
        logout revoked the presented token, and its Set-Cookie landed after the
        cookie clear — leaving a fresh LIVE refresh token in the browser jar.
        Reopening the app then silently resurrected the session ("auto sign-in
        after I just open it"). User-wide revocation makes any such in-flight
        rotation worthless server-side.
        """
        login = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        assert login.status_code == 200
        # Second login = a second "device/tab" with its own live refresh token.
        login2 = await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        assert login2.status_code == 200

        # The first tab logs out.
        resp = await client.post("/api/v1/auth/logout", json={})
        assert resp.status_code == 200

        # The second tab's still-valid refresh token (captured from its login
        # response — the shared cookie jar can't hold both) must ALSO be dead:
        # a post-logout rotation can no longer resurrect the session.
        import re as _re
        m = _re.search(r"refresh_token=([^;]+)", login2.headers.get("set-cookie", ""))
        assert m, "no refresh cookie on second login"
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": m.group(1)}
        )
        assert resp.status_code == 401

    async def test_refresh_after_logout_is_reuse_not_resurrection(self, client, registered_user):
        """Simulates the exact race: refresh in flight while logout lands.
        The rotated-out token presented later must trigger reuse handling
        (401), never issue a new live pair."""
        await client.post(
            "/api/v1/auth/login",
            data={"username": registered_user["email"], "password": registered_user["password"]},
        )
        # Rotation 1 (the "in-flight" refresh that lands after logout).
        r1 = await client.post("/api/v1/auth/refresh", json={})
        assert r1.status_code == 200
        # Logout kills the whole family.
        await client.post("/api/v1/auth/logout", json={})
        # Any further refresh — old cookie, rotated cookie, anything — 401s.
        assert (await client.post("/api/v1/auth/refresh", json={})).status_code == 401


@pytest.mark.asyncio
class TestMetricsTemplates:
    async def test_no_uuid_cardinality(self, client, auth_headers):
        """A request to a parametrized route must be recorded under its
        TEMPLATE, never the raw path with the UUID baked in."""
        resp = await client.post(
            "/api/v1/agent/chats", json={"title": f"m-{uuid.uuid4().hex[:6]}"}, headers=auth_headers
        )
        assert resp.status_code == 201
        chat_id = resp.json()["chat_id"]

        from app.core.middleware import metrics
        snap = metrics.snapshot()
        # The chat-create route is recorded as a template…
        assert any(key.startswith("POST ") and key.endswith("/chats") for key in snap), list(snap)
        # …and the raw UUID must never become a label.
        assert not any(chat_id in key for key in snap)

    async def test_unmatched_paths_collapsed(self, client):
        """Scanner noise (random 404 paths) must collapse into one bucket."""
        await client.get(f"/no-such-path-{uuid.uuid4().hex[:8]}")
        await client.get(f"/no-such-path-{uuid.uuid4().hex[:8]}")
        from app.core.middleware import metrics
        snap = metrics.snapshot()
        assert "GET _unmatched" in snap
        assert snap["GET _unmatched"]["count"] >= 2
