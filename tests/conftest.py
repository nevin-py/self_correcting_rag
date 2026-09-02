"""
Shared fixtures for all test layers.
Fresh engine per test to avoid asyncpg event-loop binding issues.
"""

import uuid
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker

from app.core.database import Base, get_db, get_session_factory
from app.main import app

# Rate limiting is IP-keyed and every test shares one client IP — the real
# limits (5/min login, 5/min register, …) would randomly 429 the suite.
# Disable the shared limiter for tests; test_rate_limiting.py re-enables it.
from app.core.limiter import limiter as _app_limiter

_app_limiter.enabled = False

TEST_DATABASE_URL = "postgresql+asyncpg://ariva:nevin@localhost:5433/self_correcting_rag_test"


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[tuple[AsyncSession, AsyncEngine], None]:
    """Fresh async engine + session per test. Tables dropped and recreated on each test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Drop all tables and recreate to ensure clean state with latest schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session, engine

    await engine.dispose()


@pytest_asyncio.fixture
async def test_session_factory(db_session) -> AsyncGenerator[async_sessionmaker, None]:
    """Session factory bound to the TEST engine — for direct DB assertions in
    tests that must NOT touch the .env/dev database via AsyncLocalSession."""
    _, engine = db_session
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(db_session: tuple[AsyncSession, AsyncEngine]) -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client with both get_db and get_session_factory overridden to use the test engine."""
    session, engine = db_session

    def _test_session_factory():
        return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_session_factory] = _test_session_factory

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()


def _enable_fixed_otp() -> None:
    """Force a deterministic OTP code so tests can complete email verification.

    The real generator produces a random 6-digit code that is only emailed;
    tests cannot read it. Patching the generator keeps the production flow
    intact while making registration verifiable end-to-end.
    """
    from app.auth import otp as otp_module

    otp_module._generate_code = lambda: "123456"


_enable_fixed_otp()

# The claim↔evidence support gate loads a local ONNX encoder on first use; keep
# the offline suite network-free by default. tests/test_support_gate.py opts in
# with injected fake embeddings and re-enables the gate explicitly.
@pytest.fixture(autouse=True)
def _support_gate_off(monkeypatch):
    from app.agent import support as support_gate

    monkeypatch.setattr(support_gate, "gate_enabled", lambda: False)


async def _register_verified(client: httpx.AsyncClient, email: str, password: str) -> dict:
    """Register, verify via the fixed OTP, log in, and return identity + headers."""
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    resp = await client.post(
        "/api/v1/auth/verify-email", json={"email": email, "code": "123456"}
    )
    assert resp.status_code == 200, f"Email verification failed: {resp.text}"
    token = resp.json()["access_token"]
    import jwt as pyjwt

    from app.core.config import settings

    user_id = pyjwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])["sub"]
    return {"email": email, "password": password, "user_id": user_id, "token": token}


@pytest_asyncio.fixture
async def registered_user(client: httpx.AsyncClient) -> dict:
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    return await _register_verified(client, email, "testpassword123")


@pytest_asyncio.fixture
async def auth_headers(client: httpx.AsyncClient, registered_user: dict) -> dict:
    return {"Authorization": f"Bearer {registered_user['token']}"}


@pytest_asyncio.fixture
async def second_user(client: httpx.AsyncClient) -> dict:
    email = f"test2_{uuid.uuid4().hex[:8]}@example.com"
    return await _register_verified(client, email, "testpassword456")


@pytest_asyncio.fixture
async def second_auth_headers(client: httpx.AsyncClient, second_user: dict) -> dict:
    return {"Authorization": f"Bearer {second_user['token']}"}
