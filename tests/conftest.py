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

TEST_DATABASE_URL = "postgresql+asyncpg://ariva:nevin@localhost:5432/self_correcting_rag_test"


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


@pytest_asyncio.fixture
async def registered_user(client: httpx.AsyncClient) -> dict:
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    return {"email": email, "password": password, "user_id": resp.json()["user_id"]}


@pytest_asyncio.fixture
async def auth_headers(client: httpx.AsyncClient, registered_user: dict) -> dict:
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": registered_user["email"], "password": registered_user["password"]},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest_asyncio.fixture
async def second_user(client: httpx.AsyncClient) -> dict:
    email = f"test2_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword456"
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201
    return {"email": email, "password": password, "user_id": resp.json()["user_id"]}


@pytest_asyncio.fixture
async def second_auth_headers(client: httpx.AsyncClient, second_user: dict) -> dict:
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": second_user["email"], "password": second_user["password"]},
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
