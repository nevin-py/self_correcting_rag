from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from app.core.config import settings
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import all models so Base.metadata.create_all() sees them.
import app.auth.models  # noqa: E402, F401
import app.agent.models  # noqa: E402, F401
import app.agent.message_models  # noqa: E402, F401
import app.observability.models  # noqa: E402, F401
import app.documents.models  # noqa: E402, F401


_sql_echo = (
    settings.SQL_ECHO
    if settings.ENVIRONMENT != "production"
    else False
)

_db_url = settings.DATABASE_URL
_engine_kwargs: dict = {"echo": _sql_echo}
_connect_args: dict = {}
# Supabase transaction pooler (6543) does not support prepared-statement
# caching — statement_cache_size=0 handles that. A PERSISTENT pool is still
# safe (and important): NullPool here opened a fresh TLS connection per
# request (~0.6-1.5s handshake), making every DB endpoint cost 1-3s.
if "pooler.supabase.com" in _db_url or ":6543/" in _db_url or _db_url.rstrip("/").endswith(":6543"):
    _connect_args["statement_cache_size"] = 0

# Bound the pool for ALL URLs: each asyncpg connection costs ~10-30MB under
# concurrency. pool_pre_ping + pool_recycle keep stale pooler connections
# from surfacing as errors after idle periods.
if "poolclass" not in _engine_kwargs:
    _engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    _engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_recycle"] = 300
if "supabase.co" in _db_url or "supabase.com" in _db_url:
    import ssl as _ssl

    # Pooler presents an intermediate chain that fails default verify.
    # Traffic is still TLS-encrypted (equivalent to sslmode=require).
    _ctx = _ssl.create_default_context()
    _ctx.check_hostname = False
    _ctx.verify_mode = _ssl.CERT_NONE
    _connect_args["ssl"] = _ctx

def make_engine():
    """Build a fresh engine with the same URL-specific tuning as the global one.

    Used by health probes so they never touch the module-level engine — a
    pooled asyncpg connection is bound to the event loop that created it and
    fails intermittently under test runners or multiple workers.
    """
    return create_async_engine(
        _db_url,
        connect_args=_connect_args,
        poolclass=NullPool,
        **{"echo": _engine_kwargs["echo"]},
    )


engine = create_async_engine(
    _db_url,
    connect_args=_connect_args,
    **_engine_kwargs,
)

AsyncLocalSession = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def _default_session_factory():
    """Production session factory — bound to the module-level engine."""
    return AsyncLocalSession


def get_session_factory():
    """
    Dependency that returns a session factory.

    Overridable in tests via `app.dependency_overrides[get_session_factory]`.
    Endpoints should call the returned factory to create short-lived sessions
    rather than importing AsyncLocalSession directly.
    """
    return AsyncLocalSession


async def get_db():
    """Legacy dependency — yields a session for simple CRUD endpoints."""
    async with AsyncLocalSession() as db:
        yield db
