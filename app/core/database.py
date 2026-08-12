from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import all models so Base.metadata.create_all() sees them.
import app.auth.models  # noqa: E402, F401
import app.agent.models  # noqa: E402, F401
import app.agent.message_models  # noqa: E402, F401
import app.documents.models  # noqa: E402, F401


_sql_echo = (
    settings.SQL_ECHO
    if settings.ENVIRONMENT != "production"
    else False
)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=_sql_echo,
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
