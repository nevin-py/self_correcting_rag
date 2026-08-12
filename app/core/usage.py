"""Usage event helpers for chat/query/tavily/ingest budgets."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import UsageEvent
from app.core.config import settings


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def record_usage(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    amount: int = 1,
) -> None:
    db.add(UsageEvent(user_id=user_id, kind=kind, amount=amount))
    await db.commit()


async def sum_usage_since(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    since: datetime,
) -> int:
    result = await db.execute(
        select(func.coalesce(func.sum(UsageEvent.amount), 0)).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= since,
        )
    )
    return int(result.scalar_one() or 0)


async def count_events_since(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    since: datetime,
) -> int:
    result = await db.execute(
        select(func.count()).select_from(UsageEvent).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= since,
        )
    )
    return int(result.scalar_one() or 0)


async def enforce_query_rate(db: AsyncSession, user_id: uuid.UUID) -> None:
    from fastapi import HTTPException

    since = _naive_utc_now() - timedelta(hours=1)
    used = await count_events_since(db, user_id, "query", since)
    if used >= settings.MAX_QUERIES_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Query limit reached ({settings.MAX_QUERIES_PER_HOUR}/hour).",
        )


async def enforce_tavily_budget(db: AsyncSession, user_id: uuid.UUID) -> None:
    from fastapi import HTTPException

    since = _naive_utc_now() - timedelta(days=1)
    used = await count_events_since(db, user_id, "tavily", since)
    if used >= settings.MAX_TAVILY_CALLS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily Tavily search budget reached ({settings.MAX_TAVILY_CALLS_PER_DAY}/day).",
        )


async def enforce_ingest_budget(
    db: AsyncSession, user_id: uuid.UUID, tokens: int
) -> None:
    from fastapi import HTTPException

    since = _naive_utc_now() - timedelta(days=1)
    used = await sum_usage_since(db, user_id, "ingest_tokens", since)
    if used + tokens > settings.MAX_INGEST_TOKENS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily ingest token budget exceeded "
                f"({settings.MAX_INGEST_TOKENS_PER_DAY}/day)."
            ),
        )
