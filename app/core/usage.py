"""Usage event helpers for chat/query/tavily/ingest budgets."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import UsageEvent
from app.core.config import settings


async def record_usage(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    amount: int = 1,
) -> None:
    db.add(UsageEvent(user_id=user_id, kind=kind, amount=amount))
    await db.commit()


async def sum_usage_in_last(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    window: timedelta,
) -> int:
    """Sum event amounts inside a trailing window on the DATABASE clock."""
    result = await db.execute(
        select(func.coalesce(func.sum(UsageEvent.amount), 0)).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= func.now() - window,
        )
    )
    return int(result.scalar_one() or 0)



async def count_events_in_last(
    db: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    window: timedelta,
) -> int:
    """Count events inside a trailing window measured on the DATABASE clock.

    ``created_at`` is filled by the server (func.now(), DB-local naive time).
    Comparing it against Python-side UTC breaks wherever the Postgres timezone
    differs from UTC — hourly caps would silently never trigger. Doing the
    arithmetic in SQL keeps both sides on the same clock.
    """
    result = await db.execute(
        select(func.count()).select_from(UsageEvent).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= func.now() - window,
        )
    )
    return int(result.scalar_one() or 0)


async def enforce_query_rate(db: AsyncSession, user_id: uuid.UUID) -> None:
    from fastapi import HTTPException

    used = await count_events_in_last(db, user_id, "query", timedelta(hours=1))
    if used >= settings.MAX_QUERIES_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Query limit reached ({settings.MAX_QUERIES_PER_HOUR}/hour).",
        )


async def enforce_tavily_budget(db: AsyncSession, user_id: uuid.UUID) -> None:
    from fastapi import HTTPException

    used = await count_events_in_last(db, user_id, "tavily", timedelta(days=1))
    if used >= settings.MAX_TAVILY_CALLS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily Tavily search budget reached ({settings.MAX_TAVILY_CALLS_PER_DAY}/day).",
        )


async def enforce_ingest_budget(
    db: AsyncSession, user_id: uuid.UUID, tokens: int
) -> None:
    from fastapi import HTTPException

    used = await sum_usage_in_last(db, user_id, "ingest_tokens", timedelta(days=1))
    if used + tokens > settings.MAX_INGEST_TOKENS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily ingest token budget exceeded "
                f"({settings.MAX_INGEST_TOKENS_PER_DAY}/day)."
            ),
        )
