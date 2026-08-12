"""Refresh token issue / rotate / revoke helpers."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import RefreshToken, User
from app.core.config import settings
from app.core.security import create_token_access


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def issue_token_pair(db: AsyncSession, user: User) -> dict:
    """Create access JWT + opaque refresh token (stored hashed)."""
    access = create_token_access({"sub": str(user.user_id)})
    raw_refresh = secrets.token_urlsafe(48)
    row = RefreshToken(
        user_id=user.user_id,
        token_hash=_hash_token(raw_refresh),
        expires_at=_naive_utc_now()
        + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(row)
    await db.commit()
    return {
        "access_token": access,
        "refresh_token": raw_refresh,
        "token_type": "bearer",
    }


async def rotate_refresh_token(db: AsyncSession, raw_refresh: str) -> dict:
    """Validate refresh token, revoke it, issue a new pair (rotation)."""
    token_hash = _hash_token(raw_refresh.strip())
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    row = result.scalar_one_or_none()
    now = _naive_utc_now()
    if not row or row.revoked_at is not None or row.expires_at < now:
        raise ValueError("Invalid or expired refresh token")

    user = await db.get(User, row.user_id)
    if not user or not user.is_active or not user.email_verified:
        raise ValueError("Invalid or expired refresh token")

    row.revoked_at = now
    new_raw = secrets.token_urlsafe(48)
    new_row = RefreshToken(
        user_id=user.user_id,
        token_hash=_hash_token(new_raw),
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_row)
    await db.flush()
    row.replaced_by = new_row.id
    await db.commit()

    access = create_token_access({"sub": str(user.user_id)})
    return {
        "access_token": access,
        "refresh_token": new_raw,
        "token_type": "bearer",
    }


async def revoke_refresh_token(db: AsyncSession, raw_refresh: str) -> None:
    token_hash = _hash_token(raw_refresh.strip())
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    row = result.scalar_one_or_none()
    if row and row.revoked_at is None:
        row.revoked_at = _naive_utc_now()
        await db.commit()
