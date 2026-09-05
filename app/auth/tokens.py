"""Refresh token issue / rotate / revoke helpers.

Security model:
- Refresh tokens are opaque, stored only as SHA-256 hashes, single-use.
- On successful rotation the used token is revoked and linked to its
  replacement (``replaced_by``) — a token family / chain.
- PRESENTING A REVOKED TOKEN is treated as theft evidence: the whole family
  (every live token of that user) is revoked and the caller gets 401.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import RefreshToken, User
from app.core.config import settings
from app.core.security import create_token_access


class RefreshTokenReuseError(Exception):
    """A previously-revoked refresh token was presented — likely theft."""


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


async def _revoke_all_for_user(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Revoke every live refresh token of a user (theft response / logout-all)."""
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=_naive_utc_now())
    )
    await db.commit()


async def revoke_all_for_user(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Public helper: kill every live session of a user."""
    await _revoke_all_for_user(db, user_id)


async def rotate_refresh_token(db: AsyncSession, raw_refresh: str) -> dict:
    """Validate refresh token, revoke it, issue a new pair (rotation).

    Raises:
        RefreshTokenReuseError: the token was already revoked (rotated before,
            logged out, or a password change killed it). Treated as theft —
            every live refresh token of the user is revoked as a side effect.
        ValueError: token unknown / expired / user ineligible.
    """
    token_hash = _hash_token(raw_refresh.strip())
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    row = result.scalar_one_or_none()
    now = _naive_utc_now()

    if row and row.revoked_at is not None:
        # Reuse of a revoked token: either a network race replaying the same
        # token, or a stolen token being replayed after the legitimate owner
        # rotated. Either way, the safe move is to kill the whole family.
        await _revoke_all_for_user(db, row.user_id)
        raise RefreshTokenReuseError("Refresh token reuse detected")

    if not row or row.expires_at < now:
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


async def user_id_for_refresh_token(db: AsyncSession, raw_refresh: str) -> uuid.UUID | None:
    """Resolve the owning user of a refresh token (for logout-all)."""
    token_hash = _hash_token(raw_refresh.strip())
    result = await db.execute(
        select(RefreshToken.user_id).where(RefreshToken.token_hash == token_hash)
    )
    return result.scalar_one_or_none()
