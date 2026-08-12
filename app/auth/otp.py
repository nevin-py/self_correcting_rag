"""Email OTP create / verify helpers."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import EmailOTP, User
from app.core.config import settings
from app.core.email import send_email
from app.core.security import hash_password, verify_password


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


async def create_and_send_otp(
    db: AsyncSession,
    user: User,
    purpose: str,
    *,
    subject: str,
    body_template: str,
) -> None:
    """Invalidate prior OTPs for purpose, create a new one, email the code."""
    now = datetime.now(UTC).replace(tzinfo=None)
    result = await db.execute(
        select(EmailOTP).where(
            EmailOTP.user_id == user.user_id,
            EmailOTP.purpose == purpose,
            EmailOTP.consumed_at.is_(None),
        )
    )
    for row in result.scalars().all():
        row.consumed_at = now

    code = _generate_code()
    otp = EmailOTP(
        user_id=user.user_id,
        purpose=purpose,
        code_hash=hash_password(code),
        expires_at=now + timedelta(minutes=settings.OTP_TTL_MINUTES),
        attempts=0,
    )
    db.add(otp)
    await db.commit()

    body = body_template.format(code=code, minutes=settings.OTP_TTL_MINUTES)
    send_email(user.email, subject, body)


async def latest_unconsumed_otp(
    db: AsyncSession, user_id, purpose: str
) -> EmailOTP | None:
    result = await db.execute(
        select(EmailOTP)
        .where(
            EmailOTP.user_id == user_id,
            EmailOTP.purpose == purpose,
            EmailOTP.consumed_at.is_(None),
        )
        .order_by(EmailOTP.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def verify_otp(
    db: AsyncSession,
    user: User,
    purpose: str,
    code: str,
) -> EmailOTP:
    """Validate OTP; raises ValueError with safe message on failure."""
    otp = await latest_unconsumed_otp(db, user.user_id, purpose)
    if not otp:
        raise ValueError("No active code. Request a new one.")

    now = datetime.now(UTC).replace(tzinfo=None)
    if otp.expires_at < now:
        raise ValueError("Code expired. Request a new one.")

    if otp.attempts >= settings.OTP_MAX_ATTEMPTS:
        raise ValueError("Too many attempts. Request a new code.")

    otp.attempts += 1
    await db.commit()

    if not verify_password(code.strip(), otp.code_hash):
        raise ValueError("Invalid code.")

    otp.consumed_at = now
    await db.commit()
    return otp


async def resend_allowed(db: AsyncSession, user: User, purpose: str) -> bool:
    otp = await latest_unconsumed_otp(db, user.user_id, purpose)
    if not otp or not otp.created_at:
        return True
    created = otp.created_at
    if created.tzinfo is not None:
        created = created.replace(tzinfo=None)
    elapsed = (datetime.now(UTC).replace(tzinfo=None) - created).total_seconds()
    return elapsed >= settings.OTP_RESEND_COOLDOWN_SECONDS
