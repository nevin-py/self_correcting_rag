"""Email OTP create / verify helpers."""

from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
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
) -> tuple[str, bool]:
    """Invalidate prior OTPs, create a new one, attempt email delivery.

    Returns ``(code, delivered)``. Callers may echo the code in API responses
    when it was not delivered (non-production only — see ``should_echo_otp``).
    """
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
    # to_thread: both backends (SMTP handshake, HTTP call) block for seconds;
    # never hold the event loop hostage during mail delivery.
    delivered = await asyncio.to_thread(send_email, user.email, subject, body)
    return code, delivered


def should_echo_otp(delivered: bool) -> bool:
    """Echo the OTP in API responses only when mail did not go out."""
    return settings.ENVIRONMENT != "production" and not delivered


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
    """True when no unconsumed OTP was created within the cooldown window.

    Measured on the DATABASE clock (created_at uses func.now()) — comparing
    against Python-side UTC breaks on any Postgres whose timezone differs
    from UTC and would block resends for hours.
    """
    result = await db.execute(
        select(EmailOTP.id).where(
            EmailOTP.user_id == user.user_id,
            EmailOTP.purpose == purpose,
            EmailOTP.consumed_at.is_(None),
            EmailOTP.created_at
            >= func.now() - timedelta(seconds=settings.OTP_RESEND_COOLDOWN_SECONDS),
        ).limit(1)
    )
    return result.scalar_one_or_none() is None
