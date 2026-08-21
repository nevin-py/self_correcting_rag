import logging
from datetime import UTC, datetime

import jwt
import uuid as _uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.core.security import hash_password, verify_password
from app.auth.models import User
from app.auth.otp import create_and_send_otp, should_echo_otp, verify_otp, resend_allowed
from app.auth.tokens import issue_token_pair, rotate_refresh_token, revoke_refresh_token
from app.auth.schemas import (
    UserCreate,
    Token,
    RefreshRequest,
    VerifyEmailRequest,
    ResendOtpRequest,
    MessageResponse,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RegisterPendingResponse,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def user_exist(email: str, db: AsyncSession):
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_current_user(
    *,
    token_str: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    try:
        decoded = jwt.decode(
            token_str,
            key=settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        id_sub = decoded["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    try:
        user_uuid = _uuid.UUID(id_sub)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(select(User).where(User.user_id == user_uuid))
    res = result.scalar_one_or_none()
    if res is None:
        raise HTTPException(status_code=401, detail="User not found")
    if not res.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")
    if not res.email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
    return res


@router.post("/register", response_model=RegisterPendingResponse, status_code=201)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create an unverified user and send email OTP. No JWT until verified."""
    logger.info("Registration attempt for email=%s", user.email)
    exists = await user_exist(user.email, db)
    if exists:
        if exists.email_verified:
            raise HTTPException(status_code=400, detail="Email already exists")
        # Allow re-registering unverified account: reset password + resend OTP
        exists.hashed_password = hash_password(user.password)
        await db.commit()
        target = exists
    else:
        target = User(
            email=user.email,
            hashed_password=hash_password(user.password),
            email_verified=False,
        )
        db.add(target)
        await db.commit()
        await db.refresh(target)

    try:
        code, delivered = await create_and_send_otp(
            db,
            target,
            "verify_email",
            subject="Verify your email",
            body_template=(
                "Your verification code is {code}.\n"
                "It expires in {minutes} minutes.\n"
            ),
        )
    except Exception:
        logger.exception("Failed to send verification email to %s", user.email)
        raise HTTPException(
            status_code=503,
            detail="Could not send verification email. Check SMTP settings or try again.",
        )

    logger.info("Registration pending verification: user_id=%s", target.user_id)
    if not delivered and settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=503,
            detail="Could not send verification email. Check SMTP settings or try again.",
        )
    return RegisterPendingResponse(
        detail="Verification code sent. Check your email.",
        email=target.email,
        email_verified=False,
        debug_otp=code if should_echo_otp(delivered) else None,
    )


@router.post("/verify-email", response_model=Token)
async def verify_email(body: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.email_verified:
        return await issue_token_pair(db, user)

    try:
        await verify_otp(db, user, "verify_email", body.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user.email_verified = True
    user.email_verified_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()

    logger.info("Email verified for user_id=%s", user.user_id)
    return await issue_token_pair(db, user)

@router.post("/resend-otp", response_model=MessageResponse)
async def resend_otp(body: ResendOtpRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    # Always return generic success to avoid email enumeration
    if not user:
        return MessageResponse(detail="If that email exists, a code was sent.")

    if body.purpose == "verify_email" and user.email_verified:
        return MessageResponse(detail="Email is already verified.")

    if not await resend_allowed(db, user, body.purpose):
        raise HTTPException(
            status_code=429,
            detail=f"Wait {settings.OTP_RESEND_COOLDOWN_SECONDS}s before resending.",
        )

    subject = (
        "Verify your email"
        if body.purpose == "verify_email"
        else "Reset your password"
    )
    try:
        code, delivered = await create_and_send_otp(
            db,
            user,
            body.purpose,
            subject=subject,
            body_template=(
                "Your code is {code}.\nIt expires in {minutes} minutes.\n"
            ),
        )
    except Exception:
        logger.exception("Resend OTP failed for %s", body.email)
        raise HTTPException(status_code=503, detail="Could not send email.")

    return MessageResponse(
        detail="If that email exists, a code was sent.",
        debug_otp=code if should_echo_otp(delivered) else None,
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Login attempt for email=%s", form_data.username)
    user = await user_exist(form_data.username, db)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Check your inbox or resend the code.",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    logger.info("Login successful for email=%s", form_data.username)
    return await issue_token_pair(db, user)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.hashed_password = hash_password(body.new_password)
    await db.commit()
    return MessageResponse(detail="Password updated.")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    if user and user.email_verified:
        if not await resend_allowed(db, user, "reset_password"):
            raise HTTPException(
                status_code=429,
                detail=f"Wait {settings.OTP_RESEND_COOLDOWN_SECONDS}s before resending.",
            )
        try:
            await create_and_send_otp(
                db,
                user,
                "reset_password",
                subject="Reset your password",
                body_template=(
                    "Your password reset code is {code}.\n"
                    "It expires in {minutes} minutes.\n"
                ),
            )
        except Exception:
            logger.exception("Forgot-password email failed for %s", body.email)
    return MessageResponse(detail="If that email exists, a reset code was sent.")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid code or email")
    try:
        await verify_otp(db, user, "reset_password", body.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user.hashed_password = hash_password(body.new_password)
    if not user.email_verified:
        user.email_verified = True
        user.email_verified_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()
    return MessageResponse(detail="Password reset. You can log in now.")


@router.post("/refresh", response_model=Token)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Rotate refresh token and issue a new access token."""
    try:
        return await rotate_refresh_token(db, body.refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/logout", response_model=MessageResponse)
async def logout(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    await revoke_refresh_token(db, body.refresh_token)
    return MessageResponse(detail="Logged out.")
