import logging
from datetime import UTC, datetime

import jwt
import uuid as _uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.core.security import hash_password, verify_password
from app.auth.models import User
from app.auth.otp import create_and_send_otp, should_echo_otp, verify_otp, resend_allowed
from app.auth.tokens import (
    issue_token_pair,
    rotate_refresh_token,
    revoke_refresh_token,
    revoke_all_for_user,
    RefreshTokenReuseError,
)
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
from app.core.limiter import limiter

logger = logging.getLogger(__name__)

# ── Refresh-token cookie ─────────────────────────────────────────────────────
# The refresh token is ALSO set as an httpOnly cookie so browser clients never
# need to keep it in localStorage (XSS-exposed). The JSON body keeps working
# for non-cookie API clients. Path is scoped to auth endpoints so the cookie
# rides only on /refresh and /logout.
REFRESH_COOKIE = "refresh_token"


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=raw_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,
        secure=settings.ENVIRONMENT == "production",  # browsers allow Secure on localhost
        # Cross-site frontend↔API (Vercel ↔ Cloud Run) requires SameSite=None;
        # same-site dev (localhost:3000 → localhost:8000) uses Lax.
        samesite="none" if settings.ENVIRONMENT == "production" else "lax",
        path="/api/v1/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE, path="/api/v1/auth")


def _refresh_token_from(request: Request, body: "RefreshRequest | None") -> str | None:
    if body and body.refresh_token:
        return body.refresh_token
    return request.cookies.get(REFRESH_COOKIE)

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
@limiter.limit("5/minute")
async def register(request: Request, user: UserCreate, db: AsyncSession = Depends(get_db)):
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
@limiter.limit("10/minute")
async def verify_email(request: Request, body: VerifyEmailRequest, response: Response, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.email_verified:
        # SECURITY: never issue tokens without a valid OTP. Returning a token
        # pair here would let anyone who knows a verified email take over the
        # account (token pair for a code that is never checked).
        raise HTTPException(
            status_code=409,
            detail="Email is already verified. Please log in.",
        )

    try:
        await verify_otp(db, user, "verify_email", body.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user.email_verified = True
    user.email_verified_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()

    logger.info("Email verified for user_id=%s", user.user_id)
    tokens = await issue_token_pair(db, user)
    _set_refresh_cookie(response, tokens["refresh_token"])
    return tokens

@router.post("/resend-otp", response_model=MessageResponse)
@limiter.limit("3/minute")
async def resend_otp(request: Request, body: ResendOtpRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    # Always return the same generic response — status, wording, and timing
    # must not reveal whether the email exists or is already verified.
    generic = MessageResponse(detail="If that email exists, a code was sent.")
    if not user:
        return generic

    if body.purpose == "verify_email" and user.email_verified:
        return generic

    if not await resend_allowed(db, user, body.purpose):
        # Silent skip — a 429 here would confirm the email exists.
        return generic

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


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    """Identity probe for the frontend session bootstrap.

    Validates the access token WITHOUT rotating the refresh token (the old
    frontend used /refresh for this — one token-family rotation per page load,
    and reuse failures across tabs). Returns the email so the UI can display
    the account even after a token refresh.
    """
    return {
        "user_id": str(current_user.user_id),
        "email": current_user.email,
        "email_verified": current_user.email_verified,
        "is_active": current_user.is_active,
    }


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
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
    tokens = await issue_token_pair(db, user)
    _set_refresh_cookie(response, tokens["refresh_token"])
    return tokens


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
    # Kill every refresh-token session: a stolen session must not survive a
    # password change. The user simply logs in again on other devices.
    await revoke_all_for_user(db, current_user.user_id)
    return MessageResponse(detail="Password updated.")


@router.post("/forgot-password", response_model=MessageResponse)
@limiter.limit("3/minute")
async def forgot_password(request: Request, body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    user = await user_exist(body.email, db)
    if user and user.email_verified:
        # NOTE: no 429 leak — a cooldown hit must look identical to a send,
        # otherwise the endpoint confirms which emails exist. The skip is
        # silent; a legitimate user just requests again after the cooldown.
        if await resend_allowed(db, user, "reset_password"):
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
@limiter.limit("5/minute")
async def reset_password(request: Request, body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
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
    # Kill every existing session — the password just changed, so any
    # refresh token issued before it (including a stolen one) must die.
    await revoke_all_for_user(db, user.user_id)
    return MessageResponse(detail="Password reset. You can log in now.")


@router.post("/refresh", response_model=Token)
@limiter.limit("30/minute")
async def refresh_token(request: Request, body: RefreshRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Rotate refresh token and issue a new access token.

    The refresh token comes from the httpOnly cookie, or the JSON body for
    non-cookie clients. Re-presenting a revoked token raises
    RefreshTokenReuseError: treated as theft — ALL of the user's refresh
    tokens are revoked, then 401.
    """
    raw = _refresh_token_from(request, body)
    if not raw:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    try:
        tokens = await rotate_refresh_token(db, raw)
    except RefreshTokenReuseError:
        logger.warning("Refresh token reuse detected — all user sessions revoked")
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    _set_refresh_cookie(response, tokens["refresh_token"])
    return tokens


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, body: RefreshRequest, response: Response, db: AsyncSession = Depends(get_db)):
    raw = _refresh_token_from(request, body)
    if raw:
        await revoke_refresh_token(db, raw)
    _clear_refresh_cookie(response)
    return MessageResponse(detail="Logged out.")
