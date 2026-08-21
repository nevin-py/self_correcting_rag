from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field
import uuid


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserResponse(UserBase):
    user_id: uuid.UUID
    is_active: bool
    email_verified: bool = False
    create_time: datetime
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20)


class TokenData(BaseModel):
    user_id: uuid.UUID | None = None


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=6)


class ResendOtpRequest(BaseModel):
    email: EmailStr
    purpose: str = Field(default="verify_email", pattern="^(verify_email|reset_password)$")


class MessageResponse(BaseModel):
    detail: str
    # Local-dev convenience: when SMTP is not configured (non-production),
    # the OTP code is echoed here so registration is testable without mail.
    debug_otp: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=6)
    new_password: str = Field(min_length=8)


class RegisterPendingResponse(BaseModel):
    detail: str
    email: EmailStr
    email_verified: bool = False
    # Same local-dev echo as MessageResponse.debug_otp.
    debug_otp: str | None = None
