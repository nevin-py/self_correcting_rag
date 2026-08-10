import jwt
import uuid as _uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_token_access
from app.auth.models import User
from app.auth.schemas import UserCreate, UserResponse, Token
from typing import Optional
from app.core.config import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Helpers (plain async functions — receive db as a regular parameter) ──────

async def user_exist(email: str, db: AsyncSession):
    """Check if a user with this email already exists."""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def passwrd_check(email: str, passwrd: str, db: AsyncSession):
    """Validate credentials and return a JWT. Raises HTTPException on failure."""
    user = await user_exist(email, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(passwrd, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token_access({'sub': str(user.user_id)})
    return token


async def create_user(user: UserCreate, db: AsyncSession):
    """Insert a new user row and return it."""
    hashed = hash_password(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_current_user(
    *,
    token_str: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Decode the JWT, look up the user, and return the User row."""
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

    result = await db.execute(
        select(User).where(User.user_id == user_uuid)
    )
    res = result.scalar_one_or_none()
    if res is None:
        raise HTTPException(status_code=401, detail="User not found")
    return res


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    exists = await user_exist(user.email, db)
    if exists:
        raise HTTPException(status_code=400, detail="Email already exists")
    return await create_user(user, db)


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    token = await passwrd_check(form_data.username, form_data.password, db)
    return {"access_token": token, "token_type": "bearer"}
