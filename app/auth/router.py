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
router=APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.get("/")
async def home(db: AsyncSession = Depends(get_db)):
    ...

async def user_exist(email:str,db:AsyncSession= Depends(get_db)):
    result=await db.execute(
        select(User).where(User.email==email)
    )
    return result.scalar_one_or_none()


async def passwrd_check(email:str,passwrd:str,db:AsyncSession= Depends(get_db)):
    user=await user_exist(email,db)
    if not user:
        raise HTTPException(401)
    if not verify_password(passwrd,user.hashed_password):
        raise HTTPException(401)
    token=create_token_access({'sub':str(user.user_id)})
    return token


async def create_user(user:UserCreate,db:AsyncSession= Depends(get_db)):
    hashed=hash_password(user.password)
    db_user=User(
        email=user.email,
        hashed_password=hashed
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def get_current_user(*,token_str:str=Depends(oauth2_scheme),db:AsyncSession= Depends(get_db) ):
    try:
        decoded=jwt.decode(token_str,key=settings.SECRET_KEY,algorithms=[settings.ALGORITHM])
        id_sub=decoded["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401,detail="Could not validate credentials")
    # JWT 'sub' field stores user_id as a string — parse it back to UUID
    try:
        user_uuid = _uuid.UUID(id_sub)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=401, detail="Invalid token payload")
    result=await db.execute(
        select(User).where(User.user_id == user_uuid)
    )
    res=result.scalar_one_or_none()
    if res is None:
        raise HTTPException(status_code=401, detail="User not found")
    return res


@router.post("/register", response_model=UserResponse)
async def register(user:UserCreate,db:AsyncSession= Depends(get_db)):
    
    exists=await user_exist(user.email,db)
    if exists:
        raise HTTPException(
        status_code=400,
        detail="Email already exists"
        )
    return await create_user(user,db)


@router.post('/login',response_model=Token)
async def login(form_data:OAuth2PasswordRequestForm=Depends(),db:AsyncSession= Depends(get_db)):
    token=await passwrd_check(form_data.username,form_data.password,db)
    return {"access_token":token,"token_type":"bearer"}


