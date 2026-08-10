#app/auth/schemas.py
from pydantic import ConfigDict
from datetime import datetime
from pydantic import BaseModel,EmailStr,Field
import uuid
class UserBase(BaseModel):
    email:EmailStr

class UserCreate(UserBase):
    password:str = Field(min_length=8)

class UserResponse(UserBase):
    user_id:uuid.UUID
    is_active:bool
    create_time:datetime
    model_config=ConfigDict(from_attributes=True)
class Token(BaseModel):
    access_token:str
    token_type:str="bearer"
class TokenData(BaseModel):
    user_id:uuid.UUID | None=None
