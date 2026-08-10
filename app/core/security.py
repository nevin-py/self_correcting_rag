from datetime import UTC
import bcrypt
from datetime import datetime, timedelta
import jwt
from app.core.config import settings
def hash_password(password:str)->str:
    hashed_bytes=bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')

def verify_password(plain_pass:str,hashed_pass:str)->bool:
    return bcrypt.checkpw(plain_pass.encode('utf-8'),hashed_pass.encode('utf-8'))
def create_token_access(data:dict,expired_delta:timedelta|None=None):
    payload=data.copy()
    expire=datetime.now(UTC)+(expired_delta or timedelta(minutes=15))
    payload.update({'exp':expire})
    encoded=jwt.encode(payload,settings.SECRET_KEY,algorithm=settings.ALGORITHM)
    return encoded
