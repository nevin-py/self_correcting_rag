from datetime import UTC
import bcrypt
from datetime import datetime, timedelta
import jwt
from app.core.config import settings


def hash_password(password: str) -> str:
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')


def verify_password(plain_pass: str, hashed_pass: str) -> bool:
    return bcrypt.checkpw(plain_pass.encode('utf-8'), hashed_pass.encode('utf-8'))


def create_token_access(data: dict, expired_delta: timedelta | None = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Payload data to encode in the token
        expired_delta: Optional custom expiry. If not provided, uses config setting.
    """
    payload = data.copy()
    # Use provided delta or fall back to config setting
    expire = datetime.now(UTC) + (expired_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    payload.update({'exp': expire})
    encoded = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded
