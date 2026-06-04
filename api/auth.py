"""
api/auth.py
────────────
JWT Bearer Token authentication — FastAPI dependency injection.

Security design:
  • HS256 signed JWTs with configurable expiry (default 60 min).
  • Token introspection via /api/v1/auth/token endpoint.
  • FastAPI OAuth2PasswordBearer integrates with Swagger UI for dev testing.
  • Passwords hashed with bcrypt (cost factor 12) — never stored plaintext.

Dependency chain:
  get_current_user()
    └─ verify_token()         — validates JWT signature + expiry
       └─ decode_access_token() — jose.jwt.decode()
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from data_pipeline.database import get_db
from data_pipeline.schemas import User

# ── Password Hashing ──────────────────────────────────────────────────────────

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/token",
    scheme_name="JWT",
)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Token Models ──────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str           # user UUID string
    exp: int           # Unix timestamp
    jti: str           # JWT ID (for future revocation list support)
    iss: str = settings.SERVICE_NAME


# ── JWT Operations ────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> Token:
    expires_delta = expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    jti = str(uuid.uuid4())

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": jti,
        "iss": settings.SERVICE_NAME,
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return Token(
        access_token=token,
        expires_in=int(expires_delta.total_seconds()),
    )


def decode_access_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenPayload(
            sub=payload["sub"],
            exp=payload["exp"],
            jti=payload["jti"],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    FastAPI dependency — resolves JWT → User ORM object.
    Inject via: current_user: Annotated[User, Depends(get_current_user)]
    """
    payload = decode_access_token(token)

    result = await db.execute(
        select(User).where(
            User.id == uuid.UUID(payload.sub),
            User.is_active == True,
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# Lighter dependency — skips DB lookup, only validates token signature.
# Use for high-frequency endpoints where DB round-trip is unaffordable.
async def get_current_user_id(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> str:
    """Returns user_id string without a database round-trip."""
    payload = decode_access_token(token)
    return payload.sub
