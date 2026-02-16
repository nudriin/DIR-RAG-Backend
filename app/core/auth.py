from datetime import datetime, timedelta, timezone
import os
import re
import secrets
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.engine import get_session
from app.db.models import AdminUser, AuthToken


logger = get_logger(__name__)
security_scheme = HTTPBearer(auto_error=False)


PASSWORD_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$"
)


def validate_password_policy(password: str) -> None:
    if not PASSWORD_REGEX.match(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password harus minimal 8 karakter dan mengandung huruf besar, huruf kecil, angka, dan simbol",
        )


def hash_password(password: str) -> str:
    import bcrypt

    settings = get_settings()
    rounds = max(10, int(os.getenv("BCRYPT_ROUNDS", "12")))
    salt = bcrypt.gensalt(rounds)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_jwt_token(data: Dict[str, Any], expires_delta: timedelta) -> str:
    import jwt

    settings = get_settings()
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    to_encode.update({"exp": expire, "iat": now})
    token = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token


def decode_jwt_token(token: str, verify_exp: bool = True) -> Dict[str, Any]:
    import jwt

    settings = get_settings()
    options = {"verify_exp": verify_exp}
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options=options,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token kedaluwarsa",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token tidak valid",
        )


async def create_session_tokens(
    session: AsyncSession,
    user: AdminUser,
) -> Dict[str, Any]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    access_expires = now + timedelta(hours=settings.access_token_exp_hours)
    refresh_expires = now + timedelta(days=settings.refresh_token_exp_days)

    jti_access = secrets.token_hex(16)
    jti_refresh = secrets.token_hex(16)

    access_payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "type": "access",
        "jti": jti_access,
    }
    refresh_payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "type": "refresh",
        "jti": jti_refresh,
    }

    access_token = create_jwt_token(
        access_payload, timedelta(hours=settings.access_token_exp_hours)
    )
    refresh_token = create_jwt_token(
        refresh_payload, timedelta(days=settings.refresh_token_exp_days)
    )

    stmt = select(AuthToken).where(
        AuthToken.user_id == user.id,
        AuthToken.revoked_at.is_(None),
    )
    result = await session.execute(stmt)
    existing_tokens = list(result.scalars().all())
    now_ts = datetime.now(timezone.utc)
    for t in existing_tokens:
        t.revoked_at = now_ts

    token_row = AuthToken(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires,
        refresh_expires_at=refresh_expires,
    )
    session.add(token_row)
    user.last_login = now
    await session.flush()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "last_login": user.last_login,
        },
    }


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    session: AsyncSession = Depends(get_session),
) -> AdminUser:
    if credentials is None or not credentials.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header tidak ditemukan",
        )

    token = credentials.credentials
    payload = decode_jwt_token(token, verify_exp=True)
    user_id = payload.get("sub")
    token_type = payload.get("type")
    if token_type != "access" or user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token tidak valid",
        )

    stmt = select(AuthToken, AdminUser).join(AdminUser).where(
        AuthToken.access_token == token,
        AuthToken.revoked_at.is_(None),
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesi tidak ditemukan atau sudah dicabut",
        )
    token_row: AuthToken = row[0]
    user: AdminUser = row[1]

    now = datetime.now(timezone.utc)
    if _ensure_aware(token_row.access_expires_at) <= now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token kedaluwarsa",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akun dinonaktifkan",
        )

    return user


async def optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    session: AsyncSession = Depends(get_session),
) -> Optional[AdminUser]:
    if credentials is None:
        return None
    try:
        return await verify_token(credentials, session)  # type: ignore[arg-type]
    except HTTPException:
        return None


def check_role(required_role: str = "admin"):
    async def role_dependency(user: AdminUser = Depends(verify_token)) -> AdminUser:
        if user.role not in {required_role, "super_admin"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Anda tidak memiliki hak akses",
            )
        return user

    return role_dependency


async def refresh_session_token(
    refresh_token: str,
    session: AsyncSession,
) -> Dict[str, Any]:
    payload = decode_jwt_token(refresh_token, verify_exp=True)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token bukan refresh token",
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload token tidak valid",
        )

    stmt = select(AuthToken, AdminUser).join(AdminUser).where(
        AuthToken.refresh_token == refresh_token,
        AuthToken.revoked_at.is_(None),
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token tidak dikenal",
        )
    token_row: AuthToken = row[0]
    user: AdminUser = row[1]

    now = datetime.now(timezone.utc)
    if _ensure_aware(token_row.refresh_expires_at) <= now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token kedaluwarsa",
        )

    token_row.revoked_at = datetime.now(timezone.utc)
    data = await create_session_tokens(session=session, user=user)
    return data


def generate_secure_password(length: int = 16) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if PASSWORD_REGEX.match(pwd):
            return pwd
