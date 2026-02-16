from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    check_role,
    create_session_tokens,
    hash_password,
    refresh_session_token,
    validate_password_policy,
    verify_password,
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.engine import get_session
from app.db.models import AdminUser


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class RefreshRequest(BaseModel):
    refresh_token: str


class AdminRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class AdminResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    created_at: datetime
    last_login: datetime | None = None


_register_attempts: dict[tuple[str, int], int] = {}


def _check_rate_limit(request: Request) -> None:
    settings = get_settings()
    ip = request.client.host if request.client and request.client.host else "unknown"
    now = datetime.now(timezone.utc)
    hour_bucket = int(now.timestamp()) // 3600
    key = (ip, hour_bucket)
    current = _register_attempts.get(key, 0)
    limit = settings.admin_register_rate_limit_per_hour
    if current >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Terlalu banyak percobaan registrasi admin dari IP ini",
        )
    _register_attempts[key] = current + 1


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    if len(payload.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username minimal 3 karakter",
        )
    validate_password_policy(payload.password)

    stmt: Select[AdminUser] = select(AdminUser).where(
        func.lower(AdminUser.username) == payload.username.lower()
    )
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah",
        )

    data = await create_session_tokens(session=session, user=user)
    await session.commit()

    logger.info(
        "Admin login",
        extra={
            "username": user.username,
            "email": user.email,
            "role": user.role,
        },
    )

    return TokenResponse(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        user=data["user"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    data = await refresh_session_token(refresh_token=payload.refresh_token, session=session)
    await session.commit()
    return TokenResponse(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        user=data["user"],
    )


@router.post(
    "/admin/register",
    response_model=AdminResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_admin(
    payload: AdminRegisterRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    _: AdminUser = Depends(check_role("admin")),
) -> AdminResponse:
    _check_rate_limit(request)
    validate_password_policy(payload.password)

    stmt = select(AdminUser).where(
        func.lower(AdminUser.username) == payload.username.lower()
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username sudah digunakan",
        )

    stmt = select(AdminUser).where(
        func.lower(AdminUser.email) == payload.email.lower()
    )
    result = await session.execute(stmt)
    existing_email = result.scalar_one_or_none()
    if existing_email is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email sudah digunakan",
        )

    password_hash = hash_password(payload.password)
    user = AdminUser(
        username=payload.username,
        email=payload.email,
        password_hash=password_hash,
        role="admin",
        is_active=True,
    )
    session.add(user)
    await session.flush()
    await session.refresh(user)
    await session.commit()

    logger.info(
        "Admin baru diregistrasi",
        extra={
            "username": user.username,
            "email": user.email,
            "role": user.role,
        },
    )

    return AdminResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
        last_login=user.last_login,
    )

