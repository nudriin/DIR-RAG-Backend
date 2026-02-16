from datetime import datetime, timezone

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import generate_secure_password, hash_password
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import AdminUser


logger = get_logger(__name__)


async def seed_default_admin(session: AsyncSession) -> None:
    stmt: Select[AdminUser] = select(AdminUser).where(
        AdminUser.username == "admin"
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        return

    settings = get_settings()

    env_password = settings.default_admin_password
    if env_password and len(env_password) >= 8:
        password = env_password
    else:
        password = generate_secure_password()
        logger.info(
            "Default admin password generated",
            extra={
                "username": "admin",
                "email": "admin@test.com",
                "password": password,
            },
        )

    password_hash = hash_password(password)
    user = AdminUser(
        username="admin",
        email="admin@test.com",
        password_hash=password_hash,
        role="super_admin",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(user)
    await session.commit()
