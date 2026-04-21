from datetime import datetime, timezone

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import generate_secure_password, hash_password
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import AdminUser, SystemSetting


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


async def seed_default_settings(session: AsyncSession) -> None:
    """Seed default system settings if not already present."""
    defaults = {
        "refinement_backend": "gemini",
        "refinement_model_gemini": "gemini-2.5-pro",
        "refinement_model_replicate": "meta/meta-llama-3-70b-instruct",
        "gemini_mode": "api_key",
        "vertex_project": "",
        "vertex_location": "us-central1",
        # Generator backend (DRAGIN) defaults
        "generator_backend": "gemini",
        "generator_model_gemini": "gemini-2.0-flash",
        "generator_model_openai": "gpt-4o",
    }
    for key, value in defaults.items():
        existing = await session.get(SystemSetting, key)
        if existing is None:
            session.add(SystemSetting(key=key, value=value))
            logger.info(f"Seeded default setting: {key}={value}")
    await session.commit()
