from pathlib import Path
import os

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DB_URL = DATABASE_URL
    DB_PATH = None
else:
    DB_PATH = Path("storage") / "chat_history.db"
    DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def _ensure_storage_dir() -> None:
    if DB_PATH is not None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    global engine, AsyncSessionLocal

    if engine is None:
        _ensure_storage_dir()
        engine = create_async_engine(DB_URL, echo=False, future=True)
        AsyncSessionLocal = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

    return engine


async def get_session() -> AsyncSession:
    if AsyncSessionLocal is None:
        get_engine()

    assert AsyncSessionLocal is not None
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    from app.db import models  # noqa: F401
    from app.db.seed import seed_default_admin

    eng = get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    assert AsyncSessionLocal is not None
    async with AsyncSessionLocal() as session:
        await seed_default_admin(session)
