from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, dashboard, evaluate, ingest
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.engine import init_db


configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()
    logger.info(
        "Application startup",
        extra={
            "environment": settings.environment,
            "rag_mode": settings.rag_mode,
        },
    )


app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(evaluate.router)
app.include_router(dashboard.router)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "rag_mode": settings.rag_mode}

