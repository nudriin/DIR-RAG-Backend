from fastapi import FastAPI

from app.api import chat, evaluate, ingest
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
async def startup_event() -> None:
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


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "rag_mode": settings.rag_mode}

