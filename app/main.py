from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from app.api import auth, chat, dashboard, evaluate, ingest
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.engine import init_db


configure_logging()
logger = get_logger(__name__)

settings = get_settings()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.environment != "local":
    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)


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
app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(evaluate.router)
app.include_router(dashboard.router)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "rag_mode": settings.rag_mode}

