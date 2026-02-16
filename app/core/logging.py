import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import asyncio
from typing import Set, Any, Dict
import json
from datetime import datetime, timezone

from .config import get_settings

# Simple broadcaster for log streaming (SSE/WebSocket subscribers)
_log_subscribers: Set[asyncio.Queue] = set()

def subscribe_to_logs() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _log_subscribers.add(q)
    return q

def unsubscribe_from_logs(q: asyncio.Queue) -> None:
    _log_subscribers.discard(q)

class StreamBroadcastHandler(logging.Handler):
    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__(level=logging.INFO)
        self._formatter = formatter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self._formatter.format(record)
            for q in list(_log_subscribers):
                # best-effort: don't block logging pipeline
                if not q.full():
                    q.put_nowait(msg)
        except Exception:
            # never raise in logging handler
            pass

def broadcast_event(stage: str, action: str, summary: str, details: Dict[str, Any] | None = None) -> None:
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "action": action,
        "summary": summary,
        "details": details or {},
    }
    msg = json.dumps(payload, ensure_ascii=False)
    for q in list(_log_subscribers):
        if not q.full():
            q.put_nowait(msg)

def configure_logging() -> None:
    settings = get_settings()
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "humbet_rag.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    broadcast_handler = StreamBroadcastHandler(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(broadcast_handler)

    # Ensure httpx client logs are visible (for Replicate HTTP calls)
    logging.getLogger("httpx").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
