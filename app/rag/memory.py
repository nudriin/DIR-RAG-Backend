from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Message

logger = get_logger(__name__)


async def load_memory_from_db(
    session: AsyncSession,
    conversation_id: int,
    max_turns: int | None = None,
) -> List[Tuple[str, str]]:
    settings = get_settings()
    if max_turns is None:
        max_turns = settings.memory_max_turns

    limit = max_turns * 2 + 2

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    messages.reverse()

    pairs: List[Tuple[str, str]] = []
    for msg in messages:
        pairs.append((msg.role, msg.content))

    if len(pairs) > max_turns * 2:
        pairs = pairs[-(max_turns * 2):]

    logger.info(
        "Loaded conversation memory",
        extra={
            "conversation_id": conversation_id,
            "messages_loaded": len(pairs),
            "max_turns": max_turns,
        },
    )
    return pairs


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return ""

    lines: list[str] = []
    for role, content in history:
        if role == "user":
            lines.append(f"Pengguna: {content}")
        elif role == "assistant":
            truncated = content[:500]
            if len(content) > 500:
                truncated += "..."
            lines.append(f"Asisten: {truncated}")

    return "\n".join(lines)
