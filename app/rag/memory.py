"""
Short-Term Memory — muat riwayat percakapan dari DB untuk di-inject ke prompt LLM.

Mirip konsep ConversationBufferMemory di LangChain:
simpan N pasang pesan (user + assistant) terakhir agar model
dapat memahami konteks percakapan yang sedang berlangsung.
"""

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
    """
    Muat riwayat percakapan dari database.

    Args:
        session: AsyncSession SQLAlchemy.
        conversation_id: ID percakapan yang sedang berlangsung.
        max_turns: Jumlah *pasangan* (user+assistant) terakhir yang dimuat.
                   Jika None, gunakan nilai dari settings.

    Returns:
        List of (role, content) — diurutkan dari yang paling lama ke terbaru.
    """
    settings = get_settings()
    if max_turns is None:
        max_turns = settings.memory_max_turns

    # Ambil 2 * max_turns pesan terakhir (user + assistant)
    # Kita over-fetch sedikit lalu trim di Python untuk memastikan
    # pasangan lengkap.
    limit = max_turns * 2 + 2  # sedikit buffer

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    # Balik urutan: dari lama → baru
    messages.reverse()

    # Ambil hanya max_turns pasangan terakhir
    pairs: List[Tuple[str, str]] = []
    for msg in messages:
        pairs.append((msg.role, msg.content))

    # Potong agar maksimal max_turns * 2 entry (user+assistant)
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
    """
    Format riwayat percakapan menjadi string untuk di-inject ke prompt.

    Args:
        history: List of (role, content) dari load_memory_from_db().

    Returns:
        String terformat, atau string kosong jika tidak ada riwayat.
    """
    if not history:
        return ""

    lines: list[str] = []
    for role, content in history:
        if role == "user":
            lines.append(f"Pengguna: {content}")
        elif role == "assistant":
            # Potong jawaban panjang agar tidak memakan token budget
            truncated = content[:500]
            if len(content) > 500:
                truncated += "..."
            lines.append(f"Asisten: {truncated}")

    return "\n".join(lines)
