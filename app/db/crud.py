from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, Feedback, AnswerContext


async def create_conversation(
    session: AsyncSession,
    title: str,
    metadata: Optional[str] = None,
) -> Conversation:
    conversation = Conversation(title=title, meta=metadata)
    session.add(conversation)
    await session.flush()
    await session.refresh(conversation)
    return conversation


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    tokens: Optional[int] = None,
    confidence: Optional[float] = None,
    rag_iterations: Optional[int] = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tokens=tokens,
        confidence=confidence,
        rag_iterations=rag_iterations,
    )
    session.add(message)
    await session.flush()
    await session.refresh(message)
    return message


async def get_conversation_history(
    session: AsyncSession,
    conversation_id: int,
) -> Tuple[Conversation, List[Message]]:
    stmt: Select[Conversation] = (
        select(Conversation).where(Conversation.id == conversation_id).limit(1)
    )
    result = await session.execute(stmt)
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise ValueError("Conversation not found")

    messages_stmt: Select[Message] = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages_result = await session.execute(messages_stmt)
    messages = list(messages_result.scalars().all())
    return conversation, messages


async def get_all_conversations(
    session: AsyncSession,
    offset: int = 0,
    limit: int = 20,
) -> List[Conversation]:
    stmt: Select[Conversation] = (
        select(Conversation)
        .order_by(Conversation.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_answer_contexts_for_messages(
    session: AsyncSession,
    message_ids: List[int],
) -> Dict[int, List[AnswerContext]]:
    if not message_ids:
        return {}
    stmt: Select[AnswerContext] = (
        select(AnswerContext)
        .where(AnswerContext.message_id.in_(message_ids))
        .order_by(AnswerContext.id)
    )
    result = await session.execute(stmt)
    contexts = list(result.scalars().all())
    grouped: Dict[int, List[AnswerContext]] = {}
    for ctx in contexts:
        grouped.setdefault(ctx.message_id, []).append(ctx)
    return grouped


async def get_dashboard_stats(session: AsyncSession) -> Dict[str, Any]:
    total_conversations_stmt = select(func.count(Conversation.id))
    total_messages_stmt = select(func.count(Message.id))
    avg_confidence_stmt = select(func.avg(Message.confidence))
    total_feedback_stmt = select(func.count(Feedback.id))
    avg_feedback_score_stmt = select(func.avg(Feedback.score))

    total_conversations = (await session.execute(total_conversations_stmt)).scalar() or 0
    total_messages = (await session.execute(total_messages_stmt)).scalar() or 0
    avg_confidence = (await session.execute(avg_confidence_stmt)).scalar()
    total_feedback = (await session.execute(total_feedback_stmt)).scalar() or 0
    avg_feedback_score = (await session.execute(avg_feedback_score_stmt)).scalar()

    last_activity_stmt = select(func.max(Message.created_at))
    last_activity = (await session.execute(last_activity_stmt)).scalar()

    return {
        "total_conversations": int(total_conversations),
        "total_messages": int(total_messages),
        "avg_confidence": float(avg_confidence) if avg_confidence is not None else None,
        "last_activity": last_activity.isoformat() if isinstance(last_activity, datetime) else None,
        "total_feedback": int(total_feedback),
        "avg_feedback_score": float(avg_feedback_score) if avg_feedback_score is not None else None,
    }


async def add_feedback(
    session: AsyncSession,
    message_id: int,
    score: int,
    comment: Optional[str] = None,
) -> Feedback:
    feedback = Feedback(
        message_id=message_id,
        score=score,
        comment=comment,
    )
    session.add(feedback)
    await session.flush()
    await session.refresh(feedback)
    return feedback
