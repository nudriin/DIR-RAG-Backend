from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.crud import (
    get_all_conversations,
    get_conversation_history,
    get_dashboard_stats,
    add_feedback,
    get_answer_contexts_for_messages,
    delete_conversation,
    delete_all_conversations,
)
from app.db.engine import get_session
from app.db.models import Message


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    confidence: Optional[float] = None
    rag_iterations: Optional[int] = None


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: List[MessageItem]


class DashboardStats(BaseModel):
    total_conversations: int
    total_messages: int
    avg_confidence: Optional[float] = None
    last_activity: Optional[str] = None
    total_feedback: int
    avg_feedback_score: Optional[float] = None


class FeedbackRequest(BaseModel):
    message_id: int
    score: int
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    score: int
    comment: Optional[str] = None


@router.get("/history", response_model=List[ConversationSummary])
async def list_conversations(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    conversations = await get_all_conversations(
        session=session,
        offset=offset,
        limit=limit,
    )
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
        )
        for c in conversations
    ]


@router.get("/history/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    try:
        conversation, messages = await get_conversation_history(
            session=session,
            conversation_id=conversation_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                confidence=m.confidence,
                rag_iterations=m.rag_iterations,
            )
            for m in messages
        ],
    )


@router.delete(
    "/history/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation_history(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    deleted = await delete_conversation(session=session, conversation_id=conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return


@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_all_history(
    session: AsyncSession = Depends(get_session),
):
    await delete_all_conversations(session=session)
    await session.commit()
    return


@router.get("/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats(
    session: AsyncSession = Depends(get_session),
):
    stats = await get_dashboard_stats(session=session)
    return DashboardStats(**stats)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    message = await session.get(Message, payload.message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Feedback hanya untuk jawaban assistant")

    feedback = await add_feedback(
        session=session,
        message_id=payload.message_id,
        score=payload.score,
        comment=payload.comment,
    )
    await session.commit()

    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        score=feedback.score,
        comment=feedback.comment,
    )


@router.get("/export")
async def export_conversations(
    conversation_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    conversations = []
    if conversation_id is not None:
        try:
            conversation, messages = await get_conversation_history(
                session=session,
                conversation_id=conversation_id,
            )
        except ValueError:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message_ids = [m.id for m in messages]
        context_map = await get_answer_contexts_for_messages(
            session=session,
            message_ids=message_ids,
        )
        conversations.append(
            {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                        "confidence": m.confidence,
                        "rag_iterations": m.rag_iterations,
                        "contexts": [
                            {
                                "source": ctx.source,
                                "chunk_id": ctx.chunk_id,
                                "content": ctx.content,
                            }
                            for ctx in context_map.get(m.id, [])
                        ],
                    }
                    for m in messages
                ],
            }
        )
    else:
        conversations_list = await get_all_conversations(
            session=session,
            offset=0,
            limit=10000,
        )
        for conv in conversations_list:
            conversation, messages = await get_conversation_history(
                session=session,
                conversation_id=conv.id,
            )
            message_ids = [m.id for m in messages]
            context_map = await get_answer_contexts_for_messages(
                session=session,
                message_ids=message_ids,
            )
            conversations.append(
                {
                    "id": conversation.id,
                    "title": conversation.title,
                    "created_at": conversation.created_at.isoformat(),
                    "messages": [
                        {
                            "id": m.id,
                            "role": m.role,
                            "content": m.content,
                            "created_at": m.created_at.isoformat(),
                            "confidence": m.confidence,
                            "rag_iterations": m.rag_iterations,
                            "contexts": [
                                {
                                    "source": ctx.source,
                                    "chunk_id": ctx.chunk_id,
                                    "content": ctx.content,
                                }
                                for ctx in context_map.get(m.id, [])
                            ],
                        }
                        for m in messages
                    ],
                }
            )

    return {"conversations": conversations}
