from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.crud import (
    get_all_conversations,
    get_conversation_history,
    get_dashboard_stats,
)
from app.db.engine import get_session


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


@router.get("/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats(
    session: AsyncSession = Depends(get_session),
):
    stats = await get_dashboard_stats(session=session)
    return DashboardStats(**stats)
