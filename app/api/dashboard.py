from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.auth import check_role
from app.core.config import get_settings
from app.core.gemini_client import validate_service_account_json, reset_configuration
from app.db.crud import (
    get_all_conversations,
    get_conversation_history,
    get_dashboard_stats,
    add_feedback,
    get_answer_contexts_for_messages,
    delete_conversation,
    delete_all_conversations,
    get_all_system_settings,
    get_system_setting,
    set_system_setting,
)
from app.db.engine import get_session
from app.db.models import Message


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["dashboard"],
    dependencies=[Depends(check_role("admin"))],
)


def to_wib(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=7)))


def to_wib_iso(dt: datetime | None) -> str | None:
    converted = to_wib(dt)
    return converted.isoformat() if converted is not None else None


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
            created_at=to_wib(c.created_at),
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
        created_at=to_wib(conversation.created_at),
        messages=[
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=to_wib(m.created_at),
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
                "created_at": to_wib_iso(conversation.created_at),
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "created_at": to_wib_iso(m.created_at),
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
                    "created_at": to_wib_iso(conversation.created_at),
                    "messages": [
                        {
                            "id": m.id,
                            "role": m.role,
                            "content": m.content,
                            "created_at": to_wib_iso(m.created_at),
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



VALID_SETTINGS: dict[str, list[str] | None] = {
    "refinement_backend": ["replicate", "gemini"],
    "refinement_model_gemini": None,
    "refinement_model_replicate": None,
    "gemini_mode": ["api_key", "vertex_ai"],
    "vertex_project": None,
    "vertex_location": None,
    "generator_backend": ["gemini", "openai"],
    "generator_model_gemini": None,
    "generator_model_openai": None,
}


class SystemSettingUpdate(BaseModel):
    key: str
    value: str


@router.get("/settings")
async def get_settings_endpoint(
    session: AsyncSession = Depends(get_session),
):
    settings = await get_all_system_settings(session)
    return {"settings": settings}


@router.put("/settings")
async def update_setting(
    payload: SystemSettingUpdate,
    session: AsyncSession = Depends(get_session),
):
    if payload.key not in VALID_SETTINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown setting key '{payload.key}'. "
                   f"Allowed keys: {list(VALID_SETTINGS.keys())}",
        )
    allowed_values = VALID_SETTINGS[payload.key]
    if allowed_values is not None and payload.value not in allowed_values:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value '{payload.value}' for '{payload.key}'. "
                   f"Allowed: {allowed_values}",
        )
    await set_system_setting(session, payload.key, payload.value)
    await session.commit()
    if payload.key == "gemini_mode":
        reset_configuration()
    return {"key": payload.key, "value": payload.value}



SA_MAX_SIZE_BYTES = 1 * 1024 * 1024


@router.post("/settings/gemini-sa", status_code=200)
async def upload_gemini_service_account(
    file: UploadFile = File(...),
):
    if file.content_type not in ("application/json", "text/plain", "application/octet-stream"):
        filename = file.filename or ""
        if not filename.endswith(".json"):
            raise HTTPException(
                status_code=400,
                detail="File harus berekstensi .json (Service Account JSON).",
            )

    content = await file.read()

    try:
        sa_data = validate_service_account_json(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg = get_settings()
    sa_dir: Path = cfg.data_dir / "service_accounts"
    sa_dir.mkdir(parents=True, exist_ok=True)
    sa_file: Path = sa_dir / "gemini_sa.json"

    try:
        sa_file.write_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menyimpan SA JSON: {exc}",
        )

    reset_configuration()

    logger.info(
        "Gemini Service Account JSON uploaded",
        extra={"project_id": sa_data.get("project_id"), "client_email": sa_data.get("client_email")},
    )

    return {
        "status": "ok",
        "message": "Service Account JSON berhasil diunggah.",
        "project_id": sa_data.get("project_id"),
        "client_email": sa_data.get("client_email"),
        "filename": sa_file.name,
    }


@router.get("/settings/gemini-sa")
async def get_gemini_sa_status():
    cfg = get_settings()
    sa_file: Path = cfg.data_dir / "service_accounts" / "gemini_sa.json"

    if not sa_file.exists():
        return {"has_sa": False, "filename": None, "project_id": None}

    try:
        import json
        data = json.loads(sa_file.read_bytes())
        return {
            "has_sa": True,
            "filename": sa_file.name,
            "project_id": data.get("project_id"),
            "client_email": data.get("client_email"),
        }
    except Exception:
        return {"has_sa": True, "filename": sa_file.name, "project_id": None}
