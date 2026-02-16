from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.rag.rag_pipeline import run_rag_pipeline
from app.db.crud import add_message, create_conversation
from app.db.engine import get_session
from app.db.models import Conversation
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponseWithTrace,
    DebugTrace,
)
from app.core.logging import subscribe_to_logs, unsubscribe_from_logs


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponseWithTrace,
    response_model_exclude_none=True,
)
async def chat_endpoint(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponseWithTrace:
    rag_result = await asyncio.to_thread(run_rag_pipeline, payload.query)

    trace_models = [
        DebugTrace(
            iteration=t.iteration,
            refined_query=t.refined_query,
            num_documents=t.num_documents,
            retrieve=t.decision.should_retry,
            retrieval_confidence=t.decision.confidence,
            reason=t.decision.reason,
            raw_query=None,
        )
        for t in rag_result.traces
    ]

    logger.info(
        "Chat request handled",
        extra={
            "query": payload.query,
            "iterations": rag_result.iterations,
            "confidence": rag_result.confidence,
        },
    )
    try:
        if payload.conversation_id is None:
            title = payload.query.strip()
            if len(title) > 100:
                title = title[:100]
            conversation = await create_conversation(
                session=session,
                title=title or "Percakapan",
            )
        else:
            conversation = await session.get(Conversation, payload.conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")

        await add_message(
            session=session,
            conversation_id=conversation.id,
            role="user",
            content=payload.query,
        )

        await add_message(
            session=session,
            conversation_id=conversation.id,
            role="assistant",
            content=rag_result.answer,
            confidence=rag_result.confidence,
            rag_iterations=rag_result.iterations,
        )

        await session.commit()
        conversation_id = conversation.id
    except Exception as exc:
        logger.error(f"Gagal menyimpan riwayat percakapan: {exc}")
        await session.rollback()
        conversation_id = payload.conversation_id or -1

    return ChatResponseWithTrace(
        answer=rag_result.answer,
        sources=rag_result.sources,
        iterations=rag_result.iterations,
        confidence=rag_result.confidence,
        conversation_id=conversation_id,
        trace=trace_models,
        debug_logs=rag_result.debug_logs,
    )


@router.get("/logs/stream")
async def logs_stream() -> StreamingResponse:
    queue = subscribe_to_logs()

    async def event_gen():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive behind proxies
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            unsubscribe_from_logs(queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
