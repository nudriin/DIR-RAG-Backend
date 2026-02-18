from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, broadcast_event
from app.core.auth import optional_current_user
from app.rag.rag_pipeline import run_rag_pipeline
from app.db.crud import add_message, create_conversation
from app.db.engine import get_session
from app.db.models import Conversation, AnswerContext
from app.data.vector_store import vector_store_manager
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponseWithTrace,
    DebugTrace,
)
from app.core.logging import subscribe_to_logs, unsubscribe_from_logs


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _is_greeting(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    simple_greetings = {
        "hi",
        "hai",
        "halo",
        "hello",
        "pagi",
        "selamat pagi",
        "selamat siang",
        "selamat sore",
        "selamat malam",
        "malam",
        "siang",
        "assalamualaikum",
        "assalamu'alaikum",
        "assalamualaikum wr wb",
        "assalamualaikum wr. wb.",
    }
    if normalized in simple_greetings:
        return True
    if len(normalized) <= 20 and any(normalized.startswith(g) for g in simple_greetings):
        if "?" not in normalized:
            return True
    return False


def _is_low_signal(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if "?" in normalized:
        return False
    if len(normalized) <= 1:
        return True
    if len(normalized) <= 4:
        return True
    letters = sum(1 for c in normalized if c.isalpha())
    if letters and letters <= 3 and len(normalized) <= 10:
        return True
    return False


def _infer_role_from_text(text: str) -> str | None:
    t = (text or "").lower()
    if any(k in t for k in ["sebagai siswa", "untuk siswa", "dashboard siswa", "menu siswa", "murid", "peserta didik", "siswa"]):
        return "siswa"
    if any(k in t for k in ["sebagai guru", "untuk guru", "dashboard guru", "menu guru", "pengajar", "guru"]):
        return "pengajar"
    if any(k in t for k in ["admin sekolah", "operator sekolah"]):
        return "admin_sekolah"
    if "pengawas" in t:
        return "pengawas"
    if "dinas" in t:
        return "dinas"
    return None


@router.post(
    "/chat",
    response_model=ChatResponseWithTrace,
    response_model_exclude_none=True,
)
async def chat_endpoint(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponseWithTrace:
    query_text = payload.query or ""
    user_role = payload.user_role
    target_role = payload.target_role
    effective_role = target_role or user_role
    if effective_role is None:
        inferred = _infer_role_from_text(query_text)
        if inferred:
            effective_role = inferred

    bypass_reason: str | None = None
    if _is_greeting(query_text):
        bypass_reason = "GREETING_BYPASS"
    elif _is_low_signal(query_text):
        bypass_reason = "LOW_SIGNAL_BYPASS"

    if bypass_reason is not None:
        if bypass_reason == "GREETING_BYPASS":
            answer_text = "Halo! Saya Humbet AI. Silakan ajukan pertanyaan terkait materi atau dokumen kamu, nanti saya bantu mencarikan jawabannya."
        else:
            answer_text = "Sepertinya pesanmu belum cukup jelas. Coba jelaskan pertanyaanmu lebih lengkap agar saya bisa membantu dengan tepat."
        sources: list[dict] = []
        iterations = 0
        confidence = 1.0
        trace_models: list[DebugTrace] = []
        debug_logs = {
            "rq_rag": {
                "refined_query": query_text,
                "sub_queries": [],
                "docs_retrieved": 0,
                "source_per_query": {},
                "refinement_type": bypass_reason,
            },
            "iterations": [],
            "final_status": {
                "stop_reason": bypass_reason,
                "is_fallback": False,
                "entropy_history": [],
            },
        }

        logger.info(
            "Chat bypassed RAG",
            extra={
                "query": query_text,
                "reason": bypass_reason,
            },
        )
        broadcast_event(
            stage="final_status",
            action="bypass",
            summary="Chat bypassed RAG pipeline",
            details={
                "stop_reason": bypass_reason,
                "query": query_text,
            },
        )
    else:
        rag_result = await asyncio.to_thread(run_rag_pipeline, query_text, effective_role)

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

        answer_text = rag_result.answer
        sources = rag_result.sources
        iterations = rag_result.iterations
        confidence = rag_result.confidence
        debug_logs = rag_result.debug_logs

        logger.info(
            "Chat request handled",
            extra={
                "query": query_text,
                "iterations": iterations,
                "confidence": confidence,
            },
        )
    try:
        if payload.conversation_id is None:
            title = query_text.strip()
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
            content=query_text,
        )

        assistant_message = await add_message(
            session=session,
            conversation_id=conversation.id,
            role="assistant",
            content=answer_text,
            confidence=confidence,
            rag_iterations=iterations,
        )

        context_records = []
        seen_keys = set()
        for src in sources:
            source = src.get("source")
            chunk_id = src.get("chunk_id")
            key = (source, chunk_id)
            if not source or key in seen_keys:
                continue
            seen_keys.add(key)
            docs = vector_store_manager.get_documents_by_source(source)
            for _, doc in docs:
                meta_chunk = doc.metadata.get("chunk_id")
                if chunk_id is not None and meta_chunk != chunk_id:
                    continue
                context_records.append(
                    AnswerContext(
                        message_id=assistant_message.id,
                        source=source,
                        chunk_id=chunk_id,
                        content=doc.page_content,
                    )
                )
                if chunk_id is not None:
                    break

        if context_records:
            session.add_all(context_records)

        await session.commit()
        conversation_id = conversation.id
    except Exception as exc:
        logger.error(f"Gagal menyimpan riwayat percakapan: {exc}")
        await session.rollback()
        conversation_id = payload.conversation_id or -1

    return ChatResponseWithTrace(
        answer=answer_text,
        sources=sources,
        iterations=iterations,
        confidence=confidence,
        conversation_id=conversation_id,
        trace=trace_models,
        debug_logs=debug_logs,
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
