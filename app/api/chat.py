from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio

from app.core.logging import get_logger
from app.rag.rag_pipeline import run_rag_pipeline
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponseWithTrace,
    DebugTrace,
)
from app.core.logging import subscribe_to_logs, unsubscribe_from_logs


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponseWithTrace, response_model_exclude_none=True)
async def chat_endpoint(payload: ChatRequest) -> ChatResponseWithTrace:
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

    return ChatResponseWithTrace(
        answer=rag_result.answer,
        sources=rag_result.sources,
        iterations=rag_result.iterations,
        confidence=rag_result.confidence,
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
