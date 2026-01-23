from fastapi import APIRouter

from app.core.logging import get_logger
from app.rag.iter_retgen import run_rag_pipeline
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponseWithTrace,
    DebugTrace,
)


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponseWithTrace)
async def chat_endpoint(payload: ChatRequest) -> ChatResponseWithTrace:
    rag_result = run_rag_pipeline(payload.query)

    trace_models = [
        DebugTrace(
            iteration=t.iteration,
            refined_query=t.refined_query,
            num_documents=t.num_documents,
            retrieve=t.decision.retrieve,
            retrieval_confidence=t.decision.confidence,
            reason=t.decision.reason,
            raw_query=t.decision.prompt,
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
