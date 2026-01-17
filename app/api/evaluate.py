from typing import List

from fastapi import APIRouter

from app.core.logging import get_logger
from app.evaluation.ragas_eval import run_ragas_evaluation
from app.evaluation.retrieval_metrics import evaluate_retrieval_batch
from app.schemas.chat_schema import EvaluateRequest, EvaluateResponse


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(payload: EvaluateRequest) -> EvaluateResponse:
    questions: List[str] = payload.questions

    batch_relevant = payload.relevant_doc_ids or [[] for _ in questions]
    batch_retrieved = [[] for _ in questions]

    retrieval_scores = evaluate_retrieval_batch(
        batch_relevant=batch_relevant,
        batch_retrieved=batch_retrieved,
        k=5,
    )

    ragas_scores = run_ragas_evaluation(
        questions=questions,
        answers=payload.ground_truth_answers or [""] * len(questions),
        contexts=[[] for _ in questions],
    )

    logger.info(
        "Evaluation executed",
        extra={
            "num_questions": len(questions),
        },
    )

    return EvaluateResponse(
        hit_rate=retrieval_scores["hit_rate"],
        mrr=retrieval_scores["mrr"],
        ragas=ragas_scores,
    )

