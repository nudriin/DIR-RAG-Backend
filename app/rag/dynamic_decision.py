from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RetrievalDecision:
    retrieve: bool
    confidence: float
    reason: str


def estimate_uncertainty(query: str) -> float:
    tokens = query.split()
    length = len(tokens)
    if length <= 3:
        return 0.8
    if length >= 20:
        return 0.3
    return 0.8 - (length - 3) * (0.5 / 17)


def decide_retrieval(query: str) -> RetrievalDecision:
    settings = get_settings()
    uncertainty = estimate_uncertainty(query)
    generic_keywords = ["apa", "siapa", "jelaskan", "definisi"]
    is_generic = any(token in query.lower() for token in generic_keywords)

    if uncertainty < 0.4 and is_generic:
        retrieve = False
        confidence = 1.0 - uncertainty
        reason = "Query generik dengan ketidakpastian model rendah, retrieval dapat dilewati."
    else:
        retrieve = True
        confidence = 1.0 - uncertainty
        reason = "Diperlukan retrieval untuk menurunkan halusinasi."

    logger.info(
        "Retrieval decision",
        extra={
            "query": query,
            "retrieve": retrieve,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "rag_mode": settings.rag_mode,
        },
    )

    return RetrievalDecision(retrieve=retrieve, confidence=confidence, reason=reason)

