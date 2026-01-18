from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.dynamic_decision import RetrievalDecision, decide_retrieval
from app.rag.generator import generate_answer
from app.rag.query_refinement import RefinedQuery, refine_query
from app.rag.retriever import retrieve_documents


logger = get_logger(__name__)


@dataclass
class IterationTrace:
    iteration: int
    refined_query: str
    num_documents: int
    decision: RetrievalDecision


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    confidence: float
    traces: List[IterationTrace]


def _run_single_iteration(
    query: str,
    iteration: int,
) -> tuple[str, List[Dict[str, Any]], IterationTrace]:
    refined: RefinedQuery = refine_query(query)
    decision = decide_retrieval(refined["refined_query"])

    if decision.retrieve:
        documents: List[Document] = retrieve_documents(refined["refined_query"])
    else:
        documents = []

    answer, sources = generate_answer(query, documents)

    trace = IterationTrace(
        iteration=iteration,
        refined_query=refined["refined_query"],
        num_documents=len(documents),
        decision=decision,
    )

    return answer, sources, trace


def run_rag_pipeline(query: str) -> RAGResult:
    settings = get_settings()

    if settings.rag_mode == "naive":
        answer, sources, trace = _run_single_iteration(query, iteration=1)
        traces = [trace]
        confidence = trace.decision.confidence
        iterations = 1
    elif settings.rag_mode == "advanced":
        answer, sources, trace = _run_single_iteration(query, iteration=1)
        traces = [trace]
        confidence = min(1.0, trace.decision.confidence + 0.05)
        iterations = 1
    else:
        traces: List[IterationTrace] = []
        last_answer = ""
        last_sources: List[Dict[str, Any]] = []
        for i in range(1, settings.max_iterations + 1):
            answer, sources, trace = _run_single_iteration(query, iteration=i)
            traces.append(trace)
            last_answer = answer
            last_sources = sources
            if len(trace.refined_query.split()) > 5 and trace.decision.confidence > 0.7:
                break
        answer = last_answer
        sources = last_sources
        iterations = len(traces)
        confidence = traces[-1].decision.confidence if traces else 0.5

    logger.info(
        "RAG pipeline finished",
        extra={
            "rag_mode": settings.rag_mode,
            "iterations": iterations,
            "confidence": confidence,
        },
    )

    return RAGResult(
        answer=answer,
        sources=sources,
        iterations=iterations,
        confidence=confidence,
        traces=traces,
    )
