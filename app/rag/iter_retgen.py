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


def run_rag_pipeline(query: str) -> RAGResult:
    settings = get_settings()

    if settings.rag_mode == "naive":
        refined: RefinedQuery = refine_query(query)
        decision = decide_retrieval(refined["refined_query"])
        if decision.retrieve:
            documents: List[Document] = retrieve_documents(refined["refined_query"])
        else:
            documents = []
        answer, sources = generate_answer(query, documents)
        trace = IterationTrace(
            iteration=1,
            refined_query=refined["refined_query"],
            num_documents=len(documents),
            decision=decision,
        )
        traces = [trace]
        confidence = trace.decision.confidence
        iterations = 1
    elif settings.rag_mode == "advanced":
        refined: RefinedQuery = refine_query(query)
        decision = decide_retrieval(refined["refined_query"])
        if decision.retrieve:
            documents = retrieve_documents(refined["refined_query"])
        else:
            documents = []
        answer, sources = generate_answer(query, documents)
        trace = IterationTrace(
            iteration=1,
            refined_query=refined["refined_query"],
            num_documents=len(documents),
            decision=decision,
        )
        traces = [trace]
        confidence = min(1.0, trace.decision.confidence + 0.05)
        iterations = 1
    else:
        traces: List[IterationTrace] = []
        last_documents: List[Document] = []
        last_decision: RetrievalDecision | None = None
        last_refined: RefinedQuery | None = None

        for i in range(1, settings.max_iterations + 1):
            refined = refine_query(query)
            decision = decide_retrieval(refined["refined_query"])
            if decision.retrieve:
                documents = retrieve_documents(refined["refined_query"])
            else:
                documents = []

            trace = IterationTrace(
                iteration=i,
                refined_query=refined["refined_query"],
                num_documents=len(documents),
                decision=decision,
            )
            traces.append(trace)
            last_documents = documents
            last_decision = decision
            last_refined = refined

            if len(refined["refined_query"].split()) > 5 and decision.confidence > 0.7:
                break

        answer, sources = generate_answer(query, last_documents)
        iterations = len(traces)
        confidence = last_decision.confidence if last_decision is not None else 0.5

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
