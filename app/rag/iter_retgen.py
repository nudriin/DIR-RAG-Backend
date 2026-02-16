from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event
from app.rag.dynamic_decision import RetrievalDecision, decide_retrieval_dragin
from app.rag.generator import generate_answer
from app.rag.query_refinement import RefinedQuery, refine_query
from app.rag.retriever import (
    retrieve_documents,
    retrieve_documents_with_scores,
    rerank_documents,
)


logger = get_logger(__name__)


@dataclass
class IterationTrace:
    iteration: int
    refined_query: str
    num_documents: int
    decision: RetrievalDecision
    raw_query: str


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    confidence: float
    traces: List[IterationTrace]
    debug_logs: Dict[str, Any]


def prune_context(all_documents: List[Document], current_draft: str) -> List[Document]:
    if not all_documents:
        return []
    pruned_docs = rerank_documents(query=current_draft, documents=all_documents, top_n=5)
    return pruned_docs

def is_answer_covering_subqueries(answer_text: str, sub_queries: List[str]) -> bool:
    if not answer_text or not sub_queries:
        return False
    ans = answer_text.lower()
    satisfied = 0
    for sq in sub_queries:
        toks = [t for t in sq.lower().split() if len(t) > 3]
        hit = sum(1 for t in toks if t in ans)
        if hit >= max(1, len(toks) // 4):
            satisfied += 1
    return satisfied >= len(sub_queries)

def run_rag_pipeline(query: str) -> RAGResult:
    settings = get_settings()
    traces: List[IterationTrace] = []
    debug_logs: Dict[str, Any] = {
        "rq_rag": {},
        "iterations": [],
        "final_status": {},
    }

    # Phase 1 — The Primer (RQ-RAG)
    rq: RefinedQuery = refine_query(query)
    sub_queries: List[str] = rq.get("sub_queries", []) or []
    instruction_queue: List[str] = [rq["refined_query"]] + sub_queries

    anchor_results = []
    for sq in instruction_queue:
        anchor_results.extend(
            retrieve_documents_with_scores(query=sq, top_k=settings.similarity_top_k)
        )
    semantic_anchor_docs: List[Document] = [doc for doc, _ in anchor_results]
    source_names = [doc.metadata.get("source") for doc, _ in anchor_results if doc.metadata.get("source")]
    unique_source_names = sorted(set(source_names))

    # Closed-domain guardrail will use these scores
    RELEVANCE_DISTANCE_THRESHOLD = 1.2  # for cosine distance in Chroma (lower is better)
    has_valid_anchor = bool(anchor_results) and any(score <= RELEVANCE_DISTANCE_THRESHOLD for _, score in anchor_results)

    debug_logs["rq_rag"] = {
        "refined_query": rq["refined_query"],
        "sub_queries": sub_queries,
        "docs_retrieved": len(semantic_anchor_docs),
        "source_names": unique_source_names,
        "refinement_type": rq.get("refinement_type", "REWRITE"),
    }
    broadcast_event(
        stage="retrieval",
        action="anchor_complete",
        summary="Retrieval anchor selesai",
        details={
            "instruction_count": len(instruction_queue),
            "docs_retrieved": len(semantic_anchor_docs),
            "sources": unique_source_names,
        },
    )

    # Controller state (non-iterative)
    entropy_history: List[float] = []
    current_instruction = rq["refined_query"]
    before_prune = len(semantic_anchor_docs)
    all_documents: List[Document] = prune_context(semantic_anchor_docs, current_draft=current_instruction)
    after_prune = len(all_documents)
    pruning_discarded = max(0, before_prune - after_prune)
    pruning_kept = after_prune
    final_sources: List[Dict[str, Any]] = []
    stop_reason = "Direct Answer"

    # DRAGIN gate logged for visibility only (no iterative RETGEN)
    if settings.enable_dragin:
        gate_decision = decide_retrieval_dragin(rq["refined_query"])
        entropy_history.append(gate_decision.entropy)
        broadcast_event(
            stage="dragin",
            action="gate",
            summary="Evaluasi kecukupan informasi",
            details={
                "retrieve": bool(gate_decision.retrieve),
                "entropy": float(gate_decision.entropy),
                "reason": gate_decision.reason,
            },
        )
    else:
        gate_decision = RetrievalDecision(
            retrieve=False, confidence=1.0, reason="DRAGIN disabled", entropy=0.0, prompt=None
        )

    final_answer, final_sources = generate_answer(query=query, documents=all_documents)
    broadcast_event(
        stage="generation",
        action="final_answer",
        summary="Jawaban akhir dihasilkan (non-iterative)",
        details={"num_sources": len(final_sources)},
    )

    traces.append(
        IterationTrace(
            iteration=1,
            refined_query=rq["refined_query"],
            num_documents=len(all_documents),
            decision=gate_decision,
            raw_query=query,
        )
    )

    debug_logs["iterations"].append(
        {
            "step": 1,
            **(
                {
                    "dragin": {
                        "entropy": float(gate_decision.entropy),
                        "triggered_retrieval": bool(gate_decision.retrieve),
                        "reason": gate_decision.reason,
                    }
                }
                if settings.enable_dragin
                else {}
            ),
            "iter_retgen": {
                "iter_query": rq["refined_query"],
                "current_draft": final_answer,
                "new_docs_found": 0,
                "pruning_discarded": int(pruning_discarded),
                "pruning_kept": int(pruning_kept),
                "executing": "Direct Answer",
            },
            "etc": {
                "current_trend": "n/a",
            },
        }
    )

    # Final assembly already built (non-iterative)

    # Final Guardrail (Closed-Domain)
    if not has_valid_anchor or not final_answer or "Tidak ada informasi tersedia untuk pertanyaan tersebut" in final_answer:
        fallback_text = "Tidak ada informasi tersedia untuk pertanyaan tersebut"
        debug_logs["final_status"] = {
            "stop_reason": stop_reason,
            "is_fallback": True,
            "entropy_history": [float(x) for x in entropy_history],
        }
        broadcast_event(
            stage="final_status",
            action="fallback",
            summary="Jawaban fallback karena konteks tidak valid",
            details={"stop_reason": stop_reason},
        )
        return RAGResult(
            answer=fallback_text,
            sources=[],
            iterations=len(traces) if traces else 1,
            confidence=0.0,
            traces=traces or [IterationTrace(1, rq["refined_query"], 0, RetrievalDecision(True, 0.0, "No valid context", 1.0, None), query)],
            debug_logs=debug_logs,
        )

    # Sources already built from generate_answer when applicable

    last_confidence = traces[-1].decision.confidence if traces else 0.0
    debug_logs["final_status"] = {
        "stop_reason": stop_reason,
        "is_fallback": False,
        "entropy_history": [float(x) for x in entropy_history],
    }
    broadcast_event(
        stage="final_status",
        action="complete",
        summary="Pipeline RAG selesai",
        details={"stop_reason": stop_reason, "iterations": len(traces)},
    )
    return RAGResult(
        answer=final_answer,
        sources=final_sources,
        iterations=len(traces),
        confidence=last_confidence,
        traces=traces,
        debug_logs=debug_logs,
    )
