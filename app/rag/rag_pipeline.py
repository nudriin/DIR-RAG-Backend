"""
RAG Pipeline — RQ-RAG + DRAGIN Reasoning Loop

Arsitektur:
    1. User Query → RQ-RAG (refine_query) → refined_query + sub_queries
    2. Anchor retrieval + rerank (dengan refined_query sebagai query reranker)
    3. Reasoning Loop (max 2 iterasi):
        a. DRAGIN: generate jawaban + hitung entropy dari logprobs
        b. Jika entropy ≤ threshold → return jawaban (confident)
        c. Jika entropy > threshold → RQ-RAG re-refine → gap retrieval → loop
    4. Jika loop habis (max iterasi) → return jawaban terakhir sebagai final
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.dynamic_decision import DRAGINResult, generate_with_dragin
from app.rag.query_refinement import RefinedQuery, refine_query
from app.rag.retriever import (
    retrieve_documents,
    retrieve_documents_with_scores,
    rerank_documents,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (kompatibel dengan chat_schema.py)
# ---------------------------------------------------------------------------

@dataclass
class IterationTrace:
    iteration: int
    refined_query: str
    num_documents: int
    decision: DRAGINResult  # Ganti RetrievalDecision → DRAGINResult
    raw_query: str


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    confidence: float
    traces: List[IterationTrace]
    debug_logs: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prune_context(
    all_documents: List[Document],
    query: str,
    top_n: int = 5,
) -> List[Document]:
    """Prune konteks menggunakan query (bukan draft answer) sebagai reranking query."""
    if not all_documents:
        return []
    return rerank_documents(query=query, documents=all_documents, top_n=top_n)


def _build_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """Ekstrak metadata sumber dari dokumen."""
    sources: List[Dict[str, Any]] = []
    seen = set()
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source")
        chunk_id = doc.metadata.get("chunk_id")
        key = (source, chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append({"id": i, "source": source, "chunk_id": chunk_id})
    return sources


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def run_rag_pipeline(query: str) -> RAGResult:
    """
    RAG Pipeline dengan RQ-RAG + DRAGIN Reasoning Loop.

    Alur:
        1. RQ-RAG: refine query → refined_query + sub_queries
        2. Anchor retrieval untuk semua (sub-)query
        3. Closed-domain guardrail (cek relevansi)
        4. Reasoning loop (max dragin_max_iterations):
           DRAGIN generate → cek entropy → re-refine jika perlu → loop
        5. Return jawaban final
    """
    settings = get_settings()
    traces: List[IterationTrace] = []
    debug_logs: Dict[str, Any] = {
        "rq_rag": {},
        "iterations": [],
        "final_status": {},
    }

    # ======================================================================
    # PHASE 1 — RQ-RAG: Query Refinement
    # ======================================================================
    rq: RefinedQuery = refine_query(query)
    sub_queries: List[str] = rq.get("sub_queries", []) or []
    search_queries: List[str] = [rq["refined_query"]] + sub_queries

    logger.info(
        "RQ-RAG refinement complete",
        extra={
            "refinement_type": rq.get("refinement_type", "REWRITE"),
            "num_sub_queries": len(sub_queries),
        },
    )

    # ======================================================================
    # PHASE 2 — Anchor Retrieval + Rerank (track per query)
    # ======================================================================
    anchor_results = []
    source_per_query: Dict[str, List[str]] = {}

    # Retrieve per query dan track source per query
    for i, sq in enumerate(search_queries):
        results = retrieve_documents_with_scores(query=sq, top_k=settings.similarity_top_k)
        anchor_results.extend(results)

        # Label: "refined_query" atau "sub_query_1", "sub_query_2", ...
        if i == 0:
            label = "refined_query"
        else:
            label = f"sub_query_{i}"

        source_per_query[label] = sorted(set(
            doc.metadata.get("source", "unknown")
            for doc, _ in results
            if doc.metadata.get("source")
        ))

    # Deduplikasi dan rerank menggunakan refined_query
    raw_anchor_docs: List[Document] = [doc for doc, _ in anchor_results]
    all_documents = _prune_context(raw_anchor_docs, query=rq["refined_query"])

    debug_logs["rq_rag"] = {
        "refined_query": rq["refined_query"],
        "sub_queries": sub_queries,
        "docs_retrieved": len(raw_anchor_docs),
        "source_per_query": source_per_query,
        "refinement_type": rq.get("refinement_type", "REWRITE"),
    }

    # ======================================================================
    # PHASE 3 — Closed-Domain Guardrail
    # ======================================================================
    RELEVANCE_DISTANCE_THRESHOLD = 1.2
    has_valid_anchor = bool(anchor_results) and any(
        score <= RELEVANCE_DISTANCE_THRESHOLD for _, score in anchor_results
    )

    if not has_valid_anchor:
        fallback_text = "Tidak ada informasi tersedia untuk pertanyaan tersebut"
        debug_logs["final_status"] = {
            "stop_reason": "No relevant anchor documents",
            "is_fallback": True,
            "entropy_history": [],
        }

        fallback_dragin = DRAGINResult(
            answer_text=fallback_text,
            entropy=1.0,
            confidence=0.0,
            should_retry=False,
            reason="No valid context",
            token_count=0,
        )

        return RAGResult(
            answer=fallback_text,
            sources=[],
            iterations=1,
            confidence=0.0,
            traces=[IterationTrace(1, rq["refined_query"], 0, fallback_dragin, query)],
            debug_logs=debug_logs,
        )

    # ======================================================================
    # PHASE 4 — Reasoning Loop: DRAGIN Generate + Evaluate
    # ======================================================================
    max_dragin_iter = settings.dragin_max_iterations  # default 2
    entropy_history: List[float] = []
    current_query = rq["refined_query"]
    final_answer = ""
    final_sources: List[Dict[str, Any]] = []

    for iteration in range(1, max_dragin_iter + 1):
        logger.info(f"Reasoning loop iteration {iteration}/{max_dragin_iter}")

        # --- DRAGIN: Generate + Evaluate ---
        dragin_result = generate_with_dragin(
            query=current_query,
            documents=all_documents,
            sub_queries=sub_queries,
        )
        entropy_history.append(dragin_result.entropy)

        # Trace
        traces.append(
            IterationTrace(
                iteration=iteration,
                refined_query=current_query,
                num_documents=len(all_documents),
                decision=dragin_result,
                raw_query=query,
            )
        )

        # Entropy trend
        if len(entropy_history) == 1:
            current_trend = "start"
        elif entropy_history[-1] < entropy_history[-2]:
            current_trend = "decreasing"
        elif entropy_history[-1] > entropy_history[-2]:
            current_trend = "increasing"
        else:
            current_trend = "stable"

        debug_logs["iterations"].append({
            "step": iteration,
            "dragin": {
                "entropy": float(dragin_result.entropy),
                "confidence": float(dragin_result.confidence),
                "should_retry": dragin_result.should_retry,
                "reason": dragin_result.reason,
                "token_count": dragin_result.token_count,
            },
            "rq_dragin": {
                "iter_query": current_query,
                "current_draft": dragin_result.answer_text[:200],
                "new_docs_found": 0,
                "pruning_discarded": 0,
                "pruning_kept": len(all_documents),
                "executing": f"RQ-DRAGIN Reasoning Loop {iteration}/{max_dragin_iter}",
            },
            "etc": {
                "current_trend": current_trend,
            },
        })

        # --- Keputusan: lanjut atau berhenti ---
        if not dragin_result.should_retry:
            # Entropy rendah → jawaban cukup, langsung final
            final_answer = dragin_result.answer_text
            final_sources = _build_sources(all_documents)
            stop_reason = (
                f"Confident answer (entropy={dragin_result.entropy:.4f} "
                f"≤ threshold={settings.dragin_threshold})"
            )
            logger.info(f"Reasoning loop stopped: {stop_reason}")
            break

        # Entropy tinggi → cek apakah masih bisa iterasi
        if iteration >= max_dragin_iter:
            # Sudah max → paksa jawaban terakhir sebagai final
            final_answer = dragin_result.answer_text
            final_sources = _build_sources(all_documents)
            stop_reason = (
                f"Max iterations reached ({max_dragin_iter}), "
                f"final entropy={dragin_result.entropy:.4f}"
            )
            logger.info(f"Reasoning loop stopped: {stop_reason}")
            break

        # --- Re-refine via RQ-RAG (bridge semantic gaps) ---
        logger.info("Entropy tinggi, re-refine query via RQ-RAG...")
        refined_next = refine_query(
            query=current_query,
            draft_answer=dragin_result.answer_text,
        )
        expanded_queries = [refined_next["refined_query"]] + (
            refined_next.get("sub_queries", []) or []
        )

        # --- Gap Retrieval ---
        gap_docs: List[Document] = []
        for qx in expanded_queries:
            gap_docs.extend(retrieve_documents(qx))

        new_docs_found = len(gap_docs)
        if gap_docs:
            all_documents.extend(gap_docs)

        # --- Prune (dengan refined query, bukan draft) ---
        before_prune = len(all_documents)
        all_documents = _prune_context(
            all_documents,
            query=refined_next["refined_query"],
        )
        after_prune = len(all_documents)

        # Update debug log untuk iterasi ini
        debug_logs["iterations"][-1]["rq_dragin"]["new_docs_found"] = new_docs_found
        debug_logs["iterations"][-1]["rq_dragin"]["pruning_discarded"] = max(
            0, before_prune - after_prune
        )
        debug_logs["iterations"][-1]["rq_dragin"]["pruning_kept"] = after_prune

        # Update query untuk iterasi berikutnya
        current_query = refined_next["refined_query"]

    # ======================================================================
    # PHASE 5 — Final Assembly
    # ======================================================================
    if not final_answer or "Tidak ada informasi tersedia untuk pertanyaan tersebut" in final_answer:
        debug_logs["final_status"] = {
            "stop_reason": stop_reason,
            "is_fallback": True,
            "entropy_history": [float(x) for x in entropy_history],
        }
        return RAGResult(
            answer="Tidak ada informasi tersedia untuk pertanyaan tersebut",
            sources=[],
            iterations=len(traces),
            confidence=0.0,
            traces=traces,
            debug_logs=debug_logs,
        )

    last_confidence = traces[-1].decision.confidence if traces else 0.0
    debug_logs["final_status"] = {
        "stop_reason": stop_reason,
        "is_fallback": False,
        "entropy_history": [float(x) for x in entropy_history],
    }

    return RAGResult(
        answer=final_answer.strip(),
        sources=final_sources,
        iterations=len(traces),
        confidence=last_confidence,
        traces=traces,
        debug_logs=debug_logs,
    )
