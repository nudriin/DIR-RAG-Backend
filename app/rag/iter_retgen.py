from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.dynamic_decision import RetrievalDecision, decide_retrieval_dragin
from app.rag.generator import generate_answer, generate_paragraph
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

    # Controller state
    entropy_history: List[float] = []
    current_instruction = rq["refined_query"]
    # Pre-prune anchors to keep context small initially
    all_documents: List[Document] = prune_context(semantic_anchor_docs, current_draft=current_instruction)
    paragraphs: List[str] = []
    final_sources: List[Dict[str, Any]] = []
    stop_reason = "Initialized"

    if settings.enable_dragin:
        gate_decision = decide_retrieval_dragin(rq["refined_query"])
        entropy_history.append(gate_decision.entropy)

        if not gate_decision.retrieve:
            answer_text, sources_from_answer = generate_answer(query=query, documents=all_documents)

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
                    "dragin": {
                        "entropy": float(gate_decision.entropy),
                        "triggered_retrieval": False,
                        "reason": gate_decision.reason,
                    },
                    "iter_retgen": {
                        "iter_query": rq["refined_query"],
                        "current_draft": answer_text,
                        "new_docs_found": 0,
                        "pruning_discarded": 0,
                        "pruning_kept": int(len(all_documents)),
                        "executing": "Direct Answer",
                    },
                    "etc": {
                        "current_trend": "start",
                    },
                }
            )

            final_answer = answer_text.strip()
            final_sources = sources_from_answer

            stop_reason = "Direct Answer"

        else:
            max_iter = min(2, settings.max_iterations)
            for i in range(1, max_iter + 1):
                previous_output = "\n\n".join(paragraphs) if paragraphs else None

                current_instruction = rq["refined_query"]
                paragraph_text, _ = generate_paragraph(
                    query=current_instruction,
                    documents=all_documents,
                    previous_output=previous_output,
                )

                refined_next = refine_query(query=current_instruction, draft_answer=paragraph_text)
                expanded_queries = [refined_next["refined_query"]] + (refined_next.get("sub_queries", []) or [])

                gap_docs: List[Document] = []
                for qx in expanded_queries:
                    gap_docs.extend(retrieve_documents(qx))

                new_docs_found = len(gap_docs)
                if gap_docs:
                    all_documents.extend(gap_docs)

                before_prune = len(all_documents)
                all_documents = prune_context(all_documents, current_draft=paragraph_text)
                after_prune = len(all_documents)
                pruning_discarded = max(0, before_prune - after_prune)
                pruning_kept = after_prune

                decision_iter = decide_retrieval_dragin(refined_next["refined_query"])
                entropy_history.append(decision_iter.entropy)

                traces.append(
                    IterationTrace(
                        iteration=i,
                        refined_query=current_instruction,
                        num_documents=len(all_documents),
                        decision=decision_iter,
                        raw_query=query,
                    )
                )

                paragraphs.append(paragraph_text)

                current_trend = "start" if len(entropy_history) == 1 else (
                    "decreasing" if entropy_history[-1] < entropy_history[-2]
                    else "increasing" if entropy_history[-1] > entropy_history[-2]
                    else "stable"
                )

                debug_logs["iterations"].append(
                    {
                        "step": i,
                        "dragin": {
                            "entropy": float(decision_iter.entropy),
                            "triggered_retrieval": True,
                            "reason": "ITER-RETGEN retrieval",
                        },
                        "iter_retgen": {
                            "iter_query": current_instruction,
                            "current_draft": paragraph_text,
                            "new_docs_found": int(new_docs_found),
                            "pruning_discarded": int(pruning_discarded),
                            "pruning_kept": int(pruning_kept),
                            "executing": f"ITER-RETGEN {i} of {max_iter}",
                        },
                        "etc": {
                            "current_trend": current_trend,
                        },
                    }
                )

            final_answer, final_sources = generate_answer(query=query, documents=all_documents)
            stop_reason = "Iter-RetGen Completed"
    else:
        max_iter = min(2, settings.max_iterations)
        for i in range(1, max_iter + 1):
            previous_output = "\n\n".join(paragraphs) if paragraphs else None

            current_instruction = rq["refined_query"]
            paragraph_text, _ = generate_paragraph(
                query=current_instruction,
                documents=all_documents,
                previous_output=previous_output,
            )

            refined_next = refine_query(query=current_instruction, draft_answer=paragraph_text)
            expanded_queries = [refined_next["refined_query"]] + (refined_next.get("sub_queries", []) or [])

            gap_docs: List[Document] = []
            for qx in expanded_queries:
                gap_docs.extend(retrieve_documents(qx))

            new_docs_found = len(gap_docs)
            if gap_docs:
                all_documents.extend(gap_docs)

            before_prune = len(all_documents)
            all_documents = prune_context(all_documents, current_draft=paragraph_text)
            after_prune = len(all_documents)
            pruning_discarded = max(0, before_prune - after_prune)
            pruning_kept = after_prune

            disabled_decision = RetrievalDecision(
                retrieve=False, confidence=1.0, reason="DRAGIN disabled", entropy=0.0, prompt=None
            )

            traces.append(
                IterationTrace(
                    iteration=i,
                    refined_query=current_instruction,
                    num_documents=len(all_documents),
                    decision=disabled_decision,
                    raw_query=query,
                )
            )

            paragraphs.append(paragraph_text)

            current_trend = "start" if len(entropy_history) == 0 else "stable"

            debug_logs["iterations"].append(
                {
                    "step": i,
                    "iter_retgen": {
                        "iter_query": current_instruction,
                        "current_draft": paragraph_text,
                        "new_docs_found": int(new_docs_found),
                        "pruning_discarded": int(pruning_discarded),
                        "pruning_kept": int(pruning_kept),
                        "executing": f"ITER-RETGEN {i} of {max_iter}",
                    },
                    "etc": {
                        "current_trend": current_trend,
                    },
                }
            )

        final_answer, final_sources = generate_answer(query=query, documents=all_documents)
        stop_reason = "Iter-RetGen Completed"

    # Final assembly
    if stop_reason != "Direct Answer":
        final_answer = "\n\n".join(paragraphs).strip()

    # Final Guardrail (Closed-Domain)
    if not has_valid_anchor or not final_answer or "Tidak ada informasi tersedia untuk pertanyaan tersebut" in final_answer:
        fallback_text = "Tidak ada informasi tersedia untuk pertanyaan tersebut"
        debug_logs["final_status"] = {
            "stop_reason": stop_reason,
            "is_fallback": True,
            "entropy_history": [float(x) for x in entropy_history],
        }
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
    return RAGResult(
        answer=final_answer,
        sources=final_sources,
        iterations=len(traces),
        confidence=last_confidence,
        traces=traces,
        debug_logs=debug_logs,
    )
