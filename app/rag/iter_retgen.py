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


def prune_context(all_documents: List[Document], current_draft: str) -> List[Document]:
    if not all_documents:
        return []
    pruned_docs = rerank_documents(query=current_draft, documents=all_documents, top_n=5)
    return pruned_docs


def run_rag_pipeline(query: str) -> RAGResult:
    settings = get_settings()
    traces: List[IterationTrace] = []

    # Phase 1 — The Primer (RQ-RAG)
    rq: RefinedQuery = refine_query(query)
    sub_queries: List[str] = rq.get("sub_queries", []) or [rq["refined_query"]]

    anchor_results = []
    for sq in sub_queries:
        anchor_results.extend(
            retrieve_documents_with_scores(query=sq, top_k=settings.similarity_top_k)
        )
    semantic_anchor_docs: List[Document] = [doc for doc, _ in anchor_results]

    # Closed-domain guardrail will use these scores
    RELEVANCE_DISTANCE_THRESHOLD = 1.2  # for cosine distance in Chroma (lower is better)
    has_valid_anchor = bool(anchor_results) and any(score <= RELEVANCE_DISTANCE_THRESHOLD for _, score in anchor_results)

    # Controller state
    entropy_history: List[float] = []
    current_instruction = rq["refined_query"]
    all_documents: List[Document] = list(semantic_anchor_docs)
    paragraphs: List[str] = []
    final_sources: List[Dict[str, Any]] = []

    max_iter = settings.max_iterations
    for i in range(1, max_iter + 1):
        previous_output = "\n\n".join(paragraphs) if paragraphs else None

        # Generate next paragraph (ITER-RETGEN)
        paragraph_text, _ = generate_paragraph(
            query=current_instruction,
            documents=all_documents,
            previous_output=previous_output,
        )

        # DRAGIN: Decide if we need gap-filling retrieval
        decision = decide_retrieval_dragin(current_instruction)
        entropy_history.append(decision.entropy)

        if decision.retrieve:
            # Use RQ-RAG again to refine based on the newly generated paragraph
            refined_next = refine_query(query=current_instruction, draft_answer=paragraph_text)
            expanded_queries = [refined_next["refined_query"]] + (refined_next.get("sub_queries", []) or [])

            gap_docs: List[Document] = []
            for qx in expanded_queries:
                gap_docs.extend(retrieve_documents(qx))

            if gap_docs:
                all_documents.extend(gap_docs)
                # Regenerate paragraph with enriched context
                paragraph_text, _ = generate_paragraph(
                    query=current_instruction,
                    documents=all_documents,
                    previous_output=previous_output,
                )

        # Pruning: keep context window clean for next iteration
        all_documents = prune_context(all_documents, current_draft=paragraph_text)

        # Trace
        traces.append(
            IterationTrace(
                iteration=i,
                refined_query=current_instruction,
                num_documents=len(all_documents),
                decision=decision,
                raw_query=query,
            )
        )

        paragraphs.append(paragraph_text)

        # ETC stopping condition: stop if entropy trend is flat or rising
        if len(entropy_history) > 1:
            prev_entropy = entropy_history[-2]
            curr_entropy = entropy_history[-1]
            if curr_entropy >= prev_entropy:
                logger.info(f"ETC Triggered: Entropy stabilized at {curr_entropy}")
                break

        # Prepare next instruction using refined query with accumulated output
        current_instruction = refine_query(query, draft_answer="\n\n".join(paragraphs))["refined_query"]

    # Final assembly
    final_answer = "\n\n".join(paragraphs).strip()

    # Final Guardrail (Closed-Domain)
    if not has_valid_anchor or not final_answer or "Tidak ada informasi tersedia untuk pertanyaan tersebut" in final_answer:
        fallback_text = "Tidak ada informasi tersedia untuk pertanyaan tersebut"
        return RAGResult(
            answer=fallback_text,
            sources=[],
            iterations=len(traces) if traces else 1,
            confidence=0.0,
            traces=traces or [IterationTrace(1, rq["refined_query"], 0, RetrievalDecision(True, 0.0, "No valid context", 1.0, None), query)],
        )

    # Build sources from pruned context
    for i, doc in enumerate(all_documents):
        final_sources.append(
            {
                "id": i,
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )

    last_confidence = traces[-1].decision.confidence if traces else 0.0
    return RAGResult(
        answer=final_answer,
        sources=final_sources,
        iterations=len(traces),
        confidence=last_confidence,
        traces=traces,
    )
