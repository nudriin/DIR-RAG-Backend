from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.dynamic_decision import RetrievalDecision, decide_retrieval_dragin
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
    raw_query: str


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    confidence: float
    traces: List[IterationTrace]


def run_rag_pipeline(query: str) -> RAGResult:
    settings = get_settings()
    traces: List[IterationTrace] = []

    # 1. GATEKEEPER: DRAGIN (Pengecekan awal ketidakpastian)
    # Dilakukan pada kueri asli sebelum masuk ke loop iteratif
    initial_decision = decide_retrieval_dragin(query)

    if not initial_decision.retrieve:
        # Jika model yakin, langsung jawab menggunakan pengetahuan internal
        docs = retrieve_documents(query)
        answer, sources = generate_answer(query, docs)
        return RAGResult(
            answer=answer,
            sources=sources,
            iterations=1,
            confidence=initial_decision.confidence,
            traces=[IterationTrace(1, query, 0, initial_decision, query)],
        )

    # 2. MODULE ITER-RETGEN (Triggered by DRAGIN)
    current_instruction = query  # Awalnya menggunakan kueri asli [cite: 26, 97]
    last_answer = ""
    all_documents = []

    # Jurnal menyebutkan T=2 biasanya paling optimal
    max_iter = settings.max_iterations

    for i in range(1, max_iter + 1):
        # A. RETRIEVAL: Mengambil informasi awal (atau tambahan) [cite: 26, 90]
        new_docs = retrieve_documents(current_instruction)
        all_documents.extend(new_docs)

        # B. GENERATE: Membuat draf jawaban berdasarkan semua dokumen [cite: 12, 26, 90]
        # Proses ini penting karena draf ini akan memandu iterasi berikutnya [cite: 11]
        last_answer, sources = generate_answer(current_instruction, all_documents)

        # C. TRACE: Catat progres iterasi
        current_trace = IterationTrace(
            iteration=i,
            refined_query=current_instruction,
            num_documents=len(new_docs),
            decision=(
                initial_decision
                if i == 1
                else RetrievalDecision(True, 1.0, "Iterative loop", 1.0, query)
            ),
            raw_query=query,
        )
        traces.append(current_trace)

        # D. STOPPING CONDITION: Cek kelengkapan (sesuai diagram Anda)
        # Jika jawaban sudah koheren/lengkap, hentikan loop
        # * if is_answer_sufficient(last_answer): # Fungsi pembantu untuk cek koherensi
        # *    break

        # E. MODULE RQ (Query Refinement): Setelah generate draf jawaban
        # Menggunakan draf jawaban untuk memperkuat kueri iterasi berikutnya [cite: 89, 192]
        if i < max_iter:
            refined_data = refine_query(query, draft_answer=last_answer)
            # Update instruksi untuk iterasi selanjutnya (Rewriting/Decomposition) [cite: 311, 435]
            current_instruction = refined_data["refined_query"]

    return RAGResult(
        answer=last_answer,
        sources=sources,
        iterations=len(traces),
        confidence=initial_decision.confidence,
        traces=traces,
    )
